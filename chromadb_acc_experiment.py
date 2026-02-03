"""
임베딩 양에 따른 ChromaDB HNSW 검색 Recall@K 평가 실험

이 스크립트는:
1. 각 컬렉션 크기별로 동일한 쿼리에 대해 Brute-Force 검색 수행 (정확한 결과)
2. HNSW 검색 수행 (근사 결과)
3. Recall@K 계산: HNSW 결과 중 Brute-Force Top-K에 포함된 비율
4. 데이터 양에 따른 Recall@K 변화 분석
"""

import os
import time
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Set
import statistics
import numpy as np

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import PyPDF2
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


class PDFProcessor:
    """PDF 파일 처리 클래스"""
    
    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path
        self.pages = []
        self.num_pages = 0
        
    def extract_text(self) -> List[str]:
        """PDF에서 페이지별 텍스트 추출"""
        try:
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                self.num_pages = len(pdf_reader.pages)
                
                pages_text = []
                for page_num in range(self.num_pages):
                    try:
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        if text and text.strip():  # 빈 페이지 제외
                            pages_text.append(text.strip())
                        else:
                            pages_text.append("")  # 빈 페이지도 포함
                    except Exception as e:
                        # 개별 페이지 읽기 실패 시 빈 텍스트로 처리
                        pages_text.append("")
                
                self.pages = pages_text
                return pages_text
        except Exception as e:
            print(f"PDF 읽기 오류 ({self.pdf_path.name}): {e}")
            return []


class RecallExperiment:
    """ChromaDB Recall@K 평가 실험 클래스"""
    
    def __init__(self, model_path: str, chroma_db_path: Path = None):
        """
        Args:
            model_path: ko-sbert-sts 모델 경로
            chroma_db_path: ChromaDB 저장 경로
        """
        self.model_path = model_path
        self.chroma_db_path = chroma_db_path or Path(__file__).parent / "chroma_db"
        
        # 모델 로드
        print("모델 로딩 중...")
        self.model = SentenceTransformer(model_path)
        print("모델 로딩 완료!")
        
        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
    def process_pdfs_and_create_collections(self, pdf_folder: Path = None,
                                           max_files: int = None, 
                                           pages_per_collection: int = 50) -> Dict[str, Dict]:
        """
        PDF 파일들을 순회하면서 일정 페이지마다 누적 컬렉션을 생성하고 임베딩 저장
        (각 컬렉션은 이전 컬렉션의 모든 데이터를 포함하는 누적 구조)
        
        Args:
            pdf_folder: PDF 파일들이 있는 폴더 (None이면 체크포인트만 사용)
            max_files: 최대 처리할 PDF 파일 수 (None이면 전체)
            pages_per_collection: 컬렉션당 페이지 수 (이 페이지 수마다 새 컬렉션 생성)
        
        Returns:
            컬렉션 정보 딕셔너리 {collection_name: {num_documents, total_pages}}
        """
        print("\nPDF 파일 처리 및 누적 컬렉션 생성 중...")
        print(f"컬렉션당 페이지 수: {pages_per_collection}페이지 (누적 구조)")
        
        # PDF 파일 목록 (체크포인트가 없을 때만 필요)
        pdf_files = []
        if pdf_folder:
            pdf_folder = Path(pdf_folder)
            pdf_files = list(pdf_folder.glob("*.pdf"))
        
            if max_files:
                pdf_files = pdf_files[:max_files]
            print(f"총 {len(pdf_files)}개 PDF 파일 처리 예정\n")
        else:
            print("PDF 폴더가 지정되지 않았습니다. 체크포인트 파일만 사용합니다.\n")
        
        # 체크포인트 파일 경로
        checkpoint_dir = Path(__file__).parent / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_file = checkpoint_dir / "extracted_texts.pkl"
        
        # 기존 추출 데이터가 있으면 로드, 없으면 새로 추출
        if checkpoint_file.exists():
            print("기존 추출 데이터 발견! 로드 중...")
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                all_texts = checkpoint_data['texts']
                all_metadatas = checkpoint_data['metadatas']
                all_ids = checkpoint_data['ids']
                total_pages_processed = len(all_texts)
            print(f"로드 완료: {total_pages_processed}개 페이지")
        else:
            # 체크포인트가 없으면 PDF 폴더가 필요함
            if not pdf_folder or not pdf_files:
                raise ValueError("체크포인트 파일이 없고 PDF 폴더도 지정되지 않았습니다. "
                               "체크포인트 파일을 생성하거나 PDF 폴더를 지정해주세요.")
            
            # 누적 구조를 위한 전체 페이지 리스트
            all_texts = []
            all_metadatas = []
            all_ids = []
            
            # PDF 처리: 모든 페이지를 누적 리스트에 저장
            total_pages_processed = 0
            for pdf_file in tqdm(pdf_files, desc="PDF 처리"):
                processor = PDFProcessor(pdf_file)
                pages_text = processor.extract_text()
                
                if not pages_text or processor.num_pages == 0:
                    continue
                
                # 각 페이지를 누적 리스트에 추가
                for page_idx, page_text in enumerate(pages_text):
                    if not page_text.strip():  # 빈 텍스트 제외
                        continue
                    
                    all_texts.append(page_text)
                    all_metadatas.append({
                        "pdf_name": pdf_file.stem,
                        "page_num": page_idx + 1,
                        "total_pages": processor.num_pages
                    })
                    all_ids.append(f"{pdf_file.stem}_page_{page_idx + 1}")
                    total_pages_processed += 1
            
            # 추출 데이터 저장
            print(f"\n총 {total_pages_processed}개 페이지 추출 완료")
            print("추출 데이터 저장 중...")
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({
                    'texts': all_texts,
                    'metadatas': all_metadatas,
                    'ids': all_ids
                }, f)
            print("저장 완료!")
        
        print(f"\n누적 컬렉션 생성 시작...\n")
        
        # 기존 컬렉션 확인 및 최대 페이지 수 찾기
        existing_collections = set()
        max_existing_pages = 0
        max_collection_num = 0
        try:
            collections = self.client.list_collections()
            for col in collections:
                existing_collections.add(col.name)
                try:
                    # 컬렉션 이름에서 페이지 수 추출 (예: collection_0016_16000pages)
                    if '_' in col.name and 'pages' in col.name:
                        parts = col.name.split('_')
                        if len(parts) >= 3:
                            pages_str = parts[2].replace('pages', '')
                            try:
                                pages = int(pages_str)
                                if pages > max_existing_pages:
                                    max_existing_pages = pages
                                    # 컬렉션 번호도 추출
                                    try:
                                        collection_num_str = parts[1]
                                        max_collection_num = int(collection_num_str)
                                    except:
                                        pass
                            except:
                                pass
                except:
                    pass
        except:
            pass
        
        if max_existing_pages > 0:
            print(f"기존 컬렉션 발견: 최대 {max_existing_pages}개 페이지까지 생성됨")
            print(f"{max_existing_pages}개 페이지 이후부터 계속 생성합니다.\n")
        else:
            print("기존 컬렉션이 없습니다. 처음부터 생성합니다.\n")
        
        # 누적 컬렉션 생성: pages_per_collection마다 새로운 컬렉션 생성 (이전까지의 모든 페이지 포함)
        collections_info = {}
        # 기존 컬렉션 정보도 추가
        for col_name in existing_collections:
            try:
                collection = self.client.get_collection(col_name)
                count = collection.count()
                if count > 0:
                    collections_info[col_name] = {
                        "num_documents": count,
                        "collection_name": col_name
                    }
            except:
                pass
        
        # 시작 인덱스: 기존 최대 페이지 수 이후부터 시작
        start_idx = max_existing_pages
        if start_idx == 0:
            start_idx = pages_per_collection
        
        # 시작 컬렉션 번호: 기존 최대 컬렉션 번호 + 1
        collection_num = max_collection_num + 1 if max_collection_num > 0 else 1
        
        # 시작 인덱스를 pages_per_collection의 배수로 조정
        if start_idx % pages_per_collection != 0:
            start_idx = ((start_idx // pages_per_collection) + 1) * pages_per_collection
        
        print(f"컬렉션 생성 시작: {start_idx}개 페이지부터, 컬렉션 번호: {collection_num:04d}\n")
        
        for end_idx in range(start_idx, len(all_texts) + 1, pages_per_collection):
            # 현재까지의 모든 페이지를 포함하는 컬렉션 생성
            collection_name = f"collection_{collection_num:04d}_{end_idx}pages"
            
            # 이미 존재하는 컬렉션은 건너뛰기
            if collection_name in existing_collections:
                print(f"[{collection_name}] 이미 존재함 - 건너뜀")
                try:
                    collection = self.client.get_collection(collection_name)
                    collections_info[collection_name] = {
                        "num_documents": collection.count(),
                        "collection_name": collection_name
                    }
                except:
                    pass
                collection_num += 1
                continue
            
            try:
                self.client.delete_collection(collection_name)
            except:
                pass
            
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"total_pages": end_idx}
            )
            
            # 처음부터 현재까지의 모든 페이지를 컬렉션에 저장
            collection_texts = all_texts[:end_idx]
            collection_metadatas = all_metadatas[:end_idx]
            collection_ids = all_ids[:end_idx]
            
            print(f"[{collection_name}] {end_idx}개 페이지 저장 중...")
            self._save_collection(collection, collection_texts, 
                                collection_metadatas, collection_ids, collections_info)
            
            collection_num += 1
        
        # 마지막으로 전체 페이지를 포함하는 컬렉션 생성 (나머지가 있는 경우)
        # 단, 이미 생성된 최대 페이지 수보다 큰 경우에만
        if len(all_texts) > max_existing_pages:
            # 마지막 컬렉션이 pages_per_collection의 배수가 아닌 경우
            if len(all_texts) % pages_per_collection != 0:
                end_idx = len(all_texts)
                collection_name = f"collection_{collection_num:04d}_{end_idx}pages"
                
                # 이미 존재하는 컬렉션은 건너뛰기
                if collection_name in existing_collections:
                    print(f"[{collection_name}] 이미 존재함 - 건너뜀")
                    try:
                        collection = self.client.get_collection(collection_name)
                        collections_info[collection_name] = {
                            "num_documents": collection.count(),
                            "collection_name": collection_name
                        }
                    except:
                        pass
                else:
                    try:
                        self.client.delete_collection(collection_name)
                    except:
                        pass
                    
                    collection = self.client.create_collection(
                        name=collection_name,
                        metadata={"total_pages": end_idx}
                    )
                    
                    print(f"[{collection_name}] {end_idx}개 페이지 저장 중...")
                    self._save_collection(collection, all_texts, 
                                        all_metadatas, all_ids, collections_info)
            # 마지막 컬렉션이 pages_per_collection의 배수인 경우, 전체 페이지 컬렉션도 생성
            elif len(all_texts) > 0:
                end_idx = len(all_texts)
                collection_name = f"collection_{collection_num:04d}_{end_idx}pages"
                
                # 이미 존재하는 컬렉션은 건너뛰기
                if collection_name not in existing_collections:
                    try:
                        self.client.delete_collection(collection_name)
                    except:
                        pass
                    
                    collection = self.client.create_collection(
                        name=collection_name,
                        metadata={"total_pages": end_idx}
                    )
                    
                    print(f"[{collection_name}] {end_idx}개 페이지 저장 중...")
                    self._save_collection(collection, all_texts, 
                                        all_metadatas, all_ids, collections_info)
        
        print(f"\n총 {len(collections_info)}개 컬렉션 생성 완료")
        print(f"총 {total_pages_processed}개 페이지 처리")
        print("\n컬렉션별 페이지 수 (누적 구조):")
        for col_name, info in sorted(collections_info.items(), key=lambda x: x[1]['num_documents']):
            print(f"  {col_name}: {info['num_documents']}개 페이지")
        
        return collections_info
    
    def _save_collection(self, collection, texts: List[str], metadatas: List[Dict], 
                        ids: List[str], collections_info: Dict):
        """컬렉션에 임베딩 생성 및 저장"""
        if not texts:
            return
        
        # 텍스트를 문자열로 변환하고 필터링 (None이나 비문자열 제거)
        valid_texts = []
        valid_metadatas = []
        valid_ids = []
        
        def is_valid_text(text):
            """텍스트가 유효한 문자열인지 검증"""
            if text is None:
                return False
            try:
                # 문자열로 변환
                text_str = str(text)
                if not text_str or not text_str.strip():
                    return False
                # 유니코드 서로게이트 문자 제거 및 인코딩 가능한지 확인
                text_str = text_str.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
                # 문자열 타입 확인
                if not isinstance(text_str, str):
                    return False
                return True
            except:
                return False
        
        for text, metadata, doc_id in zip(texts, metadatas, ids):
            if not is_valid_text(text):
                continue
            # 안전하게 문자열로 변환
            try:
                text_str = str(text).encode('utf-8', errors='ignore').decode('utf-8', errors='ignore').strip()
                if not text_str:
                    continue
                valid_texts.append(text_str)
                valid_metadatas.append(metadata)
                valid_ids.append(doc_id)
            except:
                continue
        
        if not valid_texts:
            print(f"  경고: 유효한 텍스트가 없어 컬렉션을 건너뜁니다.")
            return
        
        # 배치로 임베딩 생성
        batch_size = 32
        embeddings_list = []
        failed_indices = set()  # 실패한 텍스트의 인덱스 추적
        
        for i in tqdm(range(0, len(valid_texts), batch_size), 
                     desc=f"  임베딩 생성 ({len(valid_texts)}개)", leave=False):
            batch_start = i
            batch_end = min(i + batch_size, len(valid_texts))
            batch_texts = valid_texts[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))
            
            # 추가 안전장치: 각 텍스트가 문자열인지 확인
            safe_batch = []
            safe_indices = []
            
            for idx, t in zip(batch_indices, batch_texts):
                if idx in failed_indices:
                    continue
                try:
                    # 유니코드 정규화
                    safe_t = str(t).encode('utf-8', errors='ignore').decode('utf-8', errors='ignore').strip()
                    if safe_t and isinstance(safe_t, str):
                        safe_batch.append(safe_t)
                        safe_indices.append(idx)
                except:
                    failed_indices.add(idx)
                    continue
            
            if not safe_batch:
                continue
                
            try:
                embeddings = self.model.encode(safe_batch, show_progress_bar=False)
                embeddings_list.extend(embeddings.tolist())
            except Exception as e:
                print(f"  경고: 배치 {i//batch_size + 1} 임베딩 생성 실패: {str(e)[:100]}")
                # 실패한 배치의 텍스트를 개별적으로 처리
                for safe_text, idx in zip(safe_batch, safe_indices):
                    try:
                        embedding = self.model.encode([safe_text], show_progress_bar=False)
                        embeddings_list.extend(embedding.tolist())
                    except:
                        # 안전하게 출력
                        try:
                            preview = safe_text[:50].encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                            print(f"    텍스트 건너뜀: {preview}...")
                        except:
                            print(f"    텍스트 건너뜀: [인코딩 오류]")
                        failed_indices.add(idx)
        
        # 실패한 텍스트 제거
        if failed_indices:
            valid_texts = [t for idx, t in enumerate(valid_texts) if idx not in failed_indices]
            valid_metadatas = [m for idx, m in enumerate(valid_metadatas) if idx not in failed_indices]
            valid_ids = [id_val for idx, id_val in enumerate(valid_ids) if idx not in failed_indices]
        
        # 임베딩 수와 텍스트 수가 일치하는지 확인
        if len(embeddings_list) != len(valid_texts):
            print(f"  경고: 임베딩 수({len(embeddings_list)})와 텍스트 수({len(valid_texts)})가 일치하지 않습니다.")
            min_len = min(len(embeddings_list), len(valid_texts))
            embeddings_list = embeddings_list[:min_len]
            valid_texts = valid_texts[:min_len]
            valid_metadatas = valid_metadatas[:min_len]
            valid_ids = valid_ids[:min_len]
        
        # ChromaDB에 배치로 추가 (최대 배치 크기: 5000개로 제한)
        chroma_batch_size = 5000
        total_added = 0
        
        for i in range(0, len(valid_texts), chroma_batch_size):
            batch_end = min(i + chroma_batch_size, len(valid_texts))
            batch_embeddings = embeddings_list[i:batch_end]
            batch_texts = valid_texts[i:batch_end]
            batch_metadatas = valid_metadatas[i:batch_end]
            batch_ids = valid_ids[i:batch_end]
            
            collection.add(
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            total_added += len(batch_texts)
        
        collections_info[collection.name] = {
            "num_documents": total_added,
            "collection_name": collection.name
        }
    
    def get_existing_collections(self) -> Dict[str, Dict]:
        """이미 생성된 컬렉션 정보 가져오기 (0페이지 컬렉션 제외)"""
        collections_info = {}
        try:
            collections = self.client.list_collections()
            for col in collections:
                try:
                    collection = self.client.get_collection(col.name)
                    count = collection.count()
                    # 0페이지 컬렉션은 제외
                    if count > 0:
                        collections_info[col.name] = {
                            "num_documents": count,
                            "collection_name": col.name
                        }
                except:
                    continue
        except:
            pass
        return collections_info
    
    def brute_force_search(self, collection, query_embedding: np.ndarray, top_k: int) -> List[str]:
        """
        Brute-Force 검색: 모든 문서와의 거리를 계산하여 정확한 Top-K 반환
        
        Args:
            collection: ChromaDB 컬렉션
            query_embedding: 쿼리 임베딩 벡터
            top_k: 반환할 결과 개수
        
        Returns:
            Top-K 문서 ID 리스트
        """
        # 모든 문서 가져오기
        all_data = collection.get(include=['embeddings'])
        
        if not all_data['ids']:
            return []
        
        # 모든 임베딩과 거리 계산
        all_embeddings = np.array(all_data['embeddings'])
        query_vec = query_embedding.reshape(1, -1)
        
        # L2 거리 계산 (ChromaDB는 L2 distance 사용)
        distances = np.linalg.norm(all_embeddings - query_vec, axis=1)
        
        # 거리 순으로 정렬하여 Top-K 선택
        top_indices = np.argsort(distances)[:top_k]
        top_ids = [all_data['ids'][i] for i in top_indices]
        
        return top_ids
    
    def calculate_recall_at_k(self, ground_truth: List[str], retrieved: List[str], k: int) -> float:
        """
        Recall@K 계산
        
        Args:
            ground_truth: Brute-Force 검색 결과 (정답)
            retrieved: HNSW 검색 결과
            k: 평가할 K 값
        
        Returns:
            Recall@K 값 (0~1)
        """
        if not ground_truth or not retrieved:
            return 0.0
        
        # Top-K만 사용
        ground_truth_set = set(ground_truth[:k])
        retrieved_set = set(retrieved[:k])
        
        # 교집합 계산
        intersection = ground_truth_set & retrieved_set
        
        # Recall@K = 교집합 / ground_truth 개수
        recall = len(intersection) / len(ground_truth_set) if ground_truth_set else 0.0
        
        return recall
    
    def evaluate_recall(self, collections_info: Dict, test_queries: List[str],
                       top_k: int = 10, num_runs: int = 3) -> Dict[str, Dict]:
        """
        각 컬렉션별 Recall@K 평가
        
        Args:
            collections_info: 컬렉션 정보
            test_queries: 테스트 쿼리 리스트
            top_k: 검색 결과 개수
            num_runs: 각 쿼리당 반복 실행 횟수
        
        Returns:
            Recall 평가 결과 딕셔너리
        """
        print("\nRecall@K 평가 중...")
        print(f"Brute-Force vs HNSW 비교")
        
        results = {}
        
        # 페이지 수 순으로 정렬
        sorted_collections = sorted(collections_info.items(), 
                                   key=lambda x: x[1]['num_documents'])
        
        for collection_name, info in sorted_collections:
            collection = self.client.get_collection(collection_name)
            
            print(f"\n[{collection_name}] Recall@K 평가...")
            print(f"  컬렉션 내 페이지 수: {info['num_documents']}개")
            
            collection_results = {
                "num_documents": info["num_documents"],
                "collection_name": collection_name,
                "query_results": []
            }
            
            for query in test_queries:
                query_metrics = {
                    "query": query,
                    "runs": []
                }
                
                all_recalls_at_5 = []
                all_recalls_at_10 = []
                all_brute_force_times = []
                all_hnsw_times = []
                
                for run in range(num_runs):
                    # 쿼리 임베딩
                    query_embedding = self.model.encode([query])[0]
                    
                    # Brute-Force 검색 (정확한 결과)
                    brute_force_start = time.time()
                    brute_force_ids = self.brute_force_search(collection, query_embedding, top_k)
                    brute_force_time = time.time() - brute_force_start
                    
                    # HNSW 검색 (근사 결과)
                    hnsw_start = time.time()
                    hnsw_results = collection.query(
                        query_embeddings=[query_embedding.tolist()],
                        n_results=top_k,
                        include=['distances']
                    )
                    hnsw_time = time.time() - hnsw_start
                    hnsw_ids = hnsw_results['ids'][0]
                    
                    # Recall@K 계산 (여러 K 값에 대해)
                    recall_at_5 = self.calculate_recall_at_k(brute_force_ids, hnsw_ids, k=5)
                    recall_at_10 = self.calculate_recall_at_k(brute_force_ids, hnsw_ids, k=10)
                    
                    all_recalls_at_5.append(recall_at_5)
                    all_recalls_at_10.append(recall_at_10)
                    all_brute_force_times.append(brute_force_time)
                    all_hnsw_times.append(hnsw_time)
                    
                    query_metrics["runs"].append({
                        "brute_force_time": brute_force_time,
                        "hnsw_time": hnsw_time,
                        "recall_at_5": recall_at_5,
                        "recall_at_10": recall_at_10
                    })
                
                # 통계 계산
                query_metrics["avg_recall_at_5"] = np.mean(all_recalls_at_5)
                query_metrics["std_recall_at_5"] = np.std(all_recalls_at_5)
                query_metrics["avg_recall_at_10"] = np.mean(all_recalls_at_10)
                query_metrics["std_recall_at_10"] = np.std(all_recalls_at_10)
                query_metrics["avg_brute_force_time"] = np.mean(all_brute_force_times)
                query_metrics["avg_hnsw_time"] = np.mean(all_hnsw_times)
                
                collection_results["query_results"].append(query_metrics)
                
                print(f"    쿼리: '{query[:50]}...'")
                print(f"      Recall@5: {query_metrics['avg_recall_at_5']:.4f} ± {query_metrics['std_recall_at_5']:.4f}")
                print(f"      Recall@10: {query_metrics['avg_recall_at_10']:.4f} ± {query_metrics['std_recall_at_10']:.4f}")
                print(f"      Brute-Force 시간: {query_metrics['avg_brute_force_time']*1000:.2f}ms")
                print(f"      HNSW 시간: {query_metrics['avg_hnsw_time']*1000:.2f}ms")
            
            results[collection_name] = collection_results
        
        return results
    
    def visualize_results(self, results: Dict[str, Dict], output_path: Path = None):
        """결과 시각화"""
        if output_path is None:
            output_path = Path(__file__).parent / "experiment_results"
        output_path.mkdir(exist_ok=True)
        
        # 데이터 준비
        collection_names = []
        num_docs = []
        avg_recall_5 = []
        avg_recall_10 = []
        std_recall_5 = []
        std_recall_10 = []
        
        # 페이지 수 순으로 정렬
        sorted_results = sorted(results.items(), 
                               key=lambda x: x[1]['num_documents'])
        
        for collection_name, data in sorted_results:
            # 모든 쿼리의 평균 계산
            all_recall_5 = [q["avg_recall_at_5"] for q in data["query_results"]]
            all_recall_10 = [q["avg_recall_at_10"] for q in data["query_results"]]
            all_std_5 = [q["std_recall_at_5"] for q in data["query_results"]]
            all_std_10 = [q["std_recall_at_10"] for q in data["query_results"]]
            
            collection_names.append(f"{data['num_documents']}페이지")
            num_docs.append(data["num_documents"])
            avg_recall_5.append(np.mean(all_recall_5))
            avg_recall_10.append(np.mean(all_recall_10))
            std_recall_5.append(np.mean(all_std_5))
            std_recall_10.append(np.mean(all_std_10))
        
        # 그래프 1: Recall@K vs 페이지 수
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Recall@5
        axes[0].errorbar(num_docs, avg_recall_5, yerr=std_recall_5, 
                        marker='o', linewidth=2, markersize=8, capsize=5, label='Recall@5')
        axes[0].set_xlabel('페이지 수', fontsize=12)
        axes[0].set_ylabel('Recall@5', fontsize=12)
        axes[0].set_title('데이터 양에 따른 Recall@5 변화', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1.1])
        
        # Recall@10
        axes[1].errorbar(num_docs, avg_recall_10, yerr=std_recall_10,
                        marker='s', linewidth=2, markersize=8, capsize=5, label='Recall@10', color='orange')
        axes[1].set_xlabel('페이지 수', fontsize=12)
        axes[1].set_ylabel('Recall@10', fontsize=12)
        axes[1].set_title('데이터 양에 따른 Recall@10 변화', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(output_path / "recall_by_pages.png", dpi=300, bbox_inches='tight')
        print(f"\n그래프 저장: {output_path / 'recall_by_pages.png'}")
        
        # 그래프 2: Recall@5 vs Recall@10 비교
        fig, ax = plt.subplots(figsize=(14, 6))
        x_pos = np.arange(len(collection_names))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, avg_recall_5, width, yerr=std_recall_5,
                      label='Recall@5', alpha=0.8, capsize=5)
        bars2 = ax.bar(x_pos + width/2, avg_recall_10, width, yerr=std_recall_10,
                      label='Recall@10', alpha=0.8, capsize=5)
        
        ax.set_xlabel('컬렉션 (페이지 수)', fontsize=12)
        ax.set_ylabel('Recall', fontsize=12)
        ax.set_title('컬렉션별 Recall@K 비교 (Brute-Force vs HNSW)', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(collection_names, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        
        # 값 표시
        for i, (v1, v2) in enumerate(zip(avg_recall_5, avg_recall_10)):
            ax.text(i - width/2, v1 + std_recall_5[i] + 0.02, f'{v1:.3f}', 
                   ha='center', va='bottom', fontsize=8)
            ax.text(i + width/2, v2 + std_recall_10[i] + 0.02, f'{v2:.3f}', 
                   ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path / "recall_by_collection.png", dpi=300, bbox_inches='tight')
        print(f"그래프 저장: {output_path / 'recall_by_collection.png'}")
        
        plt.close('all')
    
    def save_results(self, results: Dict[str, Dict], output_path: Path = None):
        """결과를 JSON과 CSV로 저장"""
        if output_path is None:
            output_path = Path(__file__).parent / "experiment_results"
        output_path.mkdir(exist_ok=True)
        
        # JSON 저장
        json_path = output_path / "recall_results.json"
        
        # JSON 직렬화 가능한 형태로 변환
        json_results = {}
        for collection_name, data in results.items():
            json_results[collection_name] = {
                "num_documents": data["num_documents"],
                "queries": []
            }
            for query_data in data["query_results"]:
                json_results[collection_name]["queries"].append({
                    "query": query_data["query"],
                    "avg_recall_at_5": float(query_data["avg_recall_at_5"]),
                    "std_recall_at_5": float(query_data["std_recall_at_5"]),
                    "avg_recall_at_10": float(query_data["avg_recall_at_10"]),
                    "std_recall_at_10": float(query_data["std_recall_at_10"]),
                    "avg_brute_force_time_ms": float(query_data["avg_brute_force_time"] * 1000),
                    "avg_hnsw_time_ms": float(query_data["avg_hnsw_time"] * 1000)
                })
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n결과 저장: {json_path}")
        
        # CSV 저장 (요약)
        csv_data = []
        for collection_name, data in results.items():
            for query_data in data["query_results"]:
                csv_data.append({
                    "컬렉션명": collection_name,
                    "페이지_수": data["num_documents"],
                    "쿼리": query_data["query"],
                    "Recall_5_평균": query_data["avg_recall_at_5"],
                    "Recall_5_표준편차": query_data["std_recall_at_5"],
                    "Recall_10_평균": query_data["avg_recall_at_10"],
                    "Recall_10_표준편차": query_data["std_recall_at_10"],
                    "Brute_Force_시간_ms": query_data["avg_brute_force_time"] * 1000,
                    "HNSW_시간_ms": query_data["avg_hnsw_time"] * 1000
                })
        
        df = pd.DataFrame(csv_data)
        csv_path = output_path / "recall_summary.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"결과 저장: {csv_path}")


def main_create_collections():
    """컬렉션 생성 전용 메인 함수"""
    # 경로 설정
    base_path = Path(__file__).parent
    model_path = base_path.parent / "ko-sbert-sts"
    pdf_folder = base_path / "Data_pdf"
    
    # 실험 파라미터
    pages_per_collection = 1000  # 컬렉션당 페이지 수
    
    print("="*60)
    print("ChromaDB 컬렉션 생성")
    print("="*60)
    print(f"모델 경로: {model_path}")
    print(f"PDF 폴더: {pdf_folder}")
    print(f"컬렉션당 페이지 수: {pages_per_collection}페이지")
    print("="*60)
    
    # 실험 객체 생성
    experiment = RecallExperiment(
        model_path=str(model_path)
    )
    
    # 컬렉션 생성 (체크포인트에서 로드, 기존 컬렉션 이후부터 계속 생성)
    collections_info = experiment.process_pdfs_and_create_collections(
        pdf_folder=pdf_folder,
        pages_per_collection=pages_per_collection
    )
    
    if not collections_info:
        print("생성된 컬렉션이 없습니다.")
        return
    
    print("\n" + "="*60)
    print("컬렉션 생성 완료!")
    print("="*60)


def main():
    """Recall@K 평가 메인 함수"""
    # 경로 설정
    base_path = Path(__file__).parent
    model_path = base_path.parent / "ko-sbert-sts"
    
    # 실험 파라미터
    test_queries = [
        "딥러닝 모델의 성능 향상 방법",
        "자연어 처리 기술",
        "컴퓨터 비전 알고리즘",
        "강화학습 기법",
        "신경망 아키텍처"
    ]
    top_k = 10  # 검색 결과 개수
    num_runs = 1  # 각 쿼리당 반복 실행 횟수
    
    print("="*60)
    print("ChromaDB Recall@K 평가 실험 시작")
    print("="*60)
    print(f"모델 경로: {model_path}")
    print(f"테스트 쿼리 수: {len(test_queries)}")
    print(f"검색 결과 개수 (top_k): {top_k}")
    print(f"반복 실행 횟수: {num_runs}")
    print("="*60)
    
    # 실험 객체 생성
    experiment = RecallExperiment(
        model_path=str(model_path)
    )
    
    # 기존 컬렉션 가져오기
    print("\n기존 컬렉션 확인 중...")
    collections_info = experiment.get_existing_collections()
    
    if not collections_info:
        print("기존 컬렉션이 없습니다.")
        return
    
    print(f"기존 컬렉션 {len(collections_info)}개 발견!")
    print("\n컬렉션별 페이지 수:")
    for col_name, info in sorted(collections_info.items(), key=lambda x: x[1]['num_documents']):
        print(f"  {col_name}: {info['num_documents']}개 페이지")
    
    # Recall@K 평가
    results = experiment.evaluate_recall(
        collections_info=collections_info,
        test_queries=test_queries,
        top_k=top_k,
        num_runs=num_runs
    )
    
    # 결과 저장 및 시각화
    output_path = base_path / "experiment_results"
    experiment.save_results(results, output_path)
    experiment.visualize_results(results, output_path)
    
    print("\n" + "="*60)
    print("실험 완료!")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    # 명령줄 인자로 모드 선택
    if len(sys.argv) > 1 and sys.argv[1] == "create":
        # 컬렉션 생성 모드
        main_create_collections()
    else:
        # Recall 평가 모드 (기본)
        main()
