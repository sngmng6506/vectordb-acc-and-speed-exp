"""
PDF 페이지 수에 따른 ChromaDB 검색 속도 차이 실험

이 스크립트는:
1. PDF 파일들을 순회하면서 일정 페이지마다 컬렉션을 생성
2. 각 컬렉션의 크기(페이지 수)에 따라 검색 속도 측정
3. 페이지 수에 따른 검색 속도 차이 분석
"""

import os
import time
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import statistics

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


class ChromaDBExperiment:
    """ChromaDB 검색 속도 실험 클래스"""
    
    def __init__(self, model_path: str, pdf_folder: Path, chroma_db_path: Path = None):
        """
        Args:
            model_path: ko-sbert-sts 모델 경로
            pdf_folder: PDF 파일들이 있는 폴더
            chroma_db_path: ChromaDB 저장 경로
        """
        self.model_path = model_path
        self.pdf_folder = Path(pdf_folder)
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
        
    def process_pdfs_and_create_collections(self, max_files: int = None, 
                                           pages_per_collection: int = 50) -> Dict[str, Dict]:
        """
        PDF 파일들을 순회하면서 일정 페이지마다 누적 컬렉션을 생성하고 임베딩 저장
        (각 컬렉션은 이전 컬렉션의 모든 데이터를 포함하는 누적 구조)
        
        Args:
            max_files: 최대 처리할 PDF 파일 수 (None이면 전체)
            pages_per_collection: 컬렉션당 페이지 수 (이 페이지 수마다 새 컬렉션 생성)
        
        Returns:
            컬렉션 정보 딕셔너리 {collection_name: {num_documents, total_pages}}
        """
        print("\nPDF 파일 처리 및 누적 컬렉션 생성 중...")
        print(f"컬렉션당 페이지 수: {pages_per_collection}페이지 (누적 구조)")
        
        pdf_files = list(self.pdf_folder.glob("*.pdf"))
        
        if max_files:
            pdf_files = pdf_files[:max_files]
        
        print(f"총 {len(pdf_files)}개 PDF 파일 처리 예정\n")
        
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
        
        # 기존 컬렉션 확인
        existing_collections = set()
        try:
            collections = self.client.list_collections()
            for col in collections:
                existing_collections.add(col.name)
        except:
            pass
        
        # 누적 컬렉션 생성: pages_per_collection마다 새로운 컬렉션 생성 (이전까지의 모든 페이지 포함)
        collections_info = {}
        collection_num = 1
        
        for end_idx in range(pages_per_collection, len(all_texts) + 1, pages_per_collection):
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
    
    def measure_search_speed(self, collections_info: Dict, test_queries: List[str], 
                            top_k: int = 10, num_runs: int = 5) -> Dict[str, Dict]:
        """
        각 컬렉션별 검색 속도 측정
        
        Args:
            collections_info: 컬렉션 정보
            test_queries: 테스트 쿼리 리스트
            top_k: 검색 결과 개수
            num_runs: 각 쿼리당 반복 실행 횟수
        
        Returns:
            검색 속도 결과 딕셔너리
        """
        print("\n검색 속도 측정 중...")
        
        results = {}
        
        # 페이지 수 순으로 정렬
        sorted_collections = sorted(collections_info.items(), 
                                   key=lambda x: x[1]['num_documents'])
        
        for collection_name, info in sorted_collections:
            collection = self.client.get_collection(collection_name)
            
            print(f"\n[{collection_name}] 검색 속도 측정...")
            print(f"  컬렉션 내 페이지 수: {info['num_documents']}개")
            
            collection_results = {
                "num_documents": info["num_documents"],
                "collection_name": collection_name,
                "query_times": []
            }
            
            for query in test_queries:
                query_times = []
                
                # Warm-up: 첫 번째 검색은 캐시/인덱스 로딩을 위해 실행하되 결과에 포함하지 않음
                query_embedding_warmup = self.model.encode([query])[0]
                _ = collection.query(
                    query_embeddings=[query_embedding_warmup.tolist()],
                    n_results=top_k
                )
                
                # 각 실행 사이에 짧은 대기 (캐시 효과 최소화)
                time.sleep(0.01)  # 10ms 대기
                
                for run in range(num_runs):
                    # 쿼리 임베딩 (매번 새로 생성하여 캐싱 효과 최소화)
                    start_time = time.perf_counter()  # 더 정밀한 시간 측정
                    query_embedding = self.model.encode([query])[0]
                    
                    # 검색 실행
                    search_start = time.perf_counter()
                    results_data = collection.query(
                        query_embeddings=[query_embedding.tolist()],
                        n_results=top_k
                    )
                    search_end = time.perf_counter()
                    
                    total_time = time.perf_counter() - start_time
                    search_time = search_end - search_start
                    
                    query_times.append({
                        "total_time": total_time,
                        "embedding_time": search_start - start_time,
                        "search_time": search_time,
                        "num_results": len(results_data['ids'][0])
                    })
                    
                    # 각 실행 사이에 짧은 대기
                    if run < num_runs - 1:  # 마지막 실행이 아니면
                        time.sleep(0.01)  # 10ms 대기
                
                # 통계 계산 (평균과 중앙값 모두 계산)
                avg_total = statistics.mean([t["total_time"] for t in query_times])
                avg_search = statistics.mean([t["search_time"] for t in query_times])
                avg_embedding = statistics.mean([t["embedding_time"] for t in query_times])
                median_search = statistics.median([t["search_time"] for t in query_times])
                std_search = statistics.stdev([t["search_time"] for t in query_times]) if len(query_times) > 1 else 0.0
                
                collection_results["query_times"].append({
                    "query": query,
                    "avg_total_time": avg_total,
                    "avg_search_time": avg_search,
                    "avg_embedding_time": avg_embedding,
                    "median_search_time": median_search,
                    "std_search_time": std_search,
                    "runs": query_times
                })
                
                print(f"    쿼리: '{query[:50]}...'")
                print(f"      평균 총 시간: {avg_total*1000:.2f}ms")
                print(f"      평균 검색 시간: {avg_search*1000:.2f}ms (±{std_search*1000:.2f}ms)")
                print(f"      중앙값 검색 시간: {median_search*1000:.2f}ms")
            
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
        avg_search_times = []
        avg_total_times = []
        
        # 페이지 수 순으로 정렬
        sorted_results = sorted(results.items(), 
                               key=lambda x: x[1]['num_documents'])
        
        for collection_name, data in sorted_results:
            collection_names.append(f"{data['num_documents']}페이지")
            num_docs.append(data["num_documents"])
            
            # 모든 쿼리의 평균 검색 시간 계산
            all_search_times = []
            all_total_times = []
            for query_data in data["query_times"]:
                all_search_times.append(query_data["avg_search_time"])
                all_total_times.append(query_data["avg_total_time"])
            
            avg_search_times.append(statistics.mean(all_search_times) * 1000)  # ms로 변환
            avg_total_times.append(statistics.mean(all_total_times) * 1000)  # ms로 변환
        
        # 그래프 1: 페이지 수 vs 검색 시간
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 검색 시간 그래프
        axes[0].plot(num_docs, avg_search_times, marker='o', linewidth=2, markersize=8)
        axes[0].set_xlabel('페이지 수', fontsize=12)
        axes[0].set_ylabel('평균 검색 시간 (ms)', fontsize=12)
        axes[0].set_title('페이지 수에 따른 검색 속도', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # 총 시간 그래프
        axes[1].plot(num_docs, avg_total_times, marker='s', linewidth=2, 
                    markersize=8, color='orange')
        axes[1].set_xlabel('페이지 수', fontsize=12)
        axes[1].set_ylabel('평균 총 시간 (ms)', fontsize=12)
        axes[1].set_title('페이지 수에 따른 총 처리 시간 (임베딩+검색)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "search_speed_by_pages.png", dpi=300, bbox_inches='tight')
        print(f"\n그래프 저장: {output_path / 'search_speed_by_pages.png'}")
        
        # 바 차트
        fig, ax = plt.subplots(figsize=(14, 6))
        x_pos = range(len(collection_names))
        ax.bar(x_pos, avg_search_times, alpha=0.7, color='steelblue')
        ax.set_xlabel('컬렉션 (페이지 수)', fontsize=12)
        ax.set_ylabel('평균 검색 시간 (ms)', fontsize=12)
        ax.set_title('컬렉션별 검색 속도 비교', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(collection_names, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 값 표시
        for i, v in enumerate(avg_search_times):
            ax.text(i, v, f'{v:.1f}ms', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path / "search_speed_by_collection.png", dpi=300, bbox_inches='tight')
        print(f"그래프 저장: {output_path / 'search_speed_by_collection.png'}")
        
        plt.close('all')
    
    def save_results(self, results: Dict[str, Dict], output_path: Path = None):
        """결과를 JSON과 CSV로 저장"""
        if output_path is None:
            output_path = Path(__file__).parent / "experiment_results"
        output_path.mkdir(exist_ok=True)
        
        # JSON 저장
        json_path = output_path / "results.json"
        
        # JSON 직렬화 가능한 형태로 변환
        json_results = {}
        for collection_name, data in results.items():
            json_results[collection_name] = {
                "num_documents": data["num_documents"],
                "queries": []
            }
            for query_data in data["query_times"]:
                json_results[collection_name]["queries"].append({
                    "query": query_data["query"],
                    "avg_total_time_ms": query_data["avg_total_time"] * 1000,
                    "avg_search_time_ms": query_data["avg_search_time"] * 1000,
                    "avg_embedding_time_ms": query_data["avg_embedding_time"] * 1000
                })
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n결과 저장: {json_path}")
        
        # CSV 저장 (요약)
        csv_data = []
        for collection_name, data in results.items():
            for query_data in data["query_times"]:
                csv_data.append({
                    "컬렉션명": collection_name,
                    "페이지_수": data["num_documents"],
                    "쿼리": query_data["query"],
                    "평균_총_시간_ms": query_data["avg_total_time"] * 1000,
                    "평균_검색_시간_ms": query_data["avg_search_time"] * 1000,
                    "평균_임베딩_시간_ms": query_data["avg_embedding_time"] * 1000
                })
        
        df = pd.DataFrame(csv_data)
        csv_path = output_path / "results_summary.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"결과 저장: {csv_path}")


def main():
    """메인 실행 함수"""
    # 경로 설정
    base_path = Path(__file__).parent
    model_path = base_path.parent / "ko-sbert-sts"
    pdf_folder = base_path / "Data_pdf"
    
    # 실험 파라미터
    max_files = None  # None이면 전체 파일 처리
    pages_per_collection = 1000 # 컬렉션당 페이지 수 (이 페이지 수마다 새 컬렉션 생성)
    test_queries = [
        "딥러닝 모델의 성능 향상 방법",
        "자연어 처리 기술",
        "컴퓨터 비전 알고리즘",
        "강화학습 기법",
        "신경망 아키텍처"
    ]
    
    print("="*60)
    print("ChromaDB 검색 속도 실험 시작")
    print("="*60)
    print(f"모델 경로: {model_path}")
    print(f"PDF 폴더: {pdf_folder}")
    print(f"최대 처리 파일 수: {max_files if max_files else '전체'}")
    print(f"컬렉션당 페이지 수: {pages_per_collection}페이지")
    print(f"테스트 쿼리 수: {len(test_queries)}")
    print("="*60)
    
    # 실험 객체 생성
    experiment = ChromaDBExperiment(
        model_path=str(model_path),
        pdf_folder=pdf_folder
    )
    
    # 기존 컬렉션 가져오기
    print("\n기존 컬렉션 확인 중...")
    collections_info = experiment.get_existing_collections()
    
    if not collections_info:
        print("기존 컬렉션이 없습니다. 새로 생성합니다...")
        # 1. PDF 처리 및 컬렉션 생성 (일정 페이지마다)
        collections_info = experiment.process_pdfs_and_create_collections(
            max_files=max_files,
            pages_per_collection=pages_per_collection
        )
        
        if not collections_info:
            print("생성된 컬렉션이 없습니다.")
            return
    else:
        print(f"기존 컬렉션 {len(collections_info)}개 발견!")
        print("\n컬렉션별 페이지 수:")
        for col_name, info in sorted(collections_info.items(), key=lambda x: x[1]['num_documents']):
            print(f"  {col_name}: {info['num_documents']}개 페이지")
    
    # 2. 검색 속도 측정
    results = experiment.measure_search_speed(
        collections_info=collections_info,
        test_queries=test_queries,
        top_k=5,
        num_runs=5
    )
    
    # 3. 결과 저장 및 시각화
    output_path = base_path / "experiment_results"
    experiment.save_results(results, output_path)
    experiment.visualize_results(results, output_path)
    
    print("\n" + "="*60)
    print("실험 완료!")
    print("="*60)


if __name__ == "__main__":
    main()
