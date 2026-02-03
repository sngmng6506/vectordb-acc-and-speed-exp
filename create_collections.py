"""
ChromaDB 컬렉션 생성 스크립트

이 스크립트는:
1. PDF 파일들을 처리하여 텍스트 추출 (또는 체크포인트에서 로드)
2. 일정 페이지마다 누적 컬렉션을 생성하고 임베딩 저장
3. 기존 컬렉션이 있으면 그 이후부터 계속 생성
"""

import pickle
from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import PyPDF2
from tqdm import tqdm


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


class CollectionCreator:
    """ChromaDB 컬렉션 생성 클래스"""
    
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
                                           pages_per_collection: int = 1000) -> Dict[str, Dict]:
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


def main():
    """메인 실행 함수"""
    # 경로 설정
    base_path = Path(__file__).parent
    model_path = base_path.parent / "ko-sbert-sts"
    pdf_folder = base_path / "Data_pdf"
    
    # 실험 파라미터
    max_files = None  # None이면 전체 파일 처리
    pages_per_collection = 1000  # 컬렉션당 페이지 수
    
    print("="*60)
    print("ChromaDB 컬렉션 생성")
    print("="*60)
    print(f"모델 경로: {model_path}")
    print(f"PDF 폴더: {pdf_folder}")
    print(f"최대 처리 파일 수: {max_files if max_files else '전체'}")
    print(f"컬렉션당 페이지 수: {pages_per_collection}페이지")
    print("="*60)
    
    # 컬렉션 생성 객체 생성
    creator = CollectionCreator(
        model_path=str(model_path)
    )
    
    # 컬렉션 생성 (체크포인트에서 로드, 기존 컬렉션 이후부터 계속 생성)
    collections_info = creator.process_pdfs_and_create_collections(
        pdf_folder=pdf_folder,
        max_files=max_files,
        pages_per_collection=pages_per_collection
    )
    
    if not collections_info:
        print("생성된 컬렉션이 없습니다.")
        return
    
    print("\n" + "="*60)
    print("컬렉션 생성 완료!")
    print("="*60)


if __name__ == "__main__":
    main()
