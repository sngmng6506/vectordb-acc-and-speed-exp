import pandas as pd
import requests
import os
from pathlib import Path
from tqdm import tqdm
import time

def download_arxiv_papers(csv_path, download_folder=None):
    """
    CSV 파일에서 arxiv 논문 목록을 읽어서 PDF를 다운로드합니다.
    
    Args:
        csv_path: CSV 파일 경로
        download_folder: 다운로드할 폴더 경로 (None이면 CSV 파일과 같은 폴더)
    """
    # CSV 파일 읽기
    print(f"CSV 파일 읽는 중: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 다운로드 폴더 설정
    if download_folder is None:
        download_folder = os.path.dirname(csv_path)
    
    download_folder = Path(download_folder)
    download_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"다운로드 폴더: {download_folder}")
    print(f"총 {len(df)}개의 논문을 다운로드합니다.\n")
    
    # 통계 변수
    success_count = 0
    skip_count = 0
    error_count = 0
    errors = []
    
    # 각 논문 다운로드
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="다운로드 진행"):
        arxiv_id = str(row['arxiv_id']).strip()
        
        # arxiv_id가 비어있거나 유효하지 않은 경우 스킵
        if pd.isna(arxiv_id) or not arxiv_id:
            skip_count += 1
            continue
        
        # 파일명 생성 (arxiv_id를 사용, 특수문자 제거)
        safe_filename = arxiv_id.replace('/', '_').replace('\\', '_')
        pdf_path = download_folder / f"{safe_filename}.pdf"
        
        # 이미 다운로드된 파일이 있으면 스킵
        if pdf_path.exists():
            skip_count += 1
            continue
        
        # arxiv PDF URL 생성
        # arxiv_id 형식이 "2511.11571v1"인 경우 "2511.11571"로 변환
        arxiv_id_clean = arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf"
        
        try:
            # PDF 다운로드
            response = requests.get(pdf_url, stream=True, timeout=30)
            response.raise_for_status()
            
            # PDF 파일인지 확인
            if response.headers.get('content-type', '').startswith('application/pdf'):
                # 파일 저장
                with open(pdf_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                success_count += 1
            else:
                # PDF가 아닌 경우 (예: HTML 에러 페이지)
                error_count += 1
                errors.append(f"{arxiv_id}: PDF가 아닌 응답 (Content-Type: {response.headers.get('content-type')})")
                if pdf_path.exists():
                    pdf_path.unlink()  # 부분적으로 다운로드된 파일 삭제
        
        except requests.exceptions.RequestException as e:
            error_count += 1
            errors.append(f"{arxiv_id}: {str(e)}")
            if pdf_path.exists():
                pdf_path.unlink()  # 부분적으로 다운로드된 파일 삭제
        
        except Exception as e:
            error_count += 1
            errors.append(f"{arxiv_id}: 예상치 못한 오류 - {str(e)}")
            if pdf_path.exists():
                pdf_path.unlink()
        
        # 서버 부하를 줄이기 위해 짧은 딜레이 추가
        time.sleep(0.5)
    
    # 결과 출력
    print("\n" + "="*50)
    print("다운로드 완료!")
    print(f"성공: {success_count}개")
    print(f"스킵 (이미 존재): {skip_count}개")
    print(f"실패: {error_count}개")
    
    if errors:
        print("\n오류 목록:")
        for error in errors[:10]:  # 처음 10개만 표시
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... 외 {len(errors) - 10}개 오류")
    
    return success_count, skip_count, error_count


if __name__ == "__main__":
    # CSV 파일 경로
    csv_path = Path(__file__).parent / "Data_pdf" / "arxiv_papers_5k.csv"
    
    # 다운로드 실행
    download_arxiv_papers(csv_path)
