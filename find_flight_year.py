"""find_flight_year.py
항공편 지문(Flight Fingerprint)을 추출하여 데이터셋의 실제 연도(Year)를 역추적하는 스크립트
"""

from pathlib import Path
import urllib.parse
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train.csv"


def extract_flight_fingerprints():
    if not DATA_PATH.exists():
        print(f"[!] 데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
        return

    print(">> [1/2] 데이터 로드 중...")
    df = pd.read_csv(DATA_PATH)

    # 핵심 식별 컬럼 결측치 없는 순수 데이터 필터링
    clean_mask = (
        df["Tail_Number"].notna()
        & df["Airline"].notna()
        & df["Origin_Airport"].notna()
        & df["Destination_Airport"].notna()
        & df["Estimated_Departure_Time"].notna()
        & df["Distance"].notna()
    )
    clean_df = df[clean_mask].copy()

    # 중복 노선 제외하고 서로 다른 항공사/기체 5개 선별
    sample_df = clean_df.drop_duplicates(
        subset=["Airline", "Tail_Number"]
    ).head(5)

    print(">> [2/2] 연도 역추적용 항공편 5선 추출 완료!\n")
    print("=" * 75)
    print("★ 아래 5개 항공편 중 1~2개만 구글/항공 DB에 검색하면 연도가 바로 나옵니다.")
    print("=" * 75)

    for idx, (_, row) in enumerate(sample_df.iterrows(), 1):
        dep_time = int(row["Estimated_Departure_Time"])
        hh = dep_time // 100
        mm = dep_time % 100

        m = int(row["Month"])
        d = int(row["Day_of_Month"])
        tail = row["Tail_Number"]
        airline = row["Airline"]
        origin = row["Origin_Airport"]
        dest = row["Destination_Airport"]
        dist = int(row["Distance"])

        # 구글 검색용 쿼리 생성
        query = f'"{tail}" {origin} {dest} flight {m}월 {d}일'
        encoded_query = urllib.parse.quote(f"{tail} {origin} to {dest} flight")
        search_url = f"https://www.google.com/search?q={encoded_query}"

        print(f"[{idx}번 샘플]")
        print(
            f"  - 날짜/시각 : {m}월 {d}일 | 예정 출발: {hh:02d}:{mm:02d} (현지시각)"
        )
        print(f"  - 항공기/사 : 기체번호 [{tail}] | 항공사: {airline}")
        print(f"  - 운항 노선 : {origin} ➔ {dest} (거리: {dist:,} 마일)")
        print(f"  - 구글 검색 링크: {search_url}")
        print("-" * 75)

    print("\n💡 [확인 팁]")
    print(
        "1. 위의 '구글 검색 링크' 중 하나를 클릭해 보세요 (또는 FlightAware / BTS 검색)."
    )
    print("2. 해당 기체(Tail_Number)가 해당 노선을 운항한 기록의 '연도(예: 2021년, 2022년)'를 확인합니다.")
    print(
        "3. 연도가 확인되면 아래 2단계 날씨 수집 파이프라인의 TARGET_YEAR에 적어주시면 됩니다!\n"
    )


if __name__ == "__main__":
    extract_flight_fingerprints()