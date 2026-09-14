"""
merge_weather_pipeline.py
2022년 미국 30대 허브 공항의 실시간 기상 관측 데이터를 직통 수집하여 항공 데이터에 병합하는 파이프라인
"""
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train.csv"
OUTPUT_PATH = ROOT / "data" / "train_with_weather.csv"

# -----------------------------------------------------------------------------
# ★ 항공 데이터의 실측 연도: 2022년 확정!
# -----------------------------------------------------------------------------
TARGET_YEAR = 2022

# 미국 30대 핵심 허브 공항 <-> NOAA/Meteostat 고유 관측소 ID(WMO) 직통 매핑
# 검색 과정을 생략하고 다이렉트로 초고속 다운로드합니다.
AIRPORT_STATION_MAP = {
    "ATL": "72219",  # 애틀랜타 하츠필드-잭슨
    "DFW": "72259",  # 댈러스/포트워스
    "DEN": "72469",  # 덴버
    "ORD": "72530",  # 시카고 오헤어
    "LAX": "72295",  # 로스앤젤레스
    "CLT": "72314",  # 샬럿 더글러스
    "MCO": "72205",  # 올랜도
    "LAS": "72386",  # 라스베이거스
    "PHX": "72278",  # 피닉스
    "MIA": "72202",  # 마이애미
    "SEA": "72793",  # 시애틀-타코마
    "IAH": "72243",  # 휴스턴 조지 부시
    "JFK": "74486",  # 뉴욕 JFK
    "EWR": "72502",  # 뉴어크 리버티
    "FLL": "72203",  # 포트로더데일
    "MSP": "72658",  # 미니애폴리스-세인트폴
    "SFO": "72494",  # 샌프란시스코
    "DTW": "72537",  # 디트로이트
    "BOS": "72509",  # 보스턴 로건
    "SLC": "72572",  # 솔트레이크시티
    "PHL": "72408",  # 필라델피아
    "BWI": "72406",  # 볼티모어
    "TPA": "72211",  # 탬파
    "SAN": "72290",  # 샌디에이고
    "LGA": "72503",  # 뉴욕 라과디아
    "MDW": "72534",  # 시카고 미드웨이
    "BNA": "72327",  # 내슈빌
    "IAD": "72403",  # 워싱턴 덜레스
    "DAL": "72258",  # 댈러스 러브필드
    "AUS": "72254",  # 오스틴 버그스트롬
}


def fetch_airport_weather(year: int, station_map: dict) -> pd.DataFrame:
    """Meteostat v1/v2 버전에 상관없이 다이렉트 Station ID로 1년치 시간별 날씨 수집"""
    try:
        from meteostat import Hourly
    except ImportError:
        try:
            from meteostat import hourly as Hourly
        except ImportError:
            raise ImportError("meteostat이 설치되지 않았습니다. 터미널에 'uv add meteostat'을 실행해 주세요.")

    print(f">> [1/3] {year}년 미국 30대 핵심 허브 공항 기상 데이터 다운로드 시작...")
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31, 23, 59)

    weather_frames = []

    for iata, station_id in station_map.items():
        try:
            # 고유 관측소 번호로 다이렉트 1년치 시간대별 데이터 다운로드
            data = Hourly(station_id, start_date, end_date).fetch()
            
            if data is None or data.empty:
                print(f"   - [{iata}] 관측 데이터 없음 (건너뜁니다)")
                continue

            data = data.reset_index()
            data["Airport_Code"] = iata
            data["Month"] = data["time"].dt.month
            data["Day_of_Month"] = data["time"].dt.day
            data["Hour"] = data["time"].dt.hour

            # 핵심 날씨 피처:
            # temp(기온), dwpt(이슬점), rhum(습도), prcp(강수량), snow(적설량)
            # wspd(풍속), wpgt(돌풍 속도), pres(기압), coco(기상 코드)
            cols = [
                "Airport_Code", "Month", "Day_of_Month", "Hour",
                "temp", "dwpt", "rhum", "prcp", "snow", "wspd", "wpgt", "pres", "coco"
            ]
            available = [c for c in cols if c in data.columns]
            weather_frames.append(data[available])
            print(f"   ★ [{iata}] 2022년 시간별 기상 관측치 수집 완료 ({len(data):,}행)")
        except Exception as e:
            print(f"   ! [{iata}] 수집 실패: {e}")

    if not weather_frames:
        raise RuntimeError("날씨 데이터를 1건도 수집하지 못했습니다. 인터넷 연결 상태를 확인해 주세요.")

    all_weather = pd.concat(weather_frames, ignore_index=True)
    print(f">> 총 {len(all_weather):,}건의 시간별 공항 기상 관측치 확보 완료!")
    return all_weather


def build_weather_merged_dataset():
    if not DATA_PATH.exists():
        print(f"[!] 원본 데이터가 없습니다: {DATA_PATH}")
        return

    # 1. 항공 데이터 로드
    print(f"\n>> [2/3] 항공 데이터 로드 중: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # 출발 시간(Hour) 파싱
    dep_val = pd.to_numeric(df["Estimated_Departure_Time"], errors="coerce").fillna(-1).astype(int)
    df["Dep_Hour"] = np.where(dep_val >= 0, dep_val // 100, -1)

    # 2. 날씨 데이터 수집
    weather_df = fetch_airport_weather(TARGET_YEAR, AIRPORT_STATION_MAP)

    # 3. 컬럼명 접두어(Weather_Origin_) 부여
    weather_rename = {
        col: f"Weather_Origin_{col}"
        for col in weather_df.columns
        if col not in ["Airport_Code", "Month", "Day_of_Month", "Hour"]
    }
    weather_df = weather_df.rename(columns=weather_rename)

    # 4. 결합 (Left Join)
    print(">> [3/3] 항공 데이터에 출발지 실시간 기상 관측 데이터 병합(Merge) 중...")
    merged_df = pd.merge(
        df,
        weather_df,
        left_on=["Origin_Airport", "Month", "Day_of_Month", "Dep_Hour"],
        right_on=["Airport_Code", "Month", "Day_of_Month", "Hour"],
        how="left",
    )

    # 조인용 중복 컬럼 삭제
    merged_df = merged_df.drop(columns=["Airport_Code", "Hour"], errors="ignore")

    # CSV 저장 (utf-8-sig)
    merged_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 65)
    print(f"★ 축하합니다! 날씨 결합 데이터셋 생성 완료: {OUTPUT_PATH}")
    print(f"★ 새 데이터셋 크기 : {merged_df.shape}")
    print(
        f"★ 날씨 피처 결합 성공률(상위 30대 허브 공항 기준) : {merged_df['Weather_Origin_wspd'].notna().mean()*100:.2f}%"
    )
    print("=" * 65)


if __name__ == "__main__":
    build_weather_merged_dataset()