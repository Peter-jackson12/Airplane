"""merge_weather_pipeline.py
확인된 연도를 기반으로 미국 주요 공항의 기상 데이터를 자동 수집하여 항공 데이터에 병합(Join)하는 파이프라인
"""

from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train.csv"
OUTPUT_PATH = ROOT / "data" / "train_with_weather.csv"

# -----------------------------------------------------------------------------
# ★ 1단계에서 확인한 실제 연도를 여기에 입력해 주세요! (기본값 2022 세팅)
# -----------------------------------------------------------------------------
TARGET_YEAR = 2026

# 트래픽 상위 핵심 허브 공항 30개 IATA 코드 (필요시 확장 가능)
TOP_AIRPORTS = [
    "ATL",
    "DFW",
    "DEN",
    "ORD",
    "LAX",
    "CLT",
    "MCO",
    "LAS",
    "PHX",
    "MIA",
    "SEA",
    "IAH",
    "JFK",
    "EWR",
    "FLL",
    "MSP",
    "SFO",
    "DTW",
    "BOS",
    "SLC",
    "PHL",
    "BWI",
    "TPA",
    "SAN",
    "LGA",
    "MDW",
    "BNA",
    "IAD",
    "DAL",
    "AUS",
]


def fetch_airport_weather(year: int, airports: list) -> pd.DataFrame:
    """Meteostat을 통해 지정된 공항들의 1년치 시간별 기상 데이터를 수집"""
    try:
        from meteostat import Hourly, Stations
    except ImportError:
        raise ImportError(
            "meteostat 라이브러리가 필요합니다. 터미널에 'uv add meteostat'을 실행해 주세요."
        )

    print(f">> [1/3] {year}년 미국 주요 공항 기상 관측 데이터 다운로드 시작...")
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31, 23, 59)

    weather_frames = []

    for iata in airports:
        try:
            stations = Stations().region("US")
            station = stations.get(iata)

            if station is None:
                continue

            # 시간대별 날씨 가져오기
            data = Hourly(station, start_date, end_date).fetch()
            if data.empty:
                continue

            data = data.reset_index()
            data["Airport_Code"] = iata
            data["Month"] = data["time"].dt.month
            data["Day_of_Month"] = data["time"].dt.day
            data["Hour"] = data["time"].dt.hour

            # 핵심 날씨 피처만 선택
            # temp: 기온, dwpt: 이슬점, rhum: 습도, prcp: 강수량, snow: 적설량
            # wspd: 풍속, wpgt: 돌풍 속도, pres: 기압, coco: 기상 상태 코드
            cols = [
                "Airport_Code",
                "Month",
                "Day_of_Month",
                "Hour",
                "temp",
                "dwpt",
                "rhum",
                "prcp",
                "snow",
                "wspd",
                "wpgt",
                "pres",
                "coco",
            ]
            available = [c for c in cols if c in data.columns]
            weather_frames.append(data[available])
            print(f"   - [{iata}] 관측소 데이터 수집 완료 ({len(data):,}행)")
        except Exception as e:
            print(f"   ! [{iata}] 수집 실패: {e}")

    if not weather_frames:
        raise RuntimeError("날씨 데이터를 1건도 수집하지 못했습니다.")

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
    dep_val = (
        pd.to_numeric(df["Estimated_Departure_Time"], errors="coerce")
        .fillna(-1)
        .astype(int)
    )
    df["Dep_Hour"] = np.where(dep_val >= 0, dep_val // 100, -1)

    # 2. 날씨 데이터 수집
    weather_df = fetch_airport_weather(TARGET_YEAR, TOP_AIRPORTS)

    # 3. 날씨 피처명 앞에 Weather_ 접두어 부여
    weather_rename = {
        col: f"Weather_Origin_{col}"
        for col in weather_df.columns
        if col not in ["Airport_Code", "Month", "Day_of_Month", "Hour"]
    }
    weather_df = weather_df.rename(columns=weather_rename)

    # 4. 결합 (Merge)
    print(">> [3/3] 항공 운항 데이터와 출발지 실시간 날씨 데이터 병합(Merge) 중...")
    merged_df = pd.merge(
        df,
        weather_df,
        left_on=["Origin_Airport", "Month", "Day_of_Month", "Dep_Hour"],
        right_on=["Airport_Code", "Month", "Day_of_Month", "Hour"],
        how="left",
    )

    # 조인용 중복 컬럼 정리
    merged_df = merged_df.drop(
        columns=["Airport_Code", "Hour"], errors="ignore"
    )

    # 저장 (한글 인코딩 방지 utf-8-sig)
    merged_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 65)
    print(f"★ 날씨 데이터 병합 완료! 저장 경로: {OUTPUT_PATH}")
    print(f"★ 새 데이터셋 크기 : {merged_df.shape}")
    print(
        f"★ 날씨 피처 결측 비율(출발공항 상위 30개 기준) : {merged_df['Weather_Origin_wspd'].notna().mean()*100:.2f}% 결합 성공"
    )
    print("=" * 65)


if __name__ == "__main__":
    build_weather_merged_dataset()