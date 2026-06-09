import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# 재현성을 위한 시드 고정
random.seed(42)
np.random.seed(42)

# ──────────────────────────────────────────
# 카테고리별 가맹점 목록 및 금액 범위 정의
# ──────────────────────────────────────────
CATEGORY_CONFIG = {
    "편의점": {
        "stores": ["CU 홍대점", "GS25 신촌점", "세븐일레븐 마포점", "이마트24 합정점", "CU 상수점"],
        "amount_range": (2000, 15000),
        "monthly_freq": (8, 20),
    },
    "카페": {
        "stores": ["스타벅스 홍대입구점", "이디야 신촌점", "메가커피 마포점", "투썸플레이스 합정점", "빽다방 상수점"],
        "amount_range": (3000, 12000),
        "monthly_freq": (6, 16),
    },
    "음식점": {
        "stores": ["맥도날드 홍대점", "롯데리아 신촌점", "버거킹 마포점", "교촌치킨 합정점", "BBQ 상수점",
                   "한솥도시락 홍대점", "본죽 신촌점", "김밥천국 마포점"],
        "amount_range": (7000, 45000),
        "monthly_freq": (10, 25),
    },
    "대중교통": {
        "stores": ["서울교통공사", "카카오T", "티머니"],
        "amount_range": (1250, 3000),
        "monthly_freq": (20, 50),
    },
    "주유소": {
        "stores": ["SK주유소 마포점", "GS칼텍스 홍대점", "현대오일뱅크 신촌점"],
        "amount_range": (30000, 80000),
        "monthly_freq": (2, 5),
    },
    "마트/슈퍼": {
        "stores": ["이마트 마포점", "홈플러스 합정점", "롯데마트 상암점", "GS더프레시 홍대점"],
        "amount_range": (15000, 120000),
        "monthly_freq": (3, 8),
    },
    "온라인쇼핑": {
        "stores": ["쿠팡", "네이버쇼핑", "마켓컬리", "11번가", "G마켓"],
        "amount_range": (10000, 150000),
        "monthly_freq": (4, 12),
    },
    "배달앱": {
        "stores": ["배달의민족", "쿠팡이츠", "요기요"],
        "amount_range": (12000, 35000),
        "monthly_freq": (5, 15),
    },
    "의료/약국": {
        "stores": ["올리브영 홍대점", "CJ올리브영 신촌점", "세브란스약국", "온누리약국"],
        "amount_range": (5000, 80000),
        "monthly_freq": (1, 4),
    },
    "통신": {
        "stores": ["SKT", "KT", "LG U+"],
        "amount_range": (35000, 75000),
        "monthly_freq": (1, 1),
    },
    "OTT/구독": {
        "stores": ["넷플릭스", "유튜브프리미엄", "멜론", "왓챠", "스포티파이"],
        "amount_range": (4000, 17000),
        "monthly_freq": (1, 3),
    },
    "운동/스포츠": {
        "stores": ["헬스장 월정액", "스크린골프 마포점", "볼링장 홍대"],
        "amount_range": (5000, 60000),
        "monthly_freq": (1, 5),
    },
}

# ──────────────────────────────────────────
# 데이터 생성 함수
# ──────────────────────────────────────────
def generate_transactions(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    card_number: str = "5234-****-****-1234",
) -> pd.DataFrame:

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    records = []

    current = start
    while current <= end:
        year, month = current.year, current.month
        # 해당 월의 마지막 날 계산
        if month == 12:
            month_end = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            month_end = datetime(year, month + 1, 1) - timedelta(days=1)
        month_end = min(month_end, end)

        for category, config in CATEGORY_CONFIG.items():
            freq = random.randint(*config["monthly_freq"])
            for _ in range(freq):
                # 해당 월 내 랜덤 날짜
                day_offset = random.randint(0, (month_end - datetime(year, month, 1)).days)
                tx_date = datetime(year, month, 1) + timedelta(days=day_offset)

                # 시간 (편의점/카페는 오전, 음식점은 점심/저녁 집중)
                if category in ["편의점", "카페"]:
                    # 7~21시 (15개), 오전/오후 이용 집중
                    hour = random.choices(range(7, 22), weights=[3,5,5,4,4,4,4,5,5,4,4,3,3,2,1], k=1)[0]
                elif category == "음식점":
                    hour = random.choices([11,12,13,18,19,20,21], weights=[2,5,3,3,5,4,2], k=1)[0]
                else:
                    hour = random.randint(9, 21)
                minute = random.randint(0, 59)
                tx_datetime = tx_date.replace(hour=hour, minute=minute)

                amount = random.randint(*config["amount_range"])
                # 100원 단위로 반올림
                amount = round(amount / 100) * 100

                store = random.choice(config["stores"])
                records.append({
                    "거래일시": tx_datetime.strftime("%Y-%m-%d %H:%M"),
                    "카드번호": card_number,
                    "가맹점명": store,
                    "카테고리": category,
                    "결제금액": amount,
                    "승인번호": f"{random.randint(10000000, 99999999)}",
                })

        # 다음 달로 이동
        if month == 12:
            current = datetime(year + 1, 1, 1)
        else:
            current = datetime(year, month + 1, 1)

    df = pd.DataFrame(records)
    df = df.sort_values("거래일시").reset_index(drop=True)
    return df


def print_summary(df: pd.DataFrame):
    print("=" * 50)
    print("▶ 생성된 카드 사용 내역 요약")
    print("=" * 50)
    print(f"총 거래 건수  : {len(df):,}건")
    print(f"총 결제 금액  : {df['결제금액'].sum():,}원")
    print(f"월평균 결제   : {df['결제금액'].sum() // 12:,}원")
    print(f"기간          : {df['거래일시'].min()} ~ {df['거래일시'].max()}")
    print()
    print("── 카테고리별 지출 ──")
    summary = (
        df.groupby("카테고리")["결제금액"]
        .agg(건수="count", 합계="sum")
        .sort_values("합계", ascending=False)
    )
    summary["비율(%)"] = (summary["합계"] / summary["합계"].sum() * 100).round(1)
    print(summary.to_string())
    print("=" * 50)


if __name__ == "__main__":
    df = generate_transactions(
        start_date="2024-01-01",
        end_date="2024-12-31",
        card_number="5234-****-****-1234",
    )

    # CSV 저장
    output_path = "card_history_2024.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"파일 저장 완료: {output_path}")
    print()

    print_summary(df)
    print()
    print("── 샘플 데이터 (상위 10건) ──")
    print(df.head(10).to_string(index=False))
