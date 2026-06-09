"""
소비 패턴 분석 모듈
card_history CSV → SpendingProfile
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Union
import pandas as pd


@dataclass
class SpendingProfile:
    total_annual: int
    monthly_avg: int
    # {"카테고리": {"total": int, "count": int, "ratio": float, "monthly_avg": int}}
    categories: Dict[str, dict]
    top_categories: List[str]   # total 내림차순 정렬

    def top_n(self, n: int = 5) -> List[str]:
        return self.top_categories[:n]

    def to_dict(self) -> dict:
        return {
            "total_annual": self.total_annual,
            "monthly_avg": self.monthly_avg,
            "categories": self.categories,
            "top_categories": self.top_categories,
        }

    def summary_text(self, top: int = 6) -> str:
        lines = [
            f"월평균 지출: {self.monthly_avg:,}원 (연 {self.total_annual:,}원)",
            "주요 소비 카테고리:",
        ]
        for cat in self.top_n(top):
            s = self.categories[cat]
            lines.append(f"  • {cat}: 월 {s['monthly_avg']:,}원 ({s['ratio']}%)")
        return "\n".join(lines)


# 사용자 카테고리 → 카드 main_categories 키워드 매핑
# 신버전 main_categories는 파이프(|) 구분, 슬래시(/) 복합 토큰 형식
# ex) "식당/카페", "마트/편의점", "의료/병원", "구독/스트리밍"
# 서브스트링 매칭이므로 부분 키워드만 있어도 대부분 매칭됨
CATEGORY_KEYWORD_MAP: Dict[str, List[str]] = {
    "온라인쇼핑":   ["온라인쇼핑", "쇼핑", "간편결제"],
    "음식점":       ["식당", "음식점", "패스트푸드", "일반음식점", "푸드"],
    "마트/슈퍼":    ["마트", "생활", "할인점"],
    "배달앱":       ["배달앱", "배달", "식당", "푸드"],
    "주유소":       ["주유"],
    "편의점":       ["편의점"],
    "운동/스포츠":  ["운동", "스포츠", "골프"],
    "의료/약국":    ["의료", "병원"],
    "카페":         ["카페", "커피", "디저트"],
    "대중교통":     ["대중교통", "교통", "자동차", "하이패스"],
    "통신":         ["통신", "skt", "kt", "lg"],
    "OTT/구독":     ["구독", "스트리밍", "ott", "영화", "디지털구독"],
}


def analyze(source: Union[str, pd.DataFrame]) -> SpendingProfile:
    """
    1년치 카드 거래 내역 CSV → SpendingProfile

    월 평균은 연간 합계 / 데이터 내 실제 월 수로 계산해
    전월 실적 조건 비교에 사용한다.
    """
    df = source if isinstance(source, pd.DataFrame) else pd.read_csv(source)

    # 데이터에 포함된 실제 월 수 (부분 연도 입력 대응)
    n_months = df["거래일시"].astype(str).str[:7].nunique()
    n_months = max(n_months, 1)

    total = int(df["결제금액"].sum())
    monthly_avg = total // n_months

    categories: Dict[str, dict] = {}
    for cat, group in df.groupby("카테고리"):
        cat_total = int(group["결제금액"].sum())
        # 해당 카테고리가 실제 사용된 월 수 기준 월 평균
        cat_months = group["거래일시"].astype(str).str[:7].nunique()
        categories[cat] = {
            "total": cat_total,
            "count": int(len(group)),
            "ratio": round(cat_total / total * 100, 1),
            "monthly_avg": cat_total // cat_months,
        }

    top_categories = sorted(
        categories.keys(),
        key=lambda x: categories[x]["total"],
        reverse=True,
    )

    return SpendingProfile(
        total_annual=total,
        monthly_avg=monthly_avg,
        categories=categories,
        top_categories=top_categories,
    )
