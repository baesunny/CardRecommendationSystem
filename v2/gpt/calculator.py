"""
절감액 계산 모듈 (Stage 2, API 호출 없음)

parser.py가 만든 구조화 카드 + SpendingProfile → 카테고리별 절감액, 순 혜택, 순위
모든 수치 계산을 결정론적으로 처리하여 LLM의 산수 오류를 원천 차단.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List

_RAG = Path(__file__).parent.parent / "rag"
if str(_RAG) not in sys.path:
    sys.path.insert(0, str(_RAG))

from spending_analyzer import SpendingProfile


def calculate_card(parsed_card: Dict, profile: SpendingProfile) -> Dict:
    """
    구조화된 카드 1장에 대해 절감액 계산.

    절차:
      1. 전월 실적 조건(R) vs 사용자 월 평균 지출(U) 비교
      2. 카드의 category_discounts 순회하며 카테고리별 절감액 계산
         - 사용자가 해당 카테고리에 지출이 있는 경우에만
         - 월 한도(category_caps)가 있으면 min(절감액, 한도) 적용
      3. 달성 불가면 모든 절감액 0으로 강제
      4. net_benefit = total_annual_savings - annual_fee

    Returns:
        {
          "card_name", "card_company", "annual_fee", "previous_month_requirement",
          "monthly_req_achievable": bool,
          "monthly_req_note": str,
          "savings_breakdown": [
            {"category", "monthly_avg_spend", "discount_rate",
             "monthly_saving", "annual_saving"}
          ],
          "total_annual_savings": int,
          "net_benefit": int,
          # LLM 단계에서 활용할 원본 데이터
          "raw_benefits", "main_categories", "similarity_score"
        }
    """
    U = int(profile.monthly_avg)
    R = int(parsed_card["previous_month_requirement"])
    annual_fee = int(parsed_card["annual_fee"])

    # ── 전월 실적 조건 검토 (결정론적) ───────────────────────────
    if U >= R:
        achievable = True
        diff = U - R
        note = (
            f"전월 실적 조건 {R:,}원 / 사용자 월 평균 지출 {U:,}원 "
            f"→ 달성 가능 (여유 {diff:,}원)"
        )
    else:
        achievable = False
        diff = R - U
        note = (
            f"전월 실적 조건 {R:,}원 / 사용자 월 평균 지출 {U:,}원 "
            f"→ 달성 불가 (미달 {diff:,}원)"
        )

    # ── 카테고리별 절감액 계산 ───────────────────────────────────
    breakdown: List[Dict] = []
    total_annual = 0

    for cat, rate in parsed_card["category_discounts"].items():
        # 사용자가 해당 카테고리에 지출 있는지 확인
        if cat not in profile.categories:
            continue

        stat = profile.categories[cat]
        monthly_spend = int(stat["monthly_avg"])

        # 절감액 계산
        if achievable:
            monthly_saving = int(monthly_spend * rate / 100)
            # 월 한도 적용
            cap = parsed_card.get("category_caps", {}).get(cat)
            if cap is not None:
                monthly_saving = min(monthly_saving, cap)
            annual_saving = monthly_saving * 12
        else:
            monthly_saving = 0
            annual_saving = 0

        breakdown.append({
            "category": cat,
            "monthly_avg_spend": monthly_spend,
            "discount_rate": f"{rate}%",
            "monthly_saving": monthly_saving,
            "annual_saving": annual_saving,
        })
        total_annual += annual_saving

    # 절감액 큰 순으로 정렬
    breakdown.sort(key=lambda x: x["annual_saving"], reverse=True)

    net_benefit = total_annual - annual_fee

    return {
        "card_name": parsed_card["card_name"],
        "card_company": parsed_card["card_company"],
        "annual_fee": annual_fee,
        "previous_month_requirement": R,
        "monthly_req_achievable": achievable,
        "monthly_req_note": note,
        "savings_breakdown": breakdown,
        "total_annual_savings": total_annual,
        "net_benefit": net_benefit,
        # LLM에 전달할 원본
        "raw_benefits": parsed_card.get("raw_benefits", ""),
        "main_categories": parsed_card.get("main_categories", ""),
        "similarity_score": parsed_card.get("similarity_score", 0),
    }


def rank_cards(
    parsed_cards: List[Dict],
    profile: SpendingProfile,
    top_k: int = 3,
) -> List[Dict]:
    """
    파싱된 카드 리스트 전체를 계산하고 net_benefit 내림차순 상위 top_k 반환.

    각 결과에 "rank" 필드도 부여.
    """
    calculated = [calculate_card(c, profile) for c in parsed_cards]
    calculated.sort(key=lambda x: x["net_benefit"], reverse=True)
    top = calculated[:top_k]
    for i, card in enumerate(top, 1):
        card["rank"] = i
    return top
