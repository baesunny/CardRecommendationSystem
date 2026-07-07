"""
카드별 거래 데이터 + card_benefits_structured.json
→ OpenAI API 시뮬레이션 결과 생성

카드 100개 × 각 1회 호출로 연간 최대 순절감액 산출
"""

import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

from openai import OpenAI

# ── 경로 설정 ───────────────────────────────────────────────
BASE_DIR         = Path(__file__).resolve().parent
OUTPUT_DIR       = BASE_DIR / "output"        # 고정 데이터 (카드 DB)
DEBUG_OUTPUT_DIR = BASE_DIR / "debug_output"  # 디버그 모드 산출물
STRUCTURED_JSON  = OUTPUT_DIR / "card_benefits_structured.json"
CTM_JSON         = DEBUG_OUTPUT_DIR / "card_transactions_map.json"

# ── 디버그 모드 ─────────────────────────────────────────────
# True : 독립 실행 가능. card_transactions_map.json 읽고 simulation_results.json 저장
# False: app.py 메모리 파이프라인용 (파일 입출력 없음)
DEBUG_MODE = True

MODEL       = "gpt-4o-mini"
MAX_RETRIES = 2
SLEEP_SEC   = 1.0
MAX_WORKERS = int(os.environ.get("SIM_MAX_WORKERS", "4"))

ONLINE_MERCHANTS = {
    "쿠팡", "배달의민족", "쿠팡이츠", "요기요", "네이버쇼핑",
    "G마켓", "옥션", "11번가", "SSG.COM", "컬리",
}
OVERSEAS_KEYWORDS = {"해외", "해외이용", "해외가맹점", "해외이용금액"}
UNKNOWN_CHANNEL_KEYWORDS = {"간편결제", "앱카드", "FAN페이", "페이(앱카드)"}
AREA_RANK_KEYWORDS = {
    "영역 중",
    "이용금액 1위",
    "이용금액 2위",
    "1위 영역",
    "2위 영역",
    "상위 영역",
    "가장 많이 쓴 영역",
}
FUEL_CATEGORIES = {"주유", "주유소", "LPG", "충전소"}
MATCH_ALIASES = {
    "MGC메가커피": "메가커피",
    "메가MGC커피": "메가커피",
    "투썸플레이트": "투썸플레이스",
    "파리바게트": "파리바게뜨",
    "파스쿠치": "파스쿠찌",
    "디즈니 플러스": "디즈니+",
    "디즈니플러스": "디즈니+",
    "SSG COM": "SSG.COM",
    "할리스커피": "할리스",
    "삼성 페이": "삼성페이",
    "SSGPAY": "SSG PAY",
    "SSG페이": "SSG PAY",
    "L.pay": "L.PAY",
    "L페이": "L.PAY",
    "HD현대오일뱅크": "현대오일뱅크",
    "S-OIL": "S-Oil",
}

# ── 시스템 프롬프트 ─────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a Korean credit card benefit simulator.
Your job is to calculate the maximum possible savings
a user could have earned over 1 year using a specific card.
You must respond ONLY with a valid JSON object.
No explanation, no markdown, no code blocks."""

# ── 유저 프롬프트 템플릿 ─────────────────────────────────────
USER_PROMPT_TEMPLATE = """\
아래 카드의 혜택 정보와 사용자의 월별 거래 집계를 바탕으로
이 카드를 사용했을 때의 연간 최대 절감액을 계산해라.

[카드 정보]
{card_benefits_json}

[사용자 월별 거래 집계]
{aggregated_transactions_json}

[절대 규칙 - 반드시 준수]
- discount_value는 퍼센트(%) 단위다.
  예: discount_value: 7  → 7% 할인  → 10,000원 × 7%  = 700원 할인
  예: discount_value: 1.2 → 1.2% 할인 → 10,000원 × 1.2% = 120원 할인
  예: discount_value: 0.7 → 0.7% 할인 → 10,000원 × 0.7% = 70원 할인
  예: discount_value: 0.8 → 0.8% 할인 → 10,000원 × 0.8% = 80원 할인
  0.7을 7%로, 0.8을 8%로, 2.0을 20%로 해석하면 안 된다.
  절대로 소수나 배율로 혼동하지 마라. 7은 700%가 아니라 7%다.
- monthly_breakdown의 각 월 discount 합계가 반드시 total_discount와 일치해야 한다.
  total_discount = sum(monthly_breakdown[i].discount for all i)
- net_saving = total_discount - annual_fee
- benefit_breakdown은 benefit_id 기준으로 연간 합산해서 기록해라.
  같은 benefit_id를 월별로 반복하지 마라. 1개의 benefit_id = 1개의 항목.
- benefit_breakdown의 total_discount 합계는 반드시 total_discount 전체와 일치해야 한다.
  sum(benefit_breakdown[i].total_discount for all i) = total_discount
- total_discount는 연간 총 거래금액의 30%를 절대 초과할 수 없다.
  monthly_summary의 month_total 합계가 연간 총 거래금액이다.
  초과하면 계산이 잘못된 것이므로 처음부터 다시 계산해라.
- benefit_breakdown의 total_discount 합계와 monthly_breakdown의 discount 합계가
  서로 다르면 무조건 잘못된 결과다. 반드시 다시 계산해라.
- discount_type이 amount이면 discount_value는 퍼센트가 아니라 원 단위 정액 혜택이다.
  예: discount_type amount, discount_value 3500 → 3,500원 할인이지 3500% 할인이 아니다.
- discount_type이 point이면 현금 절감액으로 환산하지 말고 보수적으로 0 처리해라.
- category/select_option/conditions.notes 중 하나라도 해외 전용 혜택임을 나타내면
  국내 거래에는 적용하지 마라.
- merchant_type ALL이어도 간편결제, 앱카드, FAN페이처럼 결제수단 조건이 붙은 혜택은
  거래내역에서 해당 결제수단을 확인할 수 없으므로 보수적으로 0 처리해라.

[시뮬레이션 원칙]

1. 사용자에게 가장 유리한 방향으로 계산한다.
   - SELECT형 카드는 각 옵션별로 시뮬레이션 후 가장 유리한 옵션 선택
   - 복수의 혜택이 중복 적용될 수 없을 때 금액이 큰 쪽 선택

2. 월별로 독립 계산한다.
   - 전월실적 조건: prev_month_total이 previous_month_requirement 이상이면 충족
   - tiered_monthly_limit이 있으면 prev_month_total 기준으로 한도 구간 선택
   - monthly_limit은 해당 월에만 적용되고 다음 달 초기화

3. 거래별 혜택 적용 판단
   - merchant_type ALL: transactions_by_merchant 전체 적용
   - merchant_type A/B/C: matched_merchant가 해당 benefit의
     merchants 목록에 포함될 때만 적용
   - merchant_type SERVICE: 이 카드 정보에는 SERVICE 타입이 이미 제거되어 있음
   - offline_only true인 혜택: 쿠팡, 배달의민족, 쿠팡이츠, 요기요,
     네이버쇼핑, G마켓, 옥션, 11번가, SSG.COM, 컬리는 온라인으로 판단해서 제외
   - per_transaction_min 조건: 건당 평균금액(total_amount/count)이
     per_transaction_min 미만이면 해당 혜택 적용 제외
   - tiered_discount가 있는 혜택: 건당 평균금액(total_amount/count) 기준으로
     해당 구간의 할인율을 적용해라. 월 총액이 아닌 건당 금액 기준임에 주의.

4. 월 할인 계산 순서
   a. 전월실적 충족 여부 확인 → 미충족 시 해당 월 전체 혜택 0
   b. rate 혜택은 각 가맹점별 할인액 계산 (거래금액 × discount_value%)
      amount 혜택은 건당 정액 할인 계산
   c. 월 한도까지만 누적 (한도 초과분 제거)
   d. 해당 월 총 할인액 확정

5. 연간 합산
   - total_discount = monthly_breakdown의 discount 합계
   - net_saving = total_discount - annual_fee

6. 계산 불가능한 경우
   - 거래 데이터가 없으면 net_saving: 0
   - 혜택 조건 불명확 시 보수적으로 0 처리

[출력 JSON 구조]
{{
  "card_name": "카드명",
  "card_company": "카드사명",
  "selected_options": {{
    "SELECT_1": "선택된 옵션명 또는 null"
  }},
  "annual_fee": 20000,
  "total_discount": 87000,
  "net_saving": 67000,
  "monthly_breakdown": [
    {{
      "month": "2025-01",
      "month_total": 480000,
      "prev_month_total": 400000,
      "requirement_met": true,
      "discount": 7200,
      "limit_hit": false
    }}
  ],
  "benefit_breakdown": [
    {{
      "benefit_id": "b1",
      "category": "편의점",
      "merchant_type": "B",
      "total_applied_amount": 120000,
      "total_discount": 8400,
      "applied_count": 60
    }}
  ],
  "calculation_notes": "SELECT 옵션 선택 이유 등 계산 근거 요약"
}}"""


# ── 사전 집계 함수 ──────────────────────────────────────────
def aggregate_transactions(
    transactions: list[dict],
    previous_month_requirement: int,
) -> dict:
    monthly: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "merchants": defaultdict(lambda: {"count": 0, "total": 0})}
    )

    for tx in transactions:
        raw_date = tx["date"]
        parts = raw_date.replace(".", "-").split("-")
        month_key = f"{parts[0]}-{parts[1]}"
        amount = int(tx.get("amount", 0))
        merchant = tx.get("matched_merchant", "")

        monthly[month_key]["total"] += amount
        monthly[month_key]["merchants"][merchant]["count"] += 1
        monthly[month_key]["merchants"][merchant]["total"] += amount

    sorted_months = sorted(monthly.keys())
    monthly_summary = []
    prev_total = previous_month_requirement

    for month in sorted_months:
        data = monthly[month]
        tx_by_merchant = [
            {
                "matched_merchant": m,
                "count": v["count"],
                "total_amount": v["total"],
            }
            for m, v in sorted(data["merchants"].items(), key=lambda x: -x[1]["total"])
        ]
        monthly_summary.append({
            "month":                    month,
            "month_total":              data["total"],
            "prev_month_total":         prev_total,
            "transactions_by_merchant": tx_by_merchant,
        })
        prev_total = data["total"]

    return {"monthly_summary": monthly_summary}


def normalize_merchant(merchant: str | None) -> str:
    if not merchant:
        return ""
    return MATCH_ALIASES.get(merchant, merchant)


def parse_requirement_from_notes(notes: str) -> int | None:
    m = re.search(r"전월\s*(?:이용금액|실적)?\s*(\d+)\s*만원", notes or "")
    if not m:
        return None
    return int(m.group(1)) * 10000


def is_overseas_benefit(benefit: dict) -> bool:
    conditions = benefit.get("conditions") or {}
    text = " ".join(
        str(value or "")
        for value in (
            benefit.get("category"),
            benefit.get("select_option"),
            conditions.get("notes"),
        )
    )
    return any(keyword in text for keyword in OVERSEAS_KEYWORDS)


def is_unknown_channel_benefit(benefit: dict) -> bool:
    conditions = benefit.get("conditions") or {}
    text = " ".join(
        str(value or "")
        for value in (
            benefit.get("category"),
            benefit.get("select_option"),
            conditions.get("notes"),
        )
    )
    return any(keyword in text for keyword in UNKNOWN_CHANNEL_KEYWORDS)


def benefit_text(benefit: dict) -> str:
    conditions = benefit.get("conditions") or {}
    return " ".join(
        str(value or "")
        for value in (
            benefit.get("category"),
            benefit.get("select_option"),
            conditions.get("notes"),
        )
    )


def is_area_rank_all_benefit(benefit: dict) -> bool:
    if benefit.get("merchant_type") != "ALL":
        return False
    text = benefit_text(benefit)
    return any(keyword in text for keyword in AREA_RANK_KEYWORDS)


def is_suspicious_fuel_rate_benefit(benefit: dict) -> bool:
    if benefit.get("discount_type") != "rate":
        return False

    category_text = str(benefit.get("category") or "")
    note_text = str((benefit.get("conditions") or {}).get("notes") or "")
    if not any(keyword in category_text or keyword in note_text for keyword in FUEL_CATEGORIES):
        return False

    value = benefit.get("discount_value") or 0
    explicit_percent = "%" in note_text or "％" in note_text or "퍼센트" in note_text
    per_liter_unit = bool(re.search(r"(?:원\s*/\s*[Llℓ]|[Llℓ]\s*당|리터\s*당|리터당)", note_text))

    return value >= 20 and (per_liter_unit or not explicit_percent)


def is_unsupported_benefit(benefit: dict) -> bool:
    return is_area_rank_all_benefit(benefit) or is_suspicious_fuel_rate_benefit(benefit)


def benefit_requirement(card_prev_req: int, benefit: dict) -> int:
    notes = (benefit.get("conditions") or {}).get("notes", "")
    noted_req = parse_requirement_from_notes(notes)
    return max(card_prev_req or 0, noted_req or 0)


def month_key(date_text: str) -> str:
    parts = date_text.replace(".", "-").split("-")
    return f"{parts[0]}-{parts[1]}"


def transactions_by_month(transactions: list[dict]) -> dict[str, list[dict]]:
    monthly: dict[str, list[dict]] = defaultdict(list)
    for tx in transactions:
        monthly[month_key(tx["date"])].append(tx)
    return dict(sorted(monthly.items()))


def merchant_matches(tx: dict, benefit: dict) -> bool:
    merchant_type = benefit.get("merchant_type")
    if merchant_type == "SERVICE":
        return False
    if is_unsupported_benefit(benefit):
        return False
    if merchant_type == "ALL":
        return not is_overseas_benefit(benefit) and not is_unknown_channel_benefit(benefit)

    merchant = normalize_merchant(tx.get("matched_merchant"))
    eligible = {normalize_merchant(m) for m in benefit.get("merchants", [])}
    return merchant in eligible


def tiered_monthly_limit(prev_total: int, benefit: dict) -> int | None:
    conditions = benefit.get("conditions") or {}
    tiers = conditions.get("tiered_monthly_limit")
    if not tiers:
        return conditions.get("monthly_limit")

    chosen = None
    for tier in sorted(tiers, key=lambda x: x.get("min_spend", 0)):
        if prev_total >= (tier.get("min_spend") or 0):
            chosen = tier.get("limit")
    return chosen


def rate_for_amount(amount: int, benefit: dict) -> float:
    tiers = benefit.get("tiered_discount")
    if tiers:
        for tier in tiers:
            min_amount = tier.get("min_amount") or 0
            max_amount = tier.get("max_amount")
            if amount >= min_amount and (max_amount is None or amount <= max_amount):
                return (tier.get("rate") or 0) / 100
        return 0
    return (benefit.get("discount_value") or 0) / 100


def calculate_benefit(
    benefit: dict,
    transactions: list[dict],
    card_prev_req: int,
) -> dict:
    conditions = benefit.get("conditions") or {}
    per_tx_min = conditions.get("per_transaction_min")
    offline_only = conditions.get("offline_only") is True
    discount_type = benefit.get("discount_type")
    discount_value = benefit.get("discount_value") or 0
    monthly = transactions_by_month(transactions)
    prev_total = card_prev_req or 0

    monthly_discount: dict[str, int] = {}
    merchant_stats: dict[str, dict] = defaultdict(lambda: {"count": 0, "total": 0})
    total_applied_amount = 0
    applied_count = 0

    for month, rows in monthly.items():
        required = benefit_requirement(card_prev_req, benefit)
        discount = 0.0
        applied_amount = 0
        count = 0

        if prev_total >= required:
            for tx in rows:
                amount = int(tx.get("amount", 0))
                merchant = tx.get("matched_merchant")

                if not merchant_matches(tx, benefit):
                    continue
                if offline_only and normalize_merchant(merchant) in ONLINE_MERCHANTS:
                    continue
                if per_tx_min and amount < per_tx_min:
                    continue

                if discount_type == "point":
                    continue
                if discount_type == "amount":
                    discount += min(amount, discount_value)
                elif discount_type == "rate":
                    discount += amount * rate_for_amount(amount, benefit)
                else:
                    continue
                applied_amount += amount
                count += 1
                merchant_key = tx.get("matched_merchant") or "기타"
                merchant_stats[merchant_key]["count"] += 1
                merchant_stats[merchant_key]["total"] += amount

            monthly_limit = tiered_monthly_limit(prev_total, benefit)
            if monthly_limit is not None and (discount_type != "amount" or monthly_limit >= 1000):
                discount = min(discount, monthly_limit)

        monthly_discount[month] = round(discount)
        total_applied_amount += applied_amount
        applied_count += count
        prev_total = sum(int(tx.get("amount", 0)) for tx in rows)

    total_discount = sum(monthly_discount.values())
    return {
        "benefit_id": benefit.get("benefit_id", "unknown"),
        "category": benefit.get("category", ""),
        "merchant_type": benefit.get("merchant_type", ""),
        "select_group": benefit.get("select_group"),
        "select_option": benefit.get("select_option"),
        "total_applied_amount": total_applied_amount,
        "total_discount": total_discount,
        "applied_count": applied_count,
        "monthly_discount": monthly_discount,
        "merchant_stats": [
            {"merchant": merchant, "count": s["count"], "total": s["total"]}
            for merchant, s in sorted(
                merchant_stats.items(),
                key=lambda x: (-x[1]["total"], x[0]),
            )
        ],
    }


def select_benefit_results(results: list[dict]) -> tuple[list[dict], dict[str, str | None]]:
    fixed = [r for r in results if not r.get("select_group")]
    selected_options: dict[str, str | None] = {}

    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        group = r.get("select_group")
        option = r.get("select_option")
        if group:
            grouped[group][option or ""] .append(r)

    for idx, (group, options) in enumerate(grouped.items(), 1):
        best_option = None
        best_results: list[dict] = []
        best_total = -1
        for option, option_results in options.items():
            option_total = sum(r["total_discount"] for r in option_results)
            if option_total > best_total:
                best_total = option_total
                best_option = option or None
                best_results = option_results
        selected_options[f"SELECT_{idx}"] = best_option
        fixed.extend(best_results)

    if not selected_options:
        selected_options["SELECT_1"] = None

    return fixed, selected_options


def deterministic_simulate_card(card_info: dict, transactions: list[dict], note: str) -> dict:
    prev_req = card_info.get("previous_month_requirement") or 0
    benefit_results = [
        calculate_benefit(b, transactions, prev_req)
        for b in card_info.get("benefits", [])
        if b.get("merchant_type") != "SERVICE"
    ]
    selected_results, selected_options = select_benefit_results(benefit_results)

    monthly = transactions_by_month(transactions)
    monthly_breakdown = []
    prev_total = prev_req
    for month, rows in monthly.items():
        month_total = sum(int(tx.get("amount", 0)) for tx in rows)
        discount = sum(r["monthly_discount"].get(month, 0) for r in selected_results)
        monthly_breakdown.append({
            "month": month,
            "month_total": month_total,
            "prev_month_total": prev_total,
            "requirement_met": True,
            "discount": round(discount),
            "limit_hit": False,
        })
        prev_total = month_total

    total_discount = sum(m["discount"] for m in monthly_breakdown)
    annual_fee = card_info.get("annual_fee") or 0
    benefit_breakdown = [
        {
            "benefit_id": r["benefit_id"],
            "category": r["category"],
            "merchant_type": r["merchant_type"],
            "total_applied_amount": r["total_applied_amount"],
            "total_discount": r["total_discount"],
            "applied_count": r["applied_count"],
            "matched_merchants": r.get("merchant_stats", []),
        }
        for r in selected_results
        if r["total_discount"] > 0
    ]

    return {
        "card_name": card_info["card_name"],
        "card_company": card_info["card_company"],
        "selected_options": selected_options,
        "annual_fee": annual_fee,
        "total_discount": round(total_discount),
        "net_saving": round(total_discount - annual_fee),
        "monthly_breakdown": monthly_breakdown,
        "benefit_breakdown": benefit_breakdown,
        "calculation_notes": note,
    }


# ── SERVICE 타입 사전 필터링 ─────────────────────────────────
def filter_card_for_prompt(card_info: dict) -> tuple[dict, int]:
    """
    LLM에 넘기기 전 card_info를 정제한다:
    1. eligible_merchants 제거 (토큰 절약)
    2. SERVICE 타입 혜택 제거 (연 1회 바우처 등 LLM이 계산 불가)
    반환: (정제된 card_dict, 제거된 SERVICE 혜택 수)
    """
    card = {k: v for k, v in card_info.items() if k != "eligible_merchants"}

    original_benefits = card.get("benefits", [])
    non_service = [b for b in original_benefits if b.get("merchant_type") != "SERVICE"]
    removed_count = len(original_benefits) - len(non_service)

    card = dict(card)
    card["benefits"] = non_service

    return card, removed_count


# ── JSON 추출 ────────────────────────────────────────────────
def extract_json(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if m:
        return m.group(1).strip()
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1:
        return text[start:end + 1]
    return text


# ── 유효성 검사 + total_discount/net_saving 코드 재계산 ──────
def validate_and_fix(data: dict, max_expected_discount: int | None = None) -> tuple[bool, str]:
    if not isinstance(data.get("net_saving"), (int, float)):
        return False, "net_saving 필드 없거나 숫자 아님"
    if not isinstance(data.get("monthly_breakdown"), list):
        return False, "monthly_breakdown 배열 없음"

    # monthly_breakdown 합계로 재계산 (LLM 산수 오류 보정)
    monthly_sum = sum(m.get("discount", 0) for m in data["monthly_breakdown"])
    benefit_sum = sum(b.get("total_discount", 0) for b in data.get("benefit_breakdown", []))
    if abs(monthly_sum - benefit_sum) > 1:
        return False, f"월별 할인 합계({monthly_sum})와 혜택별 할인 합계({benefit_sum}) 불일치"
    if max_expected_discount is not None:
        tolerance = 1000 if max_expected_discount > 0 else 0
        if monthly_sum > max_expected_discount + tolerance:
            return False, (
                f"할인액 과대 계산: {monthly_sum}원 > 코드 검산 상한 {max_expected_discount}원"
            )

    annual_fee  = data.get("annual_fee", 0) or 0
    data["total_discount"] = round(monthly_sum)
    data["net_saving"]     = round(monthly_sum - annual_fee)

    return True, "OK"


def validate_against_deterministic(data: dict, deterministic: dict) -> tuple[bool, str]:
    expected_breakdown = [
        b for b in deterministic.get("benefit_breakdown", [])
        if b.get("total_discount", 0) > 0
    ]
    parsed_breakdown = [
        b for b in data.get("benefit_breakdown", [])
        if b.get("total_discount", 0) > 0
    ]

    expected_ids = {b.get("benefit_id") for b in expected_breakdown}
    parsed_ids = {b.get("benefit_id") for b in parsed_breakdown}
    if expected_ids != parsed_ids:
        return False, f"혜택별 breakdown ID 불일치: expected={sorted(expected_ids)} parsed={sorted(parsed_ids)}"

    expected_total = deterministic.get("total_discount", 0)
    parsed_total = data.get("total_discount", 0)
    if expected_total > 0 and parsed_total < expected_total * 0.75:
        return False, f"할인액 과소 계산 의심: {parsed_total}원 < 코드 검산 {expected_total}원의 75%"

    return True, "OK"


# ── 0건 카드용 기본 결과 ─────────────────────────────────────
def make_zero_result(card: dict, note: str = "거래 데이터 없음") -> dict:
    annual_fee = card.get("annual_fee") or 0
    return {
        "card_name":         card["card_name"],
        "card_company":      card["card_company"],
        "selected_options":  {},
        "annual_fee":        annual_fee,
        "total_discount":    0,
        "net_saving":        -annual_fee,
        "monthly_breakdown": [],
        "benefit_breakdown": [],
        "calculation_notes": note,
    }


# ── 카드 1장 시뮬레이션 ──────────────────────────────────────
def simulate_card(
    client: OpenAI,
    card_info: dict,
    transactions: list[dict],
) -> tuple[dict | None, str]:
    prev_req   = card_info.get("previous_month_requirement") or 0
    aggregated = aggregate_transactions(transactions, prev_req)

    # ── 핵심 수정: SERVICE 타입 사전 제거 ───────────────────
    card_for_prompt, removed = filter_card_for_prompt(card_info)

    # SERVICE만 있고 남은 혜택이 없으면 LLM 호출 없이 0 반환
    if not card_for_prompt.get("benefits"):
        return make_zero_result(
            card_info,
            note=f"모든 혜택이 SERVICE 타입 (바우처/멤버십 등 {removed}개) — 계산 불가"
        ), "SERVICE_ONLY"

    deterministic = deterministic_simulate_card(
        card_info,
        transactions,
        note="LLM 결과 검산 실패 시 사용되는 코드 기반 보수 계산",
    )
    max_expected_discount = deterministic.get("total_discount", 0)

    user_msg = USER_PROMPT_TEMPLATE.format(
        card_benefits_json=json.dumps(card_for_prompt, ensure_ascii=False, indent=2),
        aggregated_transactions_json=json.dumps(aggregated, ensure_ascii=False, indent=2),
    )

    last_error = ""
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw    = resp.choices[0].message.content or ""
            parsed = json.loads(extract_json(raw))
            ok, reason = validate_and_fix(parsed, max_expected_discount=max_expected_discount)
            if ok:
                ok, reason = validate_against_deterministic(parsed, deterministic)
            if ok:
                # SERVICE 제거 사실을 notes에 기록
                if removed > 0:
                    existing = parsed.get("calculation_notes", "")
                    parsed["calculation_notes"] = (
                        f"[SERVICE 타입 혜택 {removed}개 시뮬레이션 제외] " + existing
                    )
                return parsed, "OK"
            last_error = reason

        except json.JSONDecodeError as e:
            last_error = f"JSON 파싱 오류: {e}"
        except Exception as e:
            last_error = f"API 오류: {e}"

        if attempt < MAX_RETRIES:
            print(f"\n      재시도 {attempt + 1}/{MAX_RETRIES} ({last_error})", flush=True)
            time.sleep(SLEEP_SEC)

    deterministic["calculation_notes"] = (
        f"[코드 기반 보수 계산 사용: {last_error}] "
        + deterministic.get("calculation_notes", "")
    )
    if removed > 0:
        deterministic["calculation_notes"] = (
            f"[SERVICE 타입 혜택 {removed}개 시뮬레이션 제외] "
            + deterministic["calculation_notes"]
        )
    return deterministic, "DETERMINISTIC_FALLBACK"


def simulate_card_worker(
    idx: int,
    total: int,
    api_key: str,
    card_info: dict,
    cards_in_map: dict,
) -> dict:
    name = card_info["card_name"]
    company = card_info["card_company"]
    tx_list = cards_in_map.get(name, {}).get("transactions", [])
    tx_count = len(tx_list)

    if tx_count == 0:
        return {
            "idx": idx,
            "total": total,
            "card_name": name,
            "card_company": company,
            "tx_count": tx_count,
            "reason": "SKIPPED",
            "result": make_zero_result(card_info),
        }

    try:
        client = OpenAI(api_key=api_key)
        sim_result, reason = simulate_card(client, card_info, tx_list)
    except Exception as e:
        sim_result = None
        reason = f"WORKER_ERROR: {e}"

    if sim_result is None:
        sim_result = make_zero_result(card_info, note=f"시뮬레이션 실패: {reason}")
        reason = "FAILED"

    return {
        "idx": idx,
        "total": total,
        "card_name": name,
        "card_company": company,
        "tx_count": tx_count,
        "reason": reason,
        "result": sim_result,
    }


# ── 메인 ─────────────────────────────────────────────────────
def main():
    if not DEBUG_MODE:
        print("디버그 모드 비활성화 — app.py 메모리 파이프라인으로 실행하세요.")
        print("독립 실행하려면 DEBUG_MODE = True 로 변경하세요.")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

    with open(CTM_JSON, encoding="utf-8") as f:
        ctm: dict = json.load(f)
    with open(STRUCTURED_JSON, encoding="utf-8") as f:
        structured: list[dict] = json.load(f)

    cards_in_map = ctm["cards"]

    total = len(structured)

    print(f"\n{'='*62}")
    print(f"  절감액 시뮬레이션 시작 — {total}개 카드, 모델: {MODEL}")
    print(f"  병렬 처리 workers: {MAX_WORKERS}개")
    print(f"{'='*62}")

    results: list[dict] = []
    cnt_success = cnt_failed = cnt_skipped = cnt_service_only = cnt_fallback = 0

    futures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for idx, card_info in enumerate(structured, 1):
            futures.append(
                executor.submit(
                    simulate_card_worker,
                    idx,
                    total,
                    api_key,
                    card_info,
                    cards_in_map,
                )
            )

        for done_count, future in enumerate(as_completed(futures), 1):
            item = future.result()
            result = item["result"]
            reason = item["reason"]
            results.append(result)

            if reason == "SKIPPED":
                cnt_skipped += 1
                status = f"SKIP (거래 0건, net_saving: {result['net_saving']:,})"
            elif reason == "SERVICE_ONLY":
                cnt_service_only += 1
                status = "SERVICE_ONLY → 0 처리"
            elif reason == "DETERMINISTIC_FALLBACK":
                cnt_fallback += 1
                status = (
                    f"FALLBACK  총할인 {result.get('total_discount', 0):>7,}원  "
                    f"순절감 {result.get('net_saving', 0):>7,}원"
                )
            elif reason == "FAILED" or str(reason).startswith("WORKER_ERROR"):
                cnt_failed += 1
                status = f"실패 → 0 대체 ({reason})"
            else:
                cnt_success += 1
                status = (
                    f"OK  총할인 {result.get('total_discount', 0):>7,}원  "
                    f"순절감 {result.get('net_saving', 0):>7,}원"
                )

            print(
                f"[완료 {done_count:>3}/{total}] "
                f"[카드 {item['idx']:>3}/{total}] "
                f"{item['card_company'][:8]:<8} / {item['card_name'][:35]:<35} "
                f"({item['tx_count']}건) {status}",
                flush=True,
            )

    # 정렬 및 rank 부여
    results.sort(key=lambda x: x.get("net_saving", 0), reverse=True)
    ranked = [{"rank": rank, **r} for rank, r in enumerate(results, 1)]

    # ── 검증 출력 ─────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  시뮬레이션 완료 [전체 실행]")
    print(f"  처리 카드 수    : {total}개")
    print(f"  API 성공        : {cnt_success}개")
    print(f"  코드 검산 대체  : {cnt_fallback}개")
    print(f"  실패(0 대체)    : {cnt_failed}개")
    print(f"  스킵(거래 0건)  : {cnt_skipped}개")
    print(f"  SERVICE 전용    : {cnt_service_only}개 (바우처 등 계산 불가)")
    # DEBUG_MODE일 때만 파일 저장
    output_path = DEBUG_OUTPUT_DIR / "simulation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"simulated_at": str(__import__("datetime").date.today()),
                   "total_cards": total, "results": ranked},
                  f, ensure_ascii=False, indent=2)
    print(f"  저장 위치       : {output_path}")

    print(f"\n  {'─'*60}")
    print(f"  net_saving Top 10")
    print(f"  {'─'*60}")
    print(f"  {'순위':<4} {'카드명':<36} {'카드사':<10} {'총할인':>9} {'연회비':>8} {'순절감':>9}")
    print(f"  {'─'*60}")
    for r in ranked[:10]:
        print(
            f"  {r['rank']:<4} {r.get('card_name','')[:34]:<34} "
            f"{r.get('card_company','')[:9]:<9} "
            f"{r.get('total_discount',0):>9,} "
            f"{r.get('annual_fee',0):>8,} "
            f"{r.get('net_saving',0):>9,}"
        )

    negative = [r for r in ranked if r.get("net_saving", 0) < 0]
    print(f"\n  net_saving 마이너스 카드: {len(negative)}개")
    for r in negative:
        print(f"    {r.get('card_company',''):<10} {r.get('card_name','')}  ({r.get('net_saving',0):,}원)")

    zero_ns = [r for r in ranked if r.get("net_saving", 0) == 0]
    print(f"\n  net_saving 0원 카드: {len(zero_ns)}개")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
