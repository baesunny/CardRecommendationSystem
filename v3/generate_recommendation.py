"""
Top5 카드 데이터 + 필터링된 거래 데이터 + card_benefits_structured.json
→ OpenAI API
Top 5 카드에 대한 상세 추천 설명 생성 (사용자 최종 출력용)
"""

import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from simulate_savings import calculate_benefit

BASE_DIR         = Path(__file__).resolve().parent
OUTPUT_DIR       = BASE_DIR / "output"        # 고정 데이터 (카드 DB)
DEBUG_OUTPUT_DIR = BASE_DIR / "debug_output"  # 디버그 모드 산출물
STRUCTURED_JSON  = OUTPUT_DIR / "card_benefits_structured.json"
FILTERED_JSON    = DEBUG_OUTPUT_DIR / "filtered_transactions.json"

# ── 디버그 모드 ─────────────────────────────────────────────
# True : 독립 실행 가능. top5_cards.json + filtered_transactions.json 읽고 recommendation.json 저장
# False: app.py 메모리 파이프라인용 (파일 입출력 없음)
DEBUG_MODE = True

MODEL       = "gpt-4o"
MAX_RETRIES = 2
SLEEP_SEC   = 1.0
MAX_WORKERS = int(os.environ.get("REC_MAX_WORKERS", "5"))

# ── 시스템 프롬프트 (CoT 포함) ──────────────────────────────
SYSTEM_PROMPT = """\
You are a Korean credit card recommendation expert writing final user-facing content.
Your output will be shown directly to the user as a personalized card recommendation.

Before writing the recommendation for each card, reason through these steps:
1. Look at every benefit-specific transaction summary provided for this card.
2. Cross-reference each benefit with the merchants where actual spending was applied.
3. Use only the provided benefit_breakdown and applied transaction summaries.
4. Check the annual fee — does the total saving exceed it? By how much?
5. Identify the strongest reason this card fits this user's spending pattern.

Then write a natural, specific Korean recommendation based on this reasoning.

Rules:
- Never say "귀하". Use "사용자" or write directly without subject.
- Be concrete: always name the specific merchants and amounts.
- Do not invent numbers. Only use figures from the provided data.
- Respond ONLY with a valid JSON object. No markdown, no code blocks."""

# ── 유저 프롬프트 템플릿 ─────────────────────────────────────
USER_PROMPT_TEMPLATE = """\
아래 데이터를 분석해서 이 카드를 추천하는 상세한 설명을 작성해라.
이 텍스트는 사용자가 직접 읽는 최종 결과물이다.

[카드 정보]
카드명: {card_name}
카드사: {card_company}
연회비: {annual_fee}원
연간 순 절감액: {net_saving}원
연간 총 할인액: {total_discount}원

[이 카드의 혜택 상세 — 혜택별 적용 가맹점 목록]
{benefit_structure_str}

[카드 혜택별 절감 내역 — benefit_breakdown]
{benefit_breakdown_json}

[혜택별 실제 적용 거래 집계 — 전체]
아래 데이터는 이 카드의 각 혜택에 실제로 매칭된 거래를 가맹점별로 전부 집계한 것이다.
my_spending과 matched_merchants는 반드시 이 데이터를 기준으로 작성해라.
{benefit_usage_str}

[월별 절감 내역]
{monthly_breakdown_json}

[시뮬레이션 계산 근거]
{calculation_notes}

[작성 규칙]

규칙 0: benefit_details에는 실제 절감이 발생한 혜택만 포함해라.
  estimated_saving이 0이거나 사용자의 이용 내역이 없어서
  절감액이 발생하지 않는 혜택은 benefit_details에서 제외해라.
  단, summary에서는 "이 혜택은 이용 내역이 없어 이번에 적용되지 않았지만,
  향후 [가맹점] 이용 시 절감 가능" 식으로 간단히 언급해도 좋다.

규칙 1: summary는 반드시 4~6문장으로 작성해라.
  구조:
  - 1문장: 이 사용자의 소비 패턴 중 이 카드와 가장 잘 맞는 핵심 포인트
  - 2~3문장: 가장 큰 혜택 2가지를 구체적 가맹점명 + 금액으로 설명
  - 1문장: 연회비 대비 효율 (몇 배 혜택인지 또는 몇 개월 만에 회수)
  - 1문장: 이 카드를 선택해야 하는 결론적 이유

  나쁜 예: "이 카드는 다양한 혜택을 제공합니다."
  좋은 예: "편의점을 연 82건 이용하는 소비 패턴에 이 카드의 편의점 7% 할인이 정확히 맞아떨어집니다.
            CU에서 연간 587,300원을 지출했는데, 7% 적용 시 약 41,111원을 절감할 수 있습니다.
            통신비 LG U+ 608,210원에도 10% 할인이 적용되어 약 60,821원이 추가로 줄어듭니다.
            연회비 20,000원 대비 연간 절감액이 훨씬 크므로 연회비는 첫 달 만에 회수됩니다.
            편의점과 통신비 지출이 많은 사용자라면 이 카드가 가장 효율적인 선택입니다."

규칙 2: benefit_details 항목 수 = benefit_breakdown 항목 수. 합치거나 추가하지 마라.

규칙 3: my_spending은 [혜택별 실제 적용 거래 집계]에 있는 해당 benefit_id의
  적용 거래 합계와 가맹점별 전체 집계를 기준으로 작성해라.
  별도 소비 순위 요약을 만들거나 사용하지 마라.

규칙 4: matched_merchants는 [혜택별 실제 적용 거래 집계]에 있는 해당 benefit_id의
  가맹점 전체를 빠짐없이 요약해라.
  적용 거래 합계가 0보다 큰 혜택에 "이용 내역 없음"이라고 쓰면 안 된다.
  ALL 타입: "모든 가맹점에서 적용됩니다."

규칙 5: estimated_saving = benefit_breakdown의 total_discount 그대로.
  calculation은 그 금액이 나오도록 작성해라.

규칙 6: condition은 benefit_structure 또는 benefit_breakdown에 있는 내용만.
  100% 할인 등 데이터에 없는 조건 금지.
  discount_type이 amount이면 discount_value는 퍼센트가 아니라 원 단위 정액 혜택이다.
  예: discount_type amount, discount_value 3500 → "3,500원 할인"이지 "3500% 할인"이 아니다.
  discount_type이 point이면 현금 절감액 계산이 불확실하므로 benefit_breakdown에 있는 total_discount만 인용하고,
  임의의 퍼센트 계산식을 만들지 마라.

규칙 7: total_summary에는 연간 총 할인액과 연회비 차감 후 순절감액을 구분해서 써라.
  연간 총 할인액 = benefit_details estimated_saving 합계.
  순절감액 = net_saving.
  두 값을 섞어서 쓰지 말고, 월 평균은 연간 총 할인액 기준으로 포함해라.

[출력 JSON 구조]
{{
  "card_name": "카드명",
  "card_company": "카드사명",
  "rank": {rank},
  "net_saving": {net_saving},
  "annual_fee": {annual_fee},
  "summary": "4~6문장 상세 추천 이유 (구체적 가맹점명과 금액 포함)",
  "benefit_details": [
    {{
      "benefit_name": "편의점 할인",
      "condition": "CU, GS25 결제 시 7% 할인 (월 한도 5,000원)",
      "matched_merchants": "사용자는 CU를 연 60건(총 635,800원) 이용했는데, 이 카드는 CU에서 7% 혜택을 제공하므로 실제 절감이 발생합니다.",
      "my_spending": "CU 연 60건, 총 635,800원",
      "calculation": "635,800원 × 7% = 44,506원 → 월 한도 5,000원 × 12개월 = 60,000원 적용",
      "estimated_saving": 60000
    }}
  ],
  "total_summary": "연간 총 할인액과 연회비 차감 후 순절감액을 구분해서 작성. 월 평균 포함.",
  "fee_recovery": "연회비 있으면 몇 개월 만에 회수. 없으면 순수 이득 명시.",
  "cautions": "핵심 조건만. 없으면 null"
}}"""


def get_benefit_structure(card_name: str, structured_map: dict) -> str:
    card = structured_map.get(card_name)
    if not card:
        return "혜택 구조 데이터 없음"
    lines = []
    for b in card.get("benefits", []):
        bid      = b.get("benefit_id", "")
        mtype    = b.get("merchant_type", "")
        category = b.get("category", "")
        discount = b.get("discount_value", "")
        discount_type = b.get("discount_type", "")
        conditions = b.get("conditions") or {}
        notes = conditions.get("notes")
        ms       = b.get("merchants", [])
        if discount_type == "rate":
            benefit_desc = f"{discount}%"
        elif discount_type == "amount":
            benefit_desc = f"{int(discount or 0):,}원 정액"
        elif discount_type == "point":
            benefit_desc = "포인트/마일리지형 (현금 환산 불확실)"
        else:
            benefit_desc = "혜택값 불명확"
        note_desc = f" / 조건: {notes}" if notes else ""
        if mtype == "SERVICE":
            lines.append(f"[{bid}] {category} — SERVICE 타입 (가맹점 없음){note_desc}")
        elif mtype == "ALL":
            lines.append(f"[{bid}] {category} — ALL 타입 (모든 가맹점 {benefit_desc}){note_desc}")
        else:
            ms_str = ", ".join(ms)
            lines.append(f"[{bid}] {category} {benefit_desc} — 대상: {ms_str}{note_desc}")
    return "\n".join(lines) if lines else "혜택 구조 데이터 없음"


def build_benefit_usage_summary(
    card_name: str,
    breakdown: list[dict],
    structured_map: dict,
    transactions: list[dict],
) -> str:
    card = structured_map.get(card_name, {})
    benefit_map = {b.get("benefit_id"): b for b in card.get("benefits", [])}
    prev_req = card.get("previous_month_requirement") or 0
    lines: list[str] = []

    for b in breakdown:
        bid = b.get("benefit_id", "")
        benefit = benefit_map.get(bid)
        merchant_rows = b.get("matched_merchants") or []
        if not merchant_rows and benefit:
            calculated = calculate_benefit(benefit, transactions, prev_req)
            merchant_rows = calculated.get("merchant_stats", [])

        merchants = sorted(
            merchant_rows,
            key=lambda x: (-(x.get("total", 0) or 0), x.get("merchant", "")),
        )
        merchant_lines = [
            f"{row.get('merchant', '기타')} 연 {row.get('count', 0)}건 총 {row.get('total', 0):,}원"
            for row in merchants
        ]
        merchant_summary = "; ".join(merchant_lines) if merchant_lines else "가맹점별 집계 없음"
        lines.append(
            f"[{bid}] {b.get('category', '')} / {b.get('merchant_type', '')}\n"
            f"- 시뮬레이션 적용 합계: 연 {b.get('applied_count', 0)}건, "
            f"총 {b.get('total_applied_amount', 0):,}원, "
            f"절감 {b.get('total_discount', 0):,}원\n"
            f"- 적용 가맹점 전체: {merchant_summary}"
        )

    return "\n\n".join(lines) if lines else "적용 거래 집계 없음"


def merge_benefit_breakdown(breakdown: list[dict]) -> list[dict]:
    merged: dict = {}
    for b in breakdown:
        bid = b.get("benefit_id", "unknown")
        if bid not in merged:
            merged[bid] = {
                "benefit_id":           bid,
                "category":             b.get("category", ""),
                "merchant_type":        b.get("merchant_type", ""),
                "total_applied_amount": 0,
                "total_discount":       0,
                "applied_count":        0,
                "matched_merchants":    [],
            }
        merged[bid]["total_applied_amount"] += b.get("total_applied_amount", 0)
        merged[bid]["total_discount"]       += b.get("total_discount", 0)
        merged[bid]["applied_count"]        += b.get("applied_count", 0)
        merged[bid]["matched_merchants"].extend(b.get("matched_merchants", []))
    return list(merged.values())


def sanitize_card(card: dict) -> dict:
    card = dict(card)
    card["net_saving"]     = round(card.get("net_saving", 0))
    card["total_discount"] = round(card.get("total_discount", 0))
    card["annual_fee"]     = round(card.get("annual_fee") or 0)
    return card


def extract_json(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if m:
        return m.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start:end+1] if start != -1 and end != -1 else text


def validate(
    data: dict,
    expected_total_discount: int | None = None,
    expected_detail_count: int | None = None,
) -> tuple[bool, str]:
    if not isinstance(data.get("summary"), str) or not data["summary"].strip():
        return False, "summary 없거나 비어있음"
    if not isinstance(data.get("benefit_details"), list):
        return False, "benefit_details가 배열이 아님"
    if not isinstance(data.get("total_summary"), str) or not data["total_summary"].strip():
        return False, "total_summary 없거나 비어있음"
    details = data.get("benefit_details", [])
    if expected_detail_count is not None and len(details) != expected_detail_count:
        return False, f"benefit_details 개수 불일치 (기대 {expected_detail_count}, 실제 {len(details)})"
    if expected_total_discount is not None:
        detail_sum = sum(d.get("estimated_saving", 0) for d in details)
        if round(detail_sum) != round(expected_total_discount):
            return False, (
                f"benefit_details 합계 불일치 "
                f"(기대 {expected_total_discount}, 실제 {detail_sum})"
            )
    for d in details:
        condition = str(d.get("condition", ""))
        calculation = str(d.get("calculation", ""))
        matched = str(d.get("matched_merchants", ""))
        spending = str(d.get("my_spending", ""))
        saving = d.get("estimated_saving", 0) or 0
        if re.search(r"\b\d{3,}%", condition) or re.search(r"\b\d{3,}%", calculation):
            return False, "비정상 퍼센트 표현 포함"
        if saving > 0 and "모든 가맹점" not in matched:
            if "이용 내역 없음" in matched or "이용 내역 없음" in spending:
                return False, "절감액이 있는데 이용 내역 없음으로 표시됨"
    # summary가 너무 짧으면 재시도 (2문장 미만)
    if data["summary"].count(".") < 2:
        return False, "summary가 너무 짧음 (4~6문장 필요)"
    return True, "OK"


def generate_for_card(
    client: OpenAI,
    card: dict,
    benefit_structure_str: str,
    merged_breakdown: list[dict],
    benefit_usage_str: str,
) -> tuple[dict | None, str]:
    annual_fee = card.get("annual_fee") or 0

    if not merged_breakdown:
        monthly_total = sum(m.get("discount", 0) for m in card.get("monthly_breakdown", []))
        breakdown_to_use = [{
            "benefit_id": "b_fallback", "category": "전체 혜택", "merchant_type": "ALL",
            "total_applied_amount": card.get("total_discount", 0),
            "total_discount": monthly_total, "applied_count": 0,
        }]
    else:
        breakdown_to_use = merged_breakdown

    # total_discount가 0인 항목 제거 (실제 절감 없는 혜택)
    breakdown_to_use = [b for b in breakdown_to_use if b.get("total_discount", 0) > 0]

    # 필터링 후 비어있으면 monthly 합계로 fallback
    if not breakdown_to_use:
        monthly_total = sum(m.get("discount", 0) for m in card.get("monthly_breakdown", []))
        breakdown_to_use = [{
            "benefit_id": "b_fallback", "category": "전체 혜택", "merchant_type": "ALL",
            "total_applied_amount": card.get("total_discount", 0),
            "total_discount": monthly_total, "applied_count": 0,
        }]
    expected_total_discount = round(sum(b.get("total_discount", 0) for b in breakdown_to_use))
    expected_detail_count = len(breakdown_to_use)

    user_msg = USER_PROMPT_TEMPLATE.format(
        card_name=card["card_name"],
        card_company=card["card_company"],
        annual_fee=annual_fee,
        net_saving=card.get("net_saving", 0),
        total_discount=card.get("total_discount", 0),
        rank=card.get("rank", ""),
        benefit_structure_str=benefit_structure_str,
        benefit_breakdown_json=json.dumps(breakdown_to_use, ensure_ascii=False, indent=2),
        benefit_usage_str=benefit_usage_str,
        monthly_breakdown_json=json.dumps(
            card.get("monthly_breakdown", []), ensure_ascii=False, indent=2
        ),
        calculation_notes=card.get("calculation_notes", ""),
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
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            raw    = resp.choices[0].message.content or ""
            parsed = json.loads(extract_json(raw))
            ok, reason = validate(
                parsed,
                expected_total_discount=expected_total_discount,
                expected_detail_count=expected_detail_count,
            )
            if ok:
                parsed["rank"] = card.get("rank", parsed.get("rank"))
                parsed["net_saving"] = card.get("net_saving", parsed.get("net_saving"))
                parsed["annual_fee"] = annual_fee
                parsed["total_summary"] = (
                    f"연간 총 할인액은 {expected_total_discount:,}원이며, "
                    f"연회비 {annual_fee:,}원을 차감한 순절감액은 "
                    f"{card.get('net_saving', 0):,}원입니다. "
                    f"월 평균 총 할인액은 약 {expected_total_discount // 12:,}원입니다."
                )
                return parsed, "OK"
            last_error = f"유효성 실패: {reason}"

        except json.JSONDecodeError as e:
            last_error = f"JSON 파싱 오류: {e}"
        except Exception as e:
            last_error = f"API 오류: {e}"

        if attempt < MAX_RETRIES:
            print(f"      재시도 {attempt+1}/{MAX_RETRIES} ({last_error})")
            time.sleep(SLEEP_SEC)

    return None, last_error


def make_failed_recommendation(card: dict, rank: int, name: str, company: str) -> dict:
    benefit_details = [
        {
            "benefit_name": f"{b.get('category', '혜택')} 혜택",
            "condition": "추천 문장 생성 실패로 시뮬레이션 breakdown 기준 값을 표시합니다.",
            "matched_merchants": "; ".join(
                f"{m.get('merchant')} {m.get('count', 0)}건 {m.get('total', 0):,}원"
                for m in b.get("matched_merchants", [])
            ),
            "my_spending": f"총 {b.get('total_applied_amount', 0):,}원",
            "calculation": f"시뮬레이션 산출 절감액 {b.get('total_discount', 0):,}원",
            "estimated_saving": b.get("total_discount", 0),
        }
        for b in card.get("benefit_breakdown", [])
        if b.get("total_discount", 0) > 0
    ]
    total_discount = sum(d["estimated_saving"] for d in benefit_details)
    annual_fee = card.get("annual_fee") or 0
    return {
        "rank": rank,
        "card_name": name,
        "card_company": company,
        "net_saving": card.get("net_saving", 0),
        "annual_fee": annual_fee,
        "summary": "추천 설명 생성 실패",
        "benefit_details": benefit_details,
        "total_summary": (
            f"연간 총 할인액은 {total_discount:,}원이며, "
            f"연회비 {annual_fee:,}원을 차감한 순절감액은 "
            f"{card.get('net_saving', 0):,}원입니다."
        ),
        "fee_recovery": "",
        "cautions": "문장 생성에 실패해 시뮬레이션 결과를 그대로 표시했습니다.",
    }


def generate_card_worker(
    api_key: str,
    card: dict,
    structured_map: dict,
    transactions: list[dict],
) -> dict:
    card = sanitize_card(card)
    rank = card.get("rank", "?")
    name = card["card_name"]
    company = card["card_company"]
    merged_breakdown = merge_benefit_breakdown(card.get("benefit_breakdown", []))
    benefit_structure_str = get_benefit_structure(name, structured_map)
    breakdown_to_describe = [b for b in merged_breakdown if b.get("total_discount", 0) > 0]
    benefit_usage_str = build_benefit_usage_summary(
        name,
        breakdown_to_describe,
        structured_map,
        transactions,
    )

    client = OpenAI(api_key=api_key)
    result, reason = generate_for_card(
        client,
        card,
        benefit_structure_str,
        merged_breakdown,
        benefit_usage_str,
    )

    if result is None:
        result = make_failed_recommendation(card, rank, name, company)

    return {
        "rank": rank,
        "card_name": name,
        "card_company": company,
        "benefit_count": len(merged_breakdown),
        "reason": reason,
        "result": result,
    }


def main():
    if not DEBUG_MODE:
        print("디버그 모드 비활성화 — app.py 메모리 파이프라인으로 실행하세요.")
        print("독립 실행하려면 DEBUG_MODE = True 로 변경하세요.")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

    input_json = DEBUG_OUTPUT_DIR / "top5_cards.json"
    with open(input_json, encoding="utf-8") as f:
        top5_data: dict = json.load(f)
    with open(FILTERED_JSON, encoding="utf-8") as f:
        filtered: dict = json.load(f)
    with open(STRUCTURED_JSON, encoding="utf-8") as f:
        structured_list: list = json.load(f)

    structured_map = {c["card_name"]: c for c in structured_list}
    cards: list[dict]  = top5_data["top5"]
    all_tx: list[dict] = filtered.get("transactions", [])

    print(f"\n{'='*62}")
    print(f"  카드 추천 설명 생성 시작 — {len(cards)}개 카드")
    print(f"  병렬 처리 workers: {min(MAX_WORKERS, len(cards))}개")
    print(f"  추천문 근거: 카드별 혜택 적용 거래 전체 집계")
    print(f"{'='*62}")

    recommendations_by_rank: dict[int, dict] = {}
    max_workers = min(MAX_WORKERS, len(cards)) or 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(generate_card_worker, api_key, card, structured_map, all_tx)
            for card in cards
        ]

        for done_count, future in enumerate(as_completed(futures), 1):
            item = future.result()
            rank = item["rank"]
            result = item["result"]
            recommendations_by_rank[int(rank)] = result
            status = "OK" if item["reason"] == "OK" else f"실패 ({item['reason']})"
            print(
                f"  [완료 {done_count}/{len(cards)}] "
                f"[{rank}] {item['card_company']} / {item['card_name']} "
                f"(benefit {item['benefit_count']}개) {status}",
                flush=True,
            )

    recommendations = [
        recommendations_by_rank[rank]
        for rank in sorted(recommendations_by_rank)
    ]

    # DEBUG_MODE일 때만 파일 저장
    output_path = DEBUG_OUTPUT_DIR / "recommendation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"generated_at": str(__import__("datetime").date.today()),
                   "recommendations": recommendations},
                  f, ensure_ascii=False, indent=2)

    print(f"\n{'='*62}")
    print(f"  생성 완료  |  저장 위치: {output_path}")
    print(f"  {'─'*60}")
    for r in recommendations:
        print(f"\n  [{r.get('rank')}] {r.get('card_company')} / {r.get('card_name')}")
        print(f"      순절감: {r.get('net_saving',0):,}원")
        # summary 첫 문장 미리보기
        s = r.get("summary", "")
        preview = s.split(".")[0] + "." if s else "(없음)"
        print(f"      요약: {preview}")
        details = r.get("benefit_details", [])
        detail_total = sum(b.get("estimated_saving", 0) for b in details)
        print(f"      혜택 {len(details)}개  합계: {detail_total:,}원")
        for b in details:
            print(f"      - {b.get('benefit_name')}: {b.get('estimated_saving',0):,}원")
            print(f"        {b.get('matched_merchants','')[:80]}")
    print(f"\n{'='*62}\n")


if __name__ == "__main__":
    main()
