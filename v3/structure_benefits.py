"""
card_benefits_cleaned.json → OpenAI API → card_benefits_structured.json
100개 카드 전체를 gpt-4o로 구조화한다.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

# ── 경로 설정 ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON  = os.path.join(BASE_DIR, "output", "card_benefits_cleaned.json")
OUTPUT_JSON = os.path.join(BASE_DIR, "output", "card_benefits_structured.json")

MODEL       = "gpt-4o"
MAX_RETRIES = 2
SLEEP_SEC   = 1.0   # 카드 간 sleep
MAX_WORKERS = int(os.environ.get("STR_MAX_WORKERS", "4"))

SYSTEM_PROMPT = """\
You are a Korean credit card benefit parser.
Your job is to extract structured benefit information from Korean credit card benefit text.
You must respond ONLY with a valid JSON object. No explanation, no markdown, no code blocks."""

USER_PROMPT_TEMPLATE = """\
아래는 신용카드 혜택 텍스트이다. 이를 분석해서 다음 JSON 구조로 변환해라.

[카드 정보]
카드명: {card_name}
카드사: {card_company}
국내 연회비: {annual_fee_domestic}
해외 연회비: {annual_fee_overseas}
전월 실적 기준: {previous_month_requirement}

[혜택 텍스트]
{cleaned_benefits}

[출력 JSON 구조]
{{
  "card_name": "카드명",
  "card_company": "카드사명",
  "annual_fee": 20000,
  "previous_month_requirement": 400000,
  "benefits": [
    {{
      "benefit_id": "b1",
      "select_group": null,
      "select_option": null,
      "category": "편의점",
      "merchant_type": "B",
      "merchants": ["CU", "GS25", "세븐일레븐", "이마트24"],
      "discount_type": "rate",
      "discount_value": 7,
      "tiered_discount": null,
      "conditions": {{
        "per_transaction_min": null,
        "monthly_limit": 10000,
        "tiered_monthly_limit": [
          {{"min_spend": 400000, "limit": 7000}},
          {{"min_spend": 800000, "limit": 10000}}
        ],
        "offline_only": false,
        "notes": "백화점 임대매장 제외"
      }}
    }}
  ],
  "eligible_merchants": ["CU", "GS25", "스타벅스"]
}}

[merchant_type 분류 규칙]
- A: 특정 가맹점이 명시된 경우
  예: "스타벅스, 투썸플레이스 10% 할인"
  → merchants에 명시된 가맹점명 그대로

- B: 카테고리명과 함께 가맹점 목록이 명시된 경우
  예: "편의점 CU, GS25, 세븐일레븐, 이마트24 7% 할인"
  → merchants에 나열된 가맹점명 그대로

- C: 카테고리명만 있고 가맹점 목록이 없는 경우
  예: "카페 5% 할인", "주유소 5% 할인"
  → merchants에 해당 카테고리의 한국 주요 가맹점을 직접 생성해서 넣어라
  아래 예시를 참고:
    카페/커피전문점 → ["스타벅스", "투썸플레이스", "이디야", "메가커피",
                      "컴포즈커피", "할리스", "커피빈", "폴바셋", "바나프레소", "빽다방"]
    편의점 → ["CU", "GS25", "세븐일레븐", "이마트24", "미니스톱"]
    주유소 → ["SK에너지", "GS칼텍스", "S-OIL", "HD현대오일뱅크"]
    패스트푸드 → ["맥도날드", "버거킹", "롯데리아", "KFC", "맘스터치", "노브랜드버거"]
    일반음식점/한식 → ["본죽", "한솥", "김밥천국", "이삭토스트", "원할머니보쌈"]
    뷔페/패밀리레스토랑 → ["애슐리", "빕스", "아웃백", "더플레이스", "올반"]
    마트/할인점 → ["이마트", "홈플러스", "롯데마트", "코스트코", "농협하나로마트"]
    온라인쇼핑 → ["쿠팡", "네이버쇼핑", "G마켓", "옥션", "11번가", "SSG.COM", "컬리"]
    배달앱 → ["배달의민족", "쿠팡이츠", "요기요"]
    대중교통 → ["버스", "지하철", "택시"]
    통신 → ["SKT", "KT", "LG U+"]
    영화관 → ["CGV", "롯데시네마", "메가박스"]
    약국 → ["올리브영", "CJ올리브영"]

  [주의] "한식", "양식", "일식", "중식", "뷔페", "패밀리레스토랑", "패스트푸드"는
  가맹점명이 아니라 서브카테고리다. merchants에 절대 넣지 마라.
  이런 단어들이 보이면 Type C로 처리하고 위 예시를 참고해 실제 가맹점을 생성해라.

- ALL: 특정 가맹점 제한 없이 모든 가맹점에 적용되는 혜택
  예: "국내 모든 가맹점 0.7% 할인", "국내외 가맹점 포인트 적립"
  → merchants: [] 빈 배열로 두어라
  → 시뮬레이션에서 사용자의 모든 거래에 적용됨

- SERVICE: 특정 가맹점에서의 결제가 아닌 서비스/멤버십 형태의 혜택
  예: 항공 마일리지 적립, 공항 라운지 이용, 호텔 멤버십,
      여행자 보험, 골프 패키지, 컨시어지 서비스, 정부 바우처
  → merchants: [] 빈 배열로 두어라
  → 시뮬레이션에서 스킵됨 (금액 계산 불가)

[판단 기준]
- "모든 가맹점", "국내 가맹점", "국내외 가맹점" 같은 표현 → ALL
- 마일리지, 라운지, 보험, 멤버십, 바우처, 컨시어지 → SERVICE
- 특정 카테고리나 가맹점이 명시 → A, B, C 중 선택
- "등" 포함 시: 명시된 가맹점 + 동일 카테고리 유사 가맹점 추가 후 Type A 또는 B

[tiered_discount 형식] (구간별 할인율이 다를 때)
[
  {{"min_amount": 0, "max_amount": 29999, "rate": 3}},
  {{"min_amount": 30000, "max_amount": 49999, "rate": 5}},
  {{"min_amount": 50000, "max_amount": null, "rate": 7}}
]

[주의] 주중/주말 차등 할인은 tiered_discount가 아니라 conditions.notes에 기록해라.
tiered_discount는 결제 금액 구간 기반일 때만 사용한다.
예: "주중 5%, 주말 10%" →
  discount_value: 5,
  tiered_discount: null,
  conditions.notes: "주말(금~일요일) 10% 적용"

[주의사항]
- SELECT형 카드는 각 옵션을 별도 benefit 객체로 만들고 select_group과 select_option을 반드시 채워라
- 같은 카테고리라도 SELECT 옵션이 다르면 별도 benefit으로 분리해라
- eligible_merchants는 merchant_type이 A, B, C인 benefit의 merchants만 합산한 중복 없는 배열이다.
  ALL과 SERVICE는 eligible_merchants에 포함하지 않는다.
  (ALL은 시뮬레이션에서 별도 처리, SERVICE는 시뮬레이션 스킵)
- annual_fee는 국내 연회비 기준으로 숫자만 추출. "20,000원" → 20000
- select_option은 반드시 해당 옵션의 설명 문자열로 채워라. 숫자(1, 2, 3) 절대 금지.
  예: "select_option": "음식점/편의점/할인점/주유"
  예: "select_option": "온라인쇼핑몰/의료/배달앱"
- 주유 혜택에서 "60원/L", "리터당 60원", "ℓ당 60원"처럼 리터당 원 단위가 나오면
  절대 60%로 구조화하지 마라. 이 프로젝트의 거래내역에는 리터 수가 없으므로
  discount_type은 null, discount_value는 null로 두고 conditions.notes에 원문 조건을 기록해라.
  "주유소 5% 할인"처럼 % 기호가 명시된 경우에만 discount_type rate를 사용해라.
- "6개 영역 중 이용금액 1위/2위 영역", "가장 많이 쓴 영역", "상위 영역"처럼
  사용자 월별 소비 영역 순위를 먼저 판단해야 하는 혜택은 ALL로 두지 마라.
  적용 영역을 카드사 기준으로 정확히 산정할 수 없으면 SERVICE로 두고 notes에 원문 조건을 기록해라.
- 반드시 JSON만 출력. 다른 텍스트 절대 금지"""


# ── 유효성 검사 ─────────────────────────────────────────────
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


def normalize_structured_benefits(data: dict) -> dict:
    for benefit in data.get("benefits", []):
        conditions = benefit.setdefault("conditions", {})
        text = benefit_text(benefit)
        category = str(benefit.get("category") or "")
        value = benefit.get("discount_value") or 0
        notes = str(conditions.get("notes") or "")

        if (
            benefit.get("discount_type") == "rate"
            and value >= 20
            and any(keyword in category or keyword in notes for keyword in FUEL_CATEGORIES)
            and ("%" not in notes and "％" not in notes and "퍼센트" not in notes)
        ):
            benefit["discount_type"] = None
            benefit["discount_value"] = None
            conditions["notes"] = (notes + " / 리터당 원 단위 주유 혜택은 거래내역만으로 절감액 산정 불가").strip()

        if benefit.get("merchant_type") == "ALL" and any(
            keyword in text for keyword in AREA_RANK_KEYWORDS
        ):
            benefit["merchant_type"] = "SERVICE"
            benefit["merchants"] = []
            conditions["notes"] = (notes + " / 소비 영역 순위형 혜택은 카드사 영역 산정 없이는 계산 불가").strip()

    eligible = []
    for benefit in data.get("benefits", []):
        if benefit.get("merchant_type") in {"A", "B", "C"}:
            eligible.extend(benefit.get("merchants", []))
    data["eligible_merchants"] = list(dict.fromkeys(eligible))
    return data


def validate(data: dict) -> tuple[bool, str]:
    if not isinstance(data.get("benefits"), list) or len(data["benefits"]) == 0:
        return False, "benefits 배열 없음 또는 비어있음"
    if not isinstance(data.get("eligible_merchants"), list):
        return False, "eligible_merchants 없음"
    for i, b in enumerate(data["benefits"]):
        # merchants 키가 아예 없거나 list가 아닌 경우만 실패
        # 빈 배열은 허용 ("전 가맹점", "모든 가맹점" 등 광범위 혜택)
        if not isinstance(b.get("merchants"), list):
            return False, f"benefit[{i}].merchants 키 없음 또는 list가 아님"
    return True, "OK"


# ── JSON 추출 (마크다운 코드블록 안전 처리) ──────────────────
def extract_json(text: str) -> str:
    text = text.strip()
    # ```json ... ``` 또는 ``` ... ``` 블록 제거
    m = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if m:
        return m.group(1).strip()
    # 첫 { 부터 마지막 } 까지만 추출
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1:
        return text[start:end + 1]
    return text


# ── 카드 1장 처리 ────────────────────────────────────────────
def process_card(client: OpenAI, card: dict) -> tuple[dict | None, str]:
    user_msg = USER_PROMPT_TEMPLATE.format(
        card_name=card["card_name"],
        card_company=card["card_company"],
        annual_fee_domestic=card["annual_fee_domestic"],
        annual_fee_overseas=card["annual_fee_overseas"],
        previous_month_requirement=card["previous_month_requirement"],
        cleaned_benefits=card["cleaned_benefits"],
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
            raw = resp.choices[0].message.content or ""
            parsed = json.loads(extract_json(raw))
            parsed = normalize_structured_benefits(parsed)
            ok, reason = validate(parsed)
            if ok:
                return parsed, "OK"
            last_error = f"유효성 실패: {reason}"
        except json.JSONDecodeError as e:
            last_error = f"JSON 파싱 오류: {e}"
        except Exception as e:
            last_error = f"API 오류: {e}"

        if attempt < MAX_RETRIES:
            print(f"    재시도 {attempt + 1}/{MAX_RETRIES} ({last_error})")
            time.sleep(SLEEP_SEC)

    return None, last_error


def process_card_worker(idx: int, total: int, api_key: str, card: dict) -> dict:
    client = OpenAI(api_key=api_key)
    structured, reason = process_card(client, card)
    return {
        "idx": idx,
        "total": total,
        "card": card,
        "structured": structured,
        "reason": reason,
    }


def collect_result_stats(
    structured: dict,
    all_type_counts: dict[str, int],
    type_c_cards: list[dict],
) -> str:
    benefits = structured.get("benefits", [])
    type_dist: dict[str, int] = {}
    for b in benefits:
        mt = b.get("merchant_type", "?")
        type_dist[mt] = type_dist.get(mt, 0) + 1
        if mt in all_type_counts:
            all_type_counts[mt] += 1

    c_cats = [
        b["category"]
        for b in benefits
        if b.get("merchant_type") == "C"
    ]
    if c_cats:
        type_c_cards.append({
            "card_name": structured.get("card_name", ""),
            "categories": list(dict.fromkeys(c_cats)),
        })

    dist_str = " ".join(f"{k}:{v}" for k, v in sorted(type_dist.items()))
    return f"benefits: {len(benefits)}개  [{dist_str}]"


# ── 메인 ─────────────────────────────────────────────────────
def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

    with open(INPUT_JSON, encoding="utf-8") as f:
        cards = json.load(f)

    results_by_idx: dict[int, dict] = {}
    failed_cards: list[dict] = []
    type_c_cards: list[dict] = []   # {"card_name": ..., "categories": [...]}
    all_type_counts: dict[str, int] = {"A": 0, "B": 0, "C": 0, "ALL": 0, "SERVICE": 0}

    total = len(cards)
    print(f"\n{'='*60}")
    print(f"  구조화 시작 — {total}개 카드, 모델: {MODEL}")
    print(f"  병렬 처리 workers: {MAX_WORKERS}개")
    print(f"{'='*60}")

    futures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for idx, card in enumerate(cards, 1):
            futures.append(executor.submit(process_card_worker, idx, total, api_key, card))

        for done_count, future in enumerate(as_completed(futures), 1):
            item = future.result()
            idx = item["idx"]
            card = item["card"]
            name = card["card_name"]
            company = card["card_company"]
            structured = item["structured"]
            reason = item["reason"]

            if structured is None:
                print(
                    f"[완료 {done_count:>3}/{total}] "
                    f"[카드 {idx:>3}/{total}] {company} / {name[:35]} ... 실패 ({reason})",
                    flush=True,
                )
                failed_cards.append({
                    "card_name": name,
                    "card_company": company,
                    "reason": reason,
                })
            else:
                stats = collect_result_stats(structured, all_type_counts, type_c_cards)
                results_by_idx[idx] = structured
                print(
                    f"[완료 {done_count:>3}/{total}] "
                    f"[카드 {idx:>3}/{total}] {company} / {name[:35]} ... OK  ({stats})",
                    flush=True,
                )

    results = [results_by_idx[idx] for idx in sorted(results_by_idx)]

    # ── 저장 ────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ── 검증 출력 ────────────────────────────────────────────
    all_merchants: set[str] = set()
    for r in results:
        all_merchants.update(r.get("eligible_merchants", []))

    print(f"\n{'='*60}")
    print(f"  완료: {len(results)}/{total} 카드 성공")
    print(f"  저장: {OUTPUT_JSON}")

    if failed_cards:
        print(f"\n  [실패 카드 — {len(failed_cards)}건]")
        for fc in failed_cards:
            print(f"    ✗  {fc['card_company']} / {fc['card_name']}")
            print(f"       사유: {fc['reason']}")
    else:
        print("\n  [실패 카드 없음]")

    if type_c_cards:
        print(f"\n  [Type C 혜택 카드 — LLM 세상지식으로 가맹점 생성, 검토 권장]")
        for tc in type_c_cards:
            cats = ", ".join(tc["categories"])
            print(f"    ⚑  {tc['card_name']:<40}  카테고리: {cats}")
    else:
        print("\n  [Type C 혜택 없음]")

    print(f"\n  eligible_merchants 글로벌 합집합: {len(all_merchants)}개")
    print(f"\n  [merchant_type 전체 분포]")
    for mt, cnt in sorted(all_type_counts.items()):
        print(f"    {mt:8s}: {cnt:4d}건")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
