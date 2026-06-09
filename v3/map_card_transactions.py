"""
필터링된 거래 데이터 + card_benefits_structured.json
→ 카드별 거래 매핑 결과 생성

카드별로 "이 카드를 썼다면 혜택 받을 수 있었던 거래"를 분리한다.
"""

import json
import os
from collections import defaultdict
from datetime import date
from pathlib import Path

# ── 경로 설정 ───────────────────────────────────────────────
BASE_DIR         = Path(__file__).resolve().parent
OUTPUT_DIR       = BASE_DIR / "output"        # 고정 데이터 (카드 DB)
DEBUG_OUTPUT_DIR = BASE_DIR / "debug_output"  # 디버그 모드 산출물
STRUCTURED_JSON  = OUTPUT_DIR / "card_benefits_structured.json"
FILTERED_JSON    = DEBUG_OUTPUT_DIR / "filtered_transactions.json"

# ── 디버그 모드 ─────────────────────────────────────────────
# True : 독립 실행 가능. filtered_transactions.json 읽고 card_transactions_map.json 저장
# False: app.py 메모리 파이프라인용 (파일 입출력 없음)
DEBUG_MODE = True

# ── 표기 정규화 테이블 ──────────────────────────────────────
MATCH_ALIASES: dict[str, str] = {
    "MGC메가커피":   "메가커피",
    "메가MGC커피":   "메가커피",
    "투썸플레이트":  "투썸플레이스",
    "파리바게트":    "파리바게뜨",
    "파스쿠치":      "파스쿠찌",
    "디즈니 플러스": "디즈니+",
    "디즈니플러스":  "디즈니+",
    "SSG COM":      "SSG.COM",
    "할리스커피":    "할리스",
    "삼성 페이":     "삼성페이",
    "SSGPAY":       "SSG PAY",
    "SSG페이":      "SSG PAY",
    "L.pay":        "L.PAY",
    "L페이":         "L.PAY",
}


def normalize(merchant: str) -> str:
    """MATCH_ALIASES 기준으로 표기 정규화"""
    return MATCH_ALIASES.get(merchant, merchant)


def main():
    if not DEBUG_MODE:
        print("디버그 모드 비활성화 — app.py 메모리 파이프라인으로 실행하세요.")
        print("독립 실행하려면 DEBUG_MODE = True 로 변경하세요.")
        return

    # ── 데이터 로드 ──────────────────────────────────────────
    with open(FILTERED_JSON, encoding="utf-8") as f:
        ft_data: dict = json.load(f)
    with open(STRUCTURED_JSON, encoding="utf-8") as f:
        cards: list[dict] = json.load(f)

    transactions: list[dict] = ft_data["transactions"]
    total_tx = len(transactions)
    total_cards = len(cards)

    print(f"\n{'='*60}")
    print(f"  카드-거래 매핑 시작")
    print(f"  입력 거래 수: {total_tx}건  |  카드 수: {total_cards}개")
    print(f"{'='*60}\n")

    # ── 정규화 전 기준 매칭 수 (MATCH_ALIASES 효과 측정용) ────
    # 정규화 없이 단순 set 비교로 먼저 카운트
    pre_alias_matched: int = 0

    # ── 카드별 인덱스 준비 ────────────────────────────────────
    # {card_name: {"card_company", "has_all", "eligible_set(normalized)"}}
    card_index: dict[str, dict] = {}
    for card in cards:
        name     = card["card_name"]
        company  = card["card_company"]
        # 해외 전용 ALL 혜택은 국내 거래에 미적용 → 제외
        OVERSEAS_KEYWORDS = {"해외", "해외이용", "해외가맹점", "해외이용금액"}
        has_all = any(
            b["merchant_type"] == "ALL"
            and not any(kw in b.get("category", "") for kw in OVERSEAS_KEYWORDS)
            for b in card["benefits"]
        )
        # eligible_merchants를 정규화한 집합
        eligible_norm = {normalize(m) for m in card["eligible_merchants"]}
        # 정규화 전 집합 (MATCH_ALIASES 효과 측정용)
        eligible_raw  = set(card["eligible_merchants"])

        card_index[name] = {
            "card_company":   company,
            "has_all":        has_all,
            "eligible_norm":  eligible_norm,
            "eligible_raw":   eligible_raw,
        }

    # ── 매칭 수행 ─────────────────────────────────────────────
    # card_name → list[transaction]
    card_tx_map: dict[str, list[dict]] = defaultdict(list)

    alias_extra_count = 0   # 정규화로 추가 매칭된 거래 수 (카드×거래 단위)

    for tx in transactions:
        matched_m  = tx.get("matched_merchant") or ""
        matched_n  = normalize(matched_m)          # 정규화된 거래 가맹점

        for card_name, ci in card_index.items():
            # Case 2: ALL 타입 카드 → 모든 거래 포함
            if ci["has_all"]:
                card_tx_map[card_name].append(tx)
                continue

            # Case 1: eligible_merchants에 포함되는지 확인 (정규화 후 비교)
            if matched_n in ci["eligible_norm"]:
                card_tx_map[card_name].append(tx)

                # 정규화 전에는 매칭 안 됐던 건 카운트
                if matched_m not in ci["eligible_raw"]:
                    alias_extra_count += 1

    # Case 3: 거래가 하나도 없는 카드도 빈 배열로 포함
    for card_name in card_index:
        if card_name not in card_tx_map:
            card_tx_map[card_name] = []

    # ── 결과 조립 ─────────────────────────────────────────────
    cards_output: dict[str, dict] = {}
    for card in cards:           # structured.json 순서 유지
        name    = card["card_name"]
        ci      = card_index[name]
        tx_list = card_tx_map[name]
        cards_output[name] = {
            "card_company":         ci["card_company"],
            "has_all_type_benefit": ci["has_all"],
            "transaction_count":    len(tx_list),
            "transactions":         tx_list,
        }

    # ── 검증 출력 ─────────────────────────────────────────────

    # 거래 수 기준 정렬
    sorted_cards = sorted(
        cards_output.items(),
        key=lambda x: x[1]["transaction_count"],
        reverse=True,
    )

    # 1. 기본 현황
    # DEBUG_MODE일 때만 파일 저장
    output_path = DEBUG_OUTPUT_DIR / "card_transactions_map.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"generated_at": str(__import__("datetime").date.today()),
                   "total_cards": total_cards, "cards": cards_output},
                  f, ensure_ascii=False, indent=2)

    print(f"  [1] 기본 현황")
    print(f"      입력 거래 수  : {total_tx}건")
    print(f"      처리된 카드 수: {total_cards}개")
    print(f"      저장 위치     : {output_path}")

    # 2. 거래 수 Top 10 / Bottom 5
    print(f"\n  [2] 카드별 거래 수 분포")
    print(f"  {'─'*56}")
    print(f"  {'카드명':<35} {'회사':<10} {'거래':>5}  ALL")
    print(f"  {'─'*56}")
    print(f"  ▲ Top 10")
    for name, info in sorted_cards[:10]:
        all_mark = "✔" if info["has_all_type_benefit"] else " "
        print(f"    {name[:33]:<33} {info['card_company'][:9]:<9} "
              f"{info['transaction_count']:>5}건  {all_mark}")

    print(f"\n  ▼ Bottom 5")
    for name, info in sorted_cards[-5:]:
        all_mark = "✔" if info["has_all_type_benefit"] else " "
        print(f"    {name[:33]:<33} {info['card_company'][:9]:<9} "
              f"{info['transaction_count']:>5}건  {all_mark}")

    # 거래 0건 카드
    zero_cards = [(n, i) for n, i in sorted_cards if i["transaction_count"] == 0]
    if zero_cards:
        print(f"\n  ⚠  거래 0건 카드 ({len(zero_cards)}개)")
        for name, info in zero_cards:
            print(f"    - {info['card_company']} / {name}")
    else:
        print(f"\n  거래 0건 카드 없음")

    # 3. ALL 타입 카드 목록
    all_type_cards = [(n, i) for n, i in sorted_cards if i["has_all_type_benefit"]]
    print(f"\n  [3] ALL 타입 카드 ({len(all_type_cards)}개) — 전체 거래 포함")
    print(f"  {'─'*56}")
    for name, info in sorted(all_type_cards, key=lambda x: x[0]):
        print(f"    {info['card_company']:<10} {name}")

    # 4. MATCH_ALIASES 효과
    print(f"\n  [4] MATCH_ALIASES 정규화 효과")
    print(f"      추가 매칭 건수 (카드×거래 단위): {alias_extra_count}건")
    if alias_extra_count == 0:
        print(f"      (이미 표기가 통일돼 있거나 해당 가맹점 거래 없음)")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
