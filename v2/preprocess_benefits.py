"""
card_corp_top10.csv의 raw_benefits를 전처리하여
output/card_benefits_cleaned.json 생성
"""

import csv
import json
import os
import re

# ── 경로 설정 ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "크롤링", "card_corp_top10.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "card_benefits_cleaned.json")

# ── 제거할 섹션 태그 ────────────────────────────────────────
REMOVE_TAGS = {
    "[유의사항]",
    "[연회비]",
    "[프리미엄 서비스]",
    "[프리미엄]",   # [프리미엄 서비스]의 축약형도 함께 제거
    "[제휴/PLCC]",
    "[기타]",
    "[무이자할부]",  # 결제조건이므로 혜택 본문 아님
    "[수수료우대]",  # 수수료 우대는 혜택 내용 아님
}

# 혜택 섹션으로 판단하는 키워드 (% 또는 혜택 관련 단어)
BENEFIT_PATTERN = re.compile(
    r'%|할인|적립|캐시백|무료|마일리지|포인트|바우처|쿠폰|페이백|면제|제공|혜택'
)

# ── 제거할 문장 패턴 (정규식) ────────────────────────────────
REMOVE_LINE_PATTERNS = [
    # "발급월 +1개월까지는..." 으로 시작하는 문장
    re.compile(r'발급월\s*\+\s*1개월까지는[^\n]*'),
    # "결제 수단에 해당 카드 등록 후..." 문장
    re.compile(r'결제\s*수단에\s*해당\s*카드\s*등록\s*후[^\n]*'),
    # 채널 제한 단서 문장: "국내 온라인 가맹점에 한하며..." 형태
    # (단, "오프라인 결제건에 한함" 처럼 혜택 범위를 직접 규정하는 건 유지)
    re.compile(r'국내\s*온라인\s*가맹점에\s*한하며[^\n]*'),
    re.compile(r'온라인\s*가맹점에\s*한하며[^\n]*'),
]


def get_section_tag(section_text: str) -> str | None:
    """섹션 텍스트에서 [태그명] 추출"""
    m = re.match(r'\s*(\[[^\]]+\])', section_text.strip())
    return m.group(1) if m else None


def is_benefit_section(section_text: str) -> bool:
    """혜택 내용이 실제로 포함된 섹션인지 판단"""
    return bool(BENEFIT_PATTERN.search(section_text))


def clean_section_body(text: str) -> str:
    """섹션 본문 내 노이즈 문장 제거"""
    for pattern in REMOVE_LINE_PATTERNS:
        text = pattern.sub('', text)

    # 연속된 빈 줄 압축
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 줄 앞뒤 공백 정리 (각 줄)
    lines = [line.rstrip() for line in text.splitlines()]
    text = '\n'.join(lines)
    return text.strip()


def process_raw_benefits(raw: str) -> tuple[list[str], list[str]]:
    """
    raw_benefits → (원본_섹션_리스트, 정제된_섹션_리스트)
    """
    sections = raw.split('|||')
    original_sections = [s.strip() for s in sections if s.strip()]

    cleaned_sections = []
    for sec in original_sections:
        tag = get_section_tag(sec)

        # 명시적 제거 태그
        if tag and tag in REMOVE_TAGS:
            continue

        # 혜택 내용이 없는 섹션 제거
        if not is_benefit_section(sec):
            continue

        cleaned = clean_section_body(sec)
        if cleaned:
            cleaned_sections.append(cleaned)

    return original_sections, cleaned_sections


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_CSV, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    results = []
    zero_section_cards = []

    print(f"\n{'='*55}")
    print(f"  card_benefits 전처리 시작  (총 {len(rows)}개 카드)")
    print(f"{'='*55}")

    for row in rows:
        card_name = row['card_name'].strip()
        card_company = row['card_company'].strip()

        original_sections, cleaned_sections = process_raw_benefits(row['raw_benefits'])

        orig_count = len(original_sections)
        clean_count = len(cleaned_sections)

        print(
            f"  [{card_company}] {card_name[:30]:<30} "
            f"섹션 {orig_count:>2} → {clean_count:>2}"
        )

        if clean_count == 0:
            zero_section_cards.append(f"{card_company} / {card_name}")

        results.append({
            "card_name": card_name,
            "card_company": card_company,
            "annual_fee_domestic": row['annual_fee_domestic'].strip(),
            "annual_fee_overseas": row['annual_fee_overseas'].strip(),
            "previous_month_requirement": row['previous_month_requirement'].strip(),
            "cleaned_benefits": '\n\n'.join(cleaned_sections),
        })

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*55}")
    print(f"  처리 완료: {len(results)}개 카드")
    print(f"  저장 위치: {OUTPUT_JSON}")

    if zero_section_cards:
        print(f"\n  [경고] 정제 후 섹션이 0개인 카드 ({len(zero_section_cards)}건):")
        for name in zero_section_cards:
            print(f"    ⚠  {name}")
    else:
        print("\n  [OK] 섹션 0개인 카드 없음")

    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
