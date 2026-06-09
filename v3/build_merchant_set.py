"""
card_benefits_structured.json → output/global_merchant_set.json
모든 카드의 eligible_merchants를 수집·정규화·중복제거하여 글로벌 가맹점 집합 생성
"""

import json
import os
import re
from collections import Counter

# ── 경로 설정 ───────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON  = os.path.join(BASE_DIR, "output", "card_benefits_structured.json")
OUTPUT_JSON = os.path.join(BASE_DIR, "output", "global_merchant_set.json")

# ── 법인 표기 제거 패턴 ─────────────────────────────────────
# 앞에 붙는 형태: "주식회사 아성다이소", "(주)이마트", "㈜이마트", "유한회사 ..."
_CORP_PREFIX = re.compile(
    r'^(?:주식회사\s*|유한회사\s*|합자회사\s*|합명회사\s*|\(주\)\s*|㈜\s*|\(유\)\s*|㈔\s*)'
)
# 뒤에 붙는 형태: "한국맥도날드(유)", "CJ올리브영(주)"
_CORP_SUFFIX = re.compile(
    r'(?:\s*\(주\)|\s*\(유\)|\s*㈜|\s*㈔|\s*주식회사|\s*유한회사)$'
)
# "한국" 접두어 제거 (예: "한국맥도날드" → "맥도날드")
_KOREA_PREFIX = re.compile(r'^한국(?=[가-힣A-Za-z])')


# ── 동의어 통일 테이블 (정규화 후 적용) ────────────────────
# 키: 제거할 표기, 값: 대표 표기
SYNONYM_MAP: dict[str, str] = {
    # 법인명 → 브랜드명 (법인 제거 후에도 남는 경우)
    "아성다이소": "다이소",
    "코리아세븐": "세븐일레븐",
    "비지에프리테일": "CU",
    "지에스리테일": "GS25",
    # 편의점
    "씨유": "CU",
    "지에스25": "GS25",
    "세븐-일레븐": "세븐일레븐",
    "7-eleven": "세븐일레븐",
    "7eleven": "세븐일레븐",
    "이마트 24": "이마트24",
    # 배달앱
    "배민": "배달의민족",
    "baemin": "배달의민족",
    # 커피
    "starbucks": "스타벅스",
    "twosome": "투썸플레이스",
    "투썸": "투썸플레이스",
    "이디야커피": "이디야",
    "ediya": "이디야",
    "mega coffee": "메가커피",
    "compose coffee": "컴포즈커피",
    "컴포즈": "컴포즈커피",
    # 마트
    "이마트": "이마트",          # 정규화 후 중복 방지용
    "홈플러스": "홈플러스",
    "롯데마트": "롯데마트",
    # 온라인쇼핑
    "네이버플러스 스토어": "네이버쇼핑",
    "네이버 쇼핑": "네이버쇼핑",
    "naver": "네이버쇼핑",
    "gmarket": "G마켓",
    "auction": "옥션",
    "ssg.com": "SSG.COM",
    "ssg": "SSG.COM",
    "coupay": "쿠페이",
    # 통신
    "sk텔레콤": "SKT",
    "sk telecom": "SKT",
    "kt": "KT",
    "lg u+": "LG U+",
    "lgu+": "LG U+",
    # 주유
    "sk에너지": "SK에너지",
    "gs칼텍스": "GS칼텍스",
    "s-oil": "S-OIL",
    "s oil": "S-OIL",
    "현대오일뱅크": "HD현대오일뱅크",
    # 기타
    "올리브영": "CJ올리브영",
    "cgv": "CGV",
    "megabox": "메가박스",
    "lotte cinema": "롯데시네마",
}


def normalize(raw: str) -> str:
    """가맹점명 정규화"""
    m = raw.strip()
    if not m:
        return ""

    # 법인 표기 제거
    m = _CORP_PREFIX.sub("", m).strip()
    m = _CORP_SUFFIX.sub("", m).strip()

    # "한국맥도날드" → "맥도날드"
    m = _KOREA_PREFIX.sub("", m).strip()

    # 동의어 통일 (소문자 비교)
    lower = m.lower()
    if lower in SYNONYM_MAP:
        m = SYNONYM_MAP[lower]

    return m.strip()


def mixed_sort_key(s: str) -> tuple[int, str]:
    """영문 먼저, 한글 나중 정렬 키"""
    first = s[0] if s else ""
    if first.isascii():
        return (0, s.upper())   # 영문/숫자/특수
    return (1, s)               # 한글


def category_sample(merchants: list[str], keywords: list[str]) -> list[str]:
    """키워드 중 하나라도 포함되는 가맹점 필터링"""
    result = []
    for m in merchants:
        if any(kw in m for kw in keywords):
            result.append(m)
    return result


def main():
    with open(INPUT_JSON, encoding="utf-8") as f:
        cards: list[dict] = json.load(f)

    # ── 1. eligible_merchants 수집 (중복 포함) ────────────────
    raw_all: list[str] = []
    # 카드별 가맹점 집합 (top-N 계산용)
    card_merchant_sets: list[set[str]] = []

    for card in cards:
        ms = card.get("eligible_merchants", [])
        raw_all.extend(ms)
        card_merchant_sets.append(set(ms))

    raw_count = len(raw_all)

    # ── 2. 정규화 ────────────────────────────────────────────
    normalized: list[str] = []
    for m in raw_all:
        n = normalize(m)
        if n:
            normalized.append(n)

    unique_merchants = sorted(set(normalized), key=mixed_sort_key)
    unique_count = len(unique_merchants)

    # ── 3. 카드별 등장 횟수 집계 ─────────────────────────────
    # 정규화된 이름 기준으로 카드별 포함 여부 카운트
    norm_card_sets: list[set[str]] = [
        {normalize(m) for m in s} for s in card_merchant_sets
    ]
    appearance: Counter = Counter()
    for merchant in unique_merchants:
        cnt = sum(1 for s in norm_card_sets if merchant in s)
        appearance[merchant] = cnt

    # ── 4. 저장 ──────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    output = {
        "total_count": unique_count,
        "merchants": unique_merchants,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ── 5. 검증 출력 ─────────────────────────────────────────
    print(f"\n{'='*58}")
    print(f"  글로벌 가맹점 집합 생성 완료")
    print(f"{'='*58}")
    print(f"  정규화 전 전체 (중복 포함) : {raw_count:>5}개")
    print(f"  정규화 후 유니크           : {unique_count:>5}개")
    print(f"  저장 위치: {OUTPUT_JSON}")

    CATEGORY_FILTERS = [
        ("카페 계열",      ["스타벅스", "투썸", "이디야", "메가", "컴포즈", "할리스", "커피빈"]),
        ("편의점 계열",    ["CU", "GS25", "세븐일레븐", "이마트24"]),
        ("배달앱 계열",    ["배달", "쿠팡이츠", "요기요"]),
        ("온라인쇼핑 계열",["쿠팡", "네이버", "G마켓", "옥션", "11번가"]),
        ("주유소 계열",    ["SK에너지", "GS칼텍스", "S-OIL", "현대오일"]),
    ]

    print(f"\n  {'─'*54}")
    print(f"  카테고리별 샘플 (육안 확인용)")
    print(f"  {'─'*54}")
    for label, kws in CATEGORY_FILTERS:
        hits = category_sample(unique_merchants, kws)
        print(f"  [{label}] ({len(hits)}개)")
        if hits:
            print(f"    {', '.join(hits)}")
        else:
            print(f"    (없음)")

    print(f"\n  {'─'*54}")
    print(f"  가맹점 등장 카드 수 Top 10")
    print(f"  {'─'*54}")
    for rank, (merchant, cnt) in enumerate(appearance.most_common(10), 1):
        bar = "■" * min(cnt, 30)
        print(f"  {rank:>2}. {merchant:<20} {cnt:>3}개 카드  {bar}")

    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
