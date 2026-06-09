"""
카드 후보 파싱 모듈 (Stage 1)

raw_benefits → {카테고리: 할인율} 구조화.

조회 우선순위:
  1) outputs/card_parsed_benefits.json (전체 사전 파싱, 추천 시 즉시 조회)
  2) gpt/.parser_cache.json (런타임 LLM 결과 캐시)
  3) LLM API (사전 파싱·캐시 모두 없을 때만, 병렬)

오프라인 전체 파싱: scripts/preparse_benefits.py

LLM은 추출만 담당. 계산은 calculator.py가 결정론적으로 처리.
"""
from __future__ import annotations
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
from openai import APIConnectionError, AuthenticationError, OpenAI, RateLimitError

import openai_config
from openai_config import create_client

_RAG = Path(__file__).parent.parent / "rag"
if str(_RAG) not in sys.path:
    sys.path.insert(0, str(_RAG))

from spending_analyzer import CATEGORY_KEYWORD_MAP

# ── API 키 (.env 또는 openai_config.OPENAI_API_KEY) ───────────────
OPENAI_API_KEY = ""
if OPENAI_API_KEY:
    openai_config.OPENAI_API_KEY = OPENAI_API_KEY

# ── 저장소 경로 ───────────────────────────────────────────────────
_BASE = Path(__file__).parent.parent
PREPARSED_FILE = _BASE / "outputs" / "card_parsed_benefits.json"
CACHE_FILE = Path(__file__).parent / ".parser_cache.json"

_PREPARED_STORE: Dict[str, Dict] | None = None

# 표준 카테고리 (calculator에서 SpendingProfile.categories와 매칭하므로 동일해야 함)
STANDARD_CATEGORIES = list(CATEGORY_KEYWORD_MAP.keys())

# 병렬 호출 동시성 (OpenAI rate limit 고려)
_MAX_WORKERS = 8


PARSER_SYSTEM_PROMPT = f"""\
당신은 신용카드 혜택 텍스트를 구조화된 데이터로 변환하는 파서이다.

[입력]
한 장의 카드 raw_benefits 텍스트(파이프 | 로 구분된 혜택 문장들)에서
"카테고리 → 할인율(%)" 매핑을 추출한다.

[표준 카테고리 — 반드시 이 목록 안에서만 카테고리명 사용]
{", ".join(STANDARD_CATEGORIES)}

[추출 규칙]
1. "X% 할인", "X% 적립", "X%P 적립" 등 % 형태의 혜택만 추출. 정액 할인(예: 리터당 60원)은 제외.
2. 가맹점/브랜드명은 표준 카테고리로 매핑:
   - 스타벅스/투썸/이디야 등 → 카페
   - 쿠팡/11번가/G마켓 등 → 온라인쇼핑
   - 배달의민족/요기요 → 배달앱
   - GS25/CU/세븐일레븐 → 편의점
   - 이마트/홈플러스/롯데마트 → 마트/슈퍼
   - 넷플릭스/유튜브프리미엄/디즈니플러스 → OTT/구독
   - 지하철/버스/택시 → 대중교통
   - SKT/KT/LG U+ → 통신
3. 같은 카테고리에 여러 혜택이 있으면 가장 높은 할인율 채택.
4. 월 한도(예: "월 최대 1만원", "월 한도 10,000원") 추출. 없으면 null.
5. 표준 카테고리에 명확히 매핑되지 않는 혜택(예: 해외, 항공, 면세점)은 제외.

[출력] 반드시 JSON. 다른 텍스트 금지.
{{
  "category_discounts": {{"카테고리명": 10.0, ...}},
  "category_caps": {{"카테고리명": 10000, ...}}
}}\
"""


# ── 캐시 입출력 ───────────────────────────────────────────────────

def _load_cache() -> Dict[str, Dict]:
    if not CACHE_FILE.exists():
        return {}
    try:
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: Dict[str, Dict]) -> None:
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except OSError:
        pass  # 캐시 쓰기 실패해도 메인 흐름은 진행


def cache_key(card_name: str, raw_benefits: str) -> str:
    """카드명 + raw_benefits 해시 → 같은 입력이면 항상 같은 키."""
    h = hashlib.sha256(raw_benefits.encode("utf-8")).hexdigest()[:16]
    return f"{card_name}::{h}"


_cache_key = cache_key  # 내부 호환


def _load_preparsed_store() -> Dict[str, Dict]:
    """outputs/card_parsed_benefits.json → {cache_key: {category_discounts, ...}}."""
    global _PREPARED_STORE
    if _PREPARED_STORE is not None:
        return _PREPARED_STORE

    store: Dict[str, Dict] = {}
    if PREPARSED_FILE.exists():
        try:
            data = json.loads(PREPARSED_FILE.read_text(encoding="utf-8"))
            raw = data.get("entries", data) if isinstance(data, dict) else {}
            if isinstance(raw, dict):
                store = {k: v for k, v in raw.items() if isinstance(v, dict)}
        except (json.JSONDecodeError, OSError):
            pass

    _PREPARED_STORE = store
    return store


def reload_preparsed_store() -> None:
    """사전 파싱 파일 갱신 후 메모리 캐시 초기화 (배치 스크립트용)."""
    global _PREPARED_STORE
    _PREPARED_STORE = None
    _load_preparsed_store()


# ── 단일 카드 파싱 (LLM) ──────────────────────────────────────────

def parse_single_card_llm(
    card_name: str,
    raw_benefits: str,
    client: OpenAI,
    model: str,
) -> Dict:
    """
    카드 1장의 raw_benefits를 LLM에 보내 구조화. 캐시 미스 시에만 호출.
    실패하면 빈 dict 반환 (캐싱하지 않음).
    """
    user_prompt = (
        f"카드명: {card_name}\n"
        f"raw_benefits: {raw_benefits}\n\n"
        f"위 카드를 규칙대로 파싱한다."
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PARSER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            seed=42,
        )
        parsed = json.loads(response.choices[0].message.content)
        return parsed if isinstance(parsed, dict) else {}
    except (APIConnectionError, AuthenticationError, RateLimitError):
        raise
    except Exception:
        return {}


_parse_single_card = parse_single_card_llm  # 내부 호환


def _assemble_parsed_card(card: Dict, llm_data: Dict) -> Dict:
    if not isinstance(llm_data, dict):
        llm_data = {}
    return {
        "card_name": card.get("card_name", ""),
        "card_company": card.get("card_company", ""),
        "annual_fee": int(card.get("annual_fee", 0)),
        "previous_month_requirement": int(card.get("previous_month_requirement", 0)),
        "category_discounts": _sanitize_discounts(llm_data.get("category_discounts", {})),
        "category_caps": _sanitize_caps(llm_data.get("category_caps", {})),
        "raw_benefits": str(card.get("raw_benefits", "")),
        "main_categories": str(card.get("main_categories", "")),
        "similarity_score": float(card.get("similarity_score", 0)),
    }


# ── 응답 검증 ─────────────────────────────────────────────────────

def _sanitize_discounts(d) -> Dict[str, float]:
    """표준 카테고리만 통과, 할인율 float 변환. None/비-dict는 빈 dict로 처리."""
    if not isinstance(d, dict):
        return {}
    clean: Dict[str, float] = {}
    for k, v in d.items():
        if k not in STANDARD_CATEGORIES or v is None:
            continue
        try:
            clean[k] = float(v)
        except (TypeError, ValueError):
            continue
    return clean


def _sanitize_caps(d) -> Dict[str, int]:
    """월 한도 검증. None/비-dict는 빈 dict로 처리."""
    if not isinstance(d, dict):
        return {}
    clean: Dict[str, int] = {}
    for k, v in d.items():
        if k not in STANDARD_CATEGORIES or v is None:
            continue
        try:
            clean[k] = int(v)
        except (TypeError, ValueError):
            continue
    return clean


# ── 공개 API ─────────────────────────────────────────────────────

def parse_candidates(
    candidates: List[Dict],
    model: str = "gpt-4o-mini",
) -> List[Dict]:
    """
    RAG 후보 리스트 → 구조화 리스트

    사전 파싱 JSON → 런타임 캐시 → (없으면) LLM 병렬 호출 순으로 조회.

    Returns:
        [
          {
            "card_name", "card_company", "annual_fee", "previous_month_requirement",
            "category_discounts": {카테고리: 할인율(float)},
            "category_caps":      {카테고리: 월 한도(int)},
            "raw_benefits", "main_categories", "similarity_score"
          }, ...
        ]
    """
    if not candidates:
        return []

    preparsed = _load_preparsed_store()
    runtime_cache = _load_cache()

    keys = [
        cache_key(card.get("card_name", ""), str(card.get("raw_benefits", "")))
        for card in candidates
    ]

    def _lookup(key: str) -> Dict | None:
        data = preparsed.get(key) or runtime_cache.get(key)
        return data if isinstance(data, dict) else None

    miss_indices = [i for i, key in enumerate(keys) if _lookup(key) is None]

    if miss_indices:
        client = create_client()

        def _job(idx: int):
            card = candidates[idx]
            return idx, parse_single_card_llm(
                card.get("card_name", ""),
                str(card.get("raw_benefits", "")),
                client,
                model,
            )

        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
            for idx, llm_data in ex.map(_job, miss_indices):
                if llm_data:
                    runtime_cache[keys[idx]] = llm_data

        _save_cache(runtime_cache)

    return [
        _assemble_parsed_card(card, _lookup(keys[i]) or {})
        for i, card in enumerate(candidates)
    ]
