"""
GPT 카드 추천 모듈 (Stage 3, LLM API 호출)

이전 버전과의 차이:
  v1: GPT 한 번 호출로 파싱·계산·설명을 다 시킴 → 산수 오류 잦음
  v2: 역할을 3단계로 분리.
      [LLM #1] parser     : raw_benefits 자연어 → {카테고리: 할인율} 구조화
      [Python] calculator : 결정론적 절감액·순위 계산 (산수 오류 0)
      [LLM #2] recommender: 계산 결과 → 자연어 설명 (reason/key_benefits/summary)

전체 파이프라인:
  거래내역 → spending_analyzer → SpendingProfile
                                       ↓
  rag_retriever → candidates (15개)
                                       ↓
  parser.parse_candidates()    [사전 파싱 JSON / LLM] → 카테고리:할인율 구조화
                                       ↓
  calculator.rank_cards()      [Python] → 절감액·순위 결정론적 계산 (top_k)
                                       ↓
  recommender.explain()        [LLM #2] → 자연어 설명 추가
                                       ↓
                                   최종 결과
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
from typing import Dict, List, Union
import numpy as np
import pandas as pd
import openai_config
from openai_config import create_client

_RAG = Path(__file__).parent.parent / "rag"
if str(_RAG) not in sys.path:
    sys.path.insert(0, str(_RAG))

import spending_analyzer
import rag_retriever
from spending_analyzer import SpendingProfile

import benefit_parser as card_parser
import calculator

# ── 데이터 경로 ───────────────────────────────────────────────────
_BASE     = Path(__file__).parent.parent
CARD_CSV  = _BASE / "outputs" / "card_processed.csv"
CARD_EMB  = _BASE / "outputs" / "card_embeddings.npy"
CARD_META = _BASE / "outputs" / "embedding_metadata.json"

# ── API 키 설정 (.env 또는 openai_config.OPENAI_API_KEY) ──────────
OPENAI_API_KEY = ""
if OPENAI_API_KEY:
    openai_config.OPENAI_API_KEY = OPENAI_API_KEY
# ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
당신은 신용카드 혜택 분석 전문가이다.
이미 모든 절감액 계산(카테고리별 절감액, 연간 총 절감액, 순 혜택)은
시스템에서 결정론적으로 끝난 상태로 전달된다.

당신의 역할은 다음 세 가지뿐이다:
  1) 각 카드에 대한 "reason" (5~8문장의 추천 이유) 작성
  2) 각 카드의 "key_benefits" (수치 포함 핵심 혜택 3~5개) 작성
  3) 전체 "overall_summary" (3~4문장) 작성

[중요한 제약]
- 절감액·연회비·순 혜택 등 모든 수치는 입력으로 제공된 값을 그대로 인용한다.
- 새로운 계산을 하거나 입력 수치를 수정하지 않는다.
- 입력에 없는 카테고리/할인율을 임의로 만들지 않는다.
- 반드시 JSON 형식으로만 응답한다 (다른 텍스트 일절 금지).\
"""


# ── 프롬프트 구성 ─────────────────────────────────────────────────

def _format_spending(profile: SpendingProfile) -> str:
    lines = [
        f"- 연간 총 지출: {profile.total_annual:,}원",
        f"- 월 평균 지출: {profile.monthly_avg:,}원",
        "",
        "카테고리별 월 평균 지출:",
    ]
    for cat in profile.top_categories:
        stat = profile.categories[cat]
        lines.append(
            f"  • {cat}: 월 {stat['monthly_avg']:,}원 ({stat['ratio']}%)"
        )
    return "\n".join(lines)


def _format_calculated_cards(ranked: List[Dict]) -> str:
    """calculator가 계산해 둔 결과를 LLM에 보기 좋게 정렬."""
    lines = []
    for card in ranked:
        lines += [
            f"[{card['rank']}위] {card['card_company']} — {card['card_name']}",
            f"  연회비: {card['annual_fee']:,}원",
            f"  {card['monthly_req_note']}",
            f"  연간 총 절감액: {card['total_annual_savings']:,}원",
            f"  순 혜택: {card['net_benefit']:,}원",
            "  카테고리별 절감액:",
        ]
        if not card["savings_breakdown"]:
            lines.append("    (사용자 소비 카테고리와 매칭되는 혜택 없음)")
        for b in card["savings_breakdown"]:
            lines.append(
                f"    • {b['category']}: 월 {b['monthly_avg_spend']:,}원 × "
                f"{b['discount_rate']} = 월 {b['monthly_saving']:,}원 → "
                f"연간 {b['annual_saving']:,}원"
            )
        lines.append("")
    return "\n".join(lines)


def _build_prompt(profile: SpendingProfile, ranked: List[Dict]) -> str:
    return f"""\
아래는 이미 계산이 끝난 카드 추천 결과이다.
각 카드에 대해 reason / key_benefits 를, 전체에 대해 overall_summary 를 작성한다.

=== 사용자 소비 패턴 ===
{_format_spending(profile)}

=== 계산 완료된 추천 카드 (순 혜택 내림차순) ===
{_format_calculated_cards(ranked)}

[출력 JSON 형식]
{{
  "recommendations": [
    {{
      "rank": 1,
      "reason": "5~8문장. 이 카드가 사용자에게 혜택을 주는 모든 카테고리의 월 평균 지출액·할인율·월 절감액을 수치로 설명. 절감액 큰 순으로 언급. 전월 실적 달성 여부와 여유/미달 금액. 연간 총 절감액에서 연회비를 차감한 순 혜택 금액. 다른 후보보다 높은 순위인 이유.",
      "key_benefits": [
        "카테고리명 X% 할인 → 월 절감 XX,XXX원 (연간 XXX,XXX원)",
        "카테고리명 X% 할인 → 월 절감 XX,XXX원 (연간 XXX,XXX원)"
      ]
    }}
  ],
  "overall_summary": "3~4문장. 사용자 소비 패턴 특징, 추천 1위 카드의 강점과 예상 순 혜택, 2·3위와의 차별점."
}}\
"""


# ── LLM 호출 + 결과 합치기 ────────────────────────────────────────

def explain(
    profile: SpendingProfile,
    ranked: List[Dict],
    model: str = "gpt-4o-mini",
) -> Dict:
    """
    calculator로 계산이 끝난 카드 리스트에 LLM 자연어 설명을 덧붙임.

    LLM 응답의 reason/key_benefits/overall_summary 만 사용하고
    수치 관련 필드는 ranked 원본을 그대로 유지.
    """
    client = create_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(profile, ranked)},
        ],
        response_format={"type": "json_object"},
        temperature=0.4,
    )
    llm_out = json.loads(response.choices[0].message.content)

    # rank → 설명 매핑
    explanations = {
        e.get("rank"): e for e in llm_out.get("recommendations", [])
    }

    final_recs = []
    for card in ranked:
        exp = explanations.get(card["rank"], {})
        merged = dict(card)  # 계산 결과 그대로 보존
        merged["reason"] = exp.get("reason", "")
        merged["key_benefits"] = exp.get("key_benefits", [])
        final_recs.append(merged)

    return {
        "recommendations": final_recs,
        "overall_summary": llm_out.get("overall_summary", ""),
    }


# ── 전체 파이프라인 ───────────────────────────────────────────────

def run(
    history_source: Union[str, pd.DataFrame],
    top_rag: int = 15,
    top_recommend: int = 5,
    gpt_model: str = "gpt-4o-mini",
) -> dict:
    """
    카드 거래 내역 → 최종 추천 결과 (v2 파이프라인)

    Returns:
        {
          "recommendations": [...],   # 계산 결과 + LLM 설명
          "overall_summary": "...",
          "spending_profile": {...},
          "rag_candidates": [...]
        }
    """
    # 1) 소비 패턴 분석
    profile = spending_analyzer.analyze(history_source)

    # 2) RAG 후보 검색
    card_df = pd.read_csv(CARD_CSV)
    with open(CARD_META, encoding="utf-8") as f:
        meta = json.load(f)
    prev_req_map = {c["card_name"]: c["previous_month_requirement"] for c in meta["cards"]}
    card_df["previous_month_requirement"] = (
        card_df["card_name"].map(prev_req_map).fillna(0).astype(int)
    )
    card_embeddings = np.load(CARD_EMB) if CARD_EMB.exists() else None
    candidates = rag_retriever.retrieve(profile, card_df, card_embeddings, top_k=top_rag)

    # 3) 파싱 (사전 파싱 JSON 우선, 없을 때만 LLM)
    parsed = card_parser.parse_candidates(candidates, model=gpt_model)

    # 4) 결정론적 계산 + 순위
    ranked = calculator.rank_cards(parsed, profile, top_k=top_recommend)

    # 5) LLM이 자연어 설명만 생성
    result = explain(profile, ranked, model=gpt_model)

    # 6) 렌더링용 부가 데이터
    result["spending_profile"] = profile.to_dict()
    result["rag_candidates"] = candidates
    result["parsed_candidates"] = parsed   # 디버깅용 (파싱 결과 확인)

    return result
