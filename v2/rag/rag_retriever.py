"""
RAG 카드 후보 검색 모듈
SpendingProfile → 소비 비율 가중 키워드 매칭 → 후보 카드 리스트
"""
from __future__ import annotations
from typing import List, Dict
import numpy as np
import pandas as pd

from spending_analyzer import SpendingProfile, CATEGORY_KEYWORD_MAP


def _score_card(card_categories: str, profile: SpendingProfile) -> float:
    """사용자 소비 비율을 가중치로 카드 카테고리 매칭 점수 계산."""
    card_cats_lower = card_categories.lower()
    score = 0.0
    for user_cat, stat in profile.categories.items():
        keywords = CATEGORY_KEYWORD_MAP.get(user_cat, [user_cat])
        if any(kw.lower() in card_cats_lower for kw in keywords):
            score += stat["ratio"]
    return score


def retrieve(
    profile: SpendingProfile,
    card_df: pd.DataFrame,
    card_embeddings: np.ndarray | None = None,  # 인터페이스 유지용 (향후 벡터 검색 확장 가능)
    top_k: int = 7,
) -> List[Dict]:
    """
    Args:
        profile: SpendingProfile
        card_df: card_processed.csv DataFrame
        card_embeddings: card_embeddings.npy (현재 미사용)
        top_k: 반환할 후보 카드 수

    Returns:
        상위 top_k 카드 dict 리스트 (similarity_score 포함)
    """
    scores = card_df["main_categories"].apply(
        lambda cats: _score_card(str(cats), profile)
    )

    top_indices = scores.nlargest(top_k).index.tolist()

    candidates = []
    for idx in top_indices:
        row = card_df.loc[idx].to_dict()
        row["similarity_score"] = round(float(scores[idx]), 2)
        candidates.append(row)

    return candidates
