"""
AI 맞춤 신용카드 추천 — Streamlit 앱
사용자의 12개월 카드 거래내역 CSV를 업로드하면
전체 파이프라인을 실행하고 Top 5 카드를 추천한다.

사전 준비 파일 (output/ 디렉터리):
  - card_benefits_structured.json  (2번 스크립트 결과)
  - global_merchant_set.json       (3번 스크립트 결과)
"""

import html
import json
import os
import re
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAI
from generate_recommendation import make_failed_recommendation
from simulate_savings import (
    calculate_benefit,
    deterministic_simulate_card,
    make_zero_result,
    validate_against_deterministic,
    validate_and_fix,
)

# ── 경로 설정 ───────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
OUTPUT_DIR       = BASE_DIR / "output"        # 고정 데이터 (카드 DB, 가맹점 집합)
DEBUG_OUTPUT_DIR = BASE_DIR / "debug_output"  # 디버그 모드 산출물 (단계별 중간 결과)
STRUCTURED_JSON  = OUTPUT_DIR / "card_benefits_structured.json"
MERCHANT_JSON    = OUTPUT_DIR / "global_merchant_set.json"

# ── 디버그 모드 ─────────────────────────────────────────────
# True : 파이프라인을 정상 실행하면서 각 단계 결과를 debug_output/ 에 JSON으로 저장 (로그 용도)s
# False: 파이프라인만 실행, 파일 저장 없음
DEBUG_MODE = True

# ── UI 미리보기 모드 (시연 촬영용) ───────────────────────────
# True : CSV 업로드·분석 버튼 UI는 그대로, 실제로는 debug_output/ 결과를 표시 (파이프라인·API 미사용)
# False: CSV 업로드 후 전체 분석 파이프라인 실행
PREVIEW_MODE = True

DEFAULT_MATCH_MODEL = "gpt-4o"
DEFAULT_SIM_MODEL   = "gpt-4o-mini"
DEFAULT_REC_MODEL   = "gpt-4o"
BATCH_SIZE  = 50
SLEEP_SEC   = 0.5

_RANK_EMOJI = {1: "🥇", 2: "🥈", 3: "🥉", 4: "4️⃣", 5: "5️⃣"}

_MERCHANT_ALIASES: dict[str, str] = {
    "MGC메가커피": "메가커피", "메가MGC커피": "메가커피",
    "투썸플레이트": "투썸플레이스", "파리바게트": "파리바게뜨",
    "파스쿠치": "파스쿠찌", "디즈니 플러스": "디즈니+",
    "디즈니플러스": "디즈니+", "SSG COM": "SSG.COM",
    "할리스커피": "할리스", "삼성 페이": "삼성페이",
    "SSGPAY": "SSG PAY", "SSG페이": "SSG PAY",
    "L.pay": "L.PAY", "L페이": "L.PAY",
}

_ORIGINAL_MERCHANT_HINTS: list[tuple[str, str]] = [
    ("우아한형제들", "배달의민족"),
    ("교통-버스", "버스"),
    ("교통-지하철", "지하철"),
    ("카카오_택시", "카카오T"),
    ("올리브영", "CJ올리브영"),
    ("LG U+", "LG U+"),
    ("쿠팡 주식회사", "쿠팡"),
    ("한국맥도날드", "맥도날드"),
]


def _parse_amount(value) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    cleaned = str(value).replace(",", "").replace("원", "").strip()
    if not cleaned:
        return 0
    return int(float(cleaned))


def _prepare_transactions(df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for row in df.to_dict("records"):
        rows.append({
            "date": str(row["date"]).strip(),
            "merchant_name": str(row["merchant_name"]).strip(),
            "amount": _parse_amount(row["amount"]),
        })
    return rows


def _normalize_matched_merchant(name: str | None) -> str | None:
    if not name:
        return None
    normalized = _MERCHANT_ALIASES.get(str(name).strip(), str(name).strip())
    return normalized or None


def _local_match_merchant(original: str, merchant_set: set[str]) -> str | None:
    text = (original or "").strip()
    if not text:
        return None
    for hint, canonical in _ORIGINAL_MERCHANT_HINTS:
        if hint in text and canonical in merchant_set:
            return canonical
    for merchant in sorted(merchant_set, key=len, reverse=True):
        if len(merchant) >= 2 and merchant in text:
            return merchant
    return None


def _finalize_match_row(row: dict, merchant_set: set[str], fallback_original: str = "") -> dict:
    original = str(row.get("original_merchant") or fallback_original).strip()
    matched = _normalize_matched_merchant(row.get("matched_merchant"))
    if not matched:
        matched = _local_match_merchant(original, merchant_set)
    return {
        "date": str(row.get("date", "")).strip(),
        "original_merchant": original,
        "matched_merchant": matched,
        "amount": _parse_amount(row.get("amount", 0)),
    }


# ── 스타일 (ui_demo.py 기반 · 화이트 테마) ───────────────────
def _inject_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700&display=swap');

    :root {
        --bg: #f5f6f8;
        --glass: rgba(255, 255, 255, 0.72);
        --glass-strong: rgba(255, 255, 255, 0.88);
        --surface-2: #eef0f3;
        --border: rgba(0, 0, 0, 0.08);
        --border-strong: rgba(0, 0, 0, 0.12);
        --text: #1a1a1a;
        --muted: #6b7280;
        --navy: #1e3a5f;
        --navy-dark: #111827;
        --navy-mid: #2c4a7c;
        --navy-glow: rgba(30, 58, 95, 0.14);
        --green: #2d8a47;
        --green-bg: rgba(45, 138, 71, 0.1);
        --shadow: 0 2px 16px rgba(0, 0, 0, 0.06);
        --radius: 14px;
    }

    html, body, [class*="css"] {
        font-family: 'Noto Sans KR', sans-serif;
        color: var(--text);
    }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
    [data-testid="stToolbar"], .main, .main .block-container {
        background: var(--bg) !important;
        color: var(--text) !important;
    }
    .block-container { max-width: 1280px; padding-top: 1rem; }

    .glass-box {
        background: var(--glass);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        box-shadow: var(--shadow);
    }

    .demo-hero, .app-hero {
        background: linear-gradient(135deg, rgba(255,255,255,0.92) 0%, rgba(240,245,252,0.88) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-left: 4px solid var(--navy);
        border-radius: 18px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.25rem;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow);
    }
    .demo-hero::before, .app-hero::before {
        content: '';
        position: absolute;
        top: -40%; right: -10%;
        width: 320px; height: 320px;
        background: radial-gradient(circle, var(--navy-glow) 0%, transparent 70%);
        pointer-events: none;
    }
    .demo-hero h1, .app-hero h1 {
        margin: 0 0 0.35rem;
        font-size: 1.65rem;
        font-weight: 700;
        color: var(--navy-dark);
        letter-spacing: -0.02em;
    }
    .demo-hero p, .app-hero p { margin: 0; color: var(--muted); font-size: 0.92rem; }
    .demo-badge, .accent-tag, .hero-tag {
        display: inline-block;
        margin-top: 0.75rem;
        margin-right: 0.35rem;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(30, 58, 95, 0.08);
        border: 1px solid rgba(30, 58, 95, 0.22);
        color: var(--navy);
    }
    .accent-tags { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.75rem; }

    .empty-state {
        text-align: center;
        padding: 2.5rem 1.5rem;
        background: var(--glass);
        border: 1px dashed var(--border-strong);
        border-radius: var(--radius);
        box-shadow: var(--shadow);
    }
    .empty-state .icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
    .empty-state p { color: var(--text); margin: 0.35rem 0; }
    .accent-box, .accent-hint, .summary-box {
        background: var(--glass-strong) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        line-height: 1.65;
        font-size: 0.88rem;
        box-shadow: var(--shadow);
    }

    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.75rem;
        margin-bottom: 1.25rem;
    }
    @media (max-width: 768px) { .kpi-grid { grid-template-columns: repeat(2, 1fr); } }
    .kpi-card {
        background: var(--glass);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.85rem 1rem;
        box-shadow: var(--shadow);
    }
    .kpi-card .label {
        font-size: 0.72rem; color: var(--muted); font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.04em;
    }
    .kpi-card .value {
        font-size: 1.35rem; font-weight: 700; color: var(--text);
        margin-top: 0.15rem; line-height: 1.2;
    }
    .kpi-card .sub { font-size: 0.72rem; color: var(--green); margin-top: 0.2rem; }
    .kpi-card.highlight {
        background: linear-gradient(135deg, rgba(240,245,252,0.92), rgba(255,255,255,0.78));
        border-color: rgba(30, 58, 95, 0.22);
    }
    .kpi-card.highlight .value { color: var(--navy-dark); }

    .section-hd, .section-label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 1.25rem 0 0.85rem;
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text);
    }
    .section-hd .dot {
        width: 6px; height: 6px; border-radius: 50%; background: var(--navy);
    }

    .chart-panel {
        background: var(--glass);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1rem 1.15rem;
        height: 100%;
        box-shadow: var(--shadow);
    }
    .chart-panel h4 {
        margin: 0 0 0.85rem; font-size: 0.85rem;
        font-weight: 600; color: var(--muted);
    }

    .hbar-row {
        display: flex; align-items: center; gap: 0.6rem;
        margin-bottom: 0.45rem; font-size: 0.78rem;
    }
    .hbar-label {
        width: 72px; text-align: right; color: var(--text);
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex-shrink: 0;
    }
    .hbar-track {
        flex: 1; height: 18px; background: var(--surface-2);
        border-radius: 4px; overflow: hidden;
    }
    .hbar-fill {
        height: 100%; border-radius: 4px;
        background: linear-gradient(90deg, rgba(30,58,95,0.45), var(--navy));
    }
    .hbar-val {
        width: 72px; text-align: right; color: var(--muted);
        font-size: 0.72rem; flex-shrink: 0;
    }

    .cat-grid { display: flex; flex-wrap: wrap; gap: 0.5rem; }
    .cat-chip {
        display: flex; align-items: center; gap: 0.45rem;
        padding: 0.45rem 0.75rem;
        background: var(--glass-strong);
        border: 1px solid var(--border);
        border-radius: 10px; font-size: 0.78rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .cat-chip .pct { font-weight: 700; color: var(--navy-dark); }
    .cat-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }

    .credit-card {
        background: linear-gradient(135deg, rgba(248,250,255,0.95) 0%, rgba(240,245,252,0.9) 50%, rgba(255,255,255,0.85) 100%);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(30, 58, 95, 0.22);
        border-radius: 18px;
        padding: 1.5rem 1.75rem;
        position: relative; overflow: hidden;
        box-shadow: 0 8px 32px rgba(30, 58, 95, 0.1), var(--shadow);
    }
    .credit-card::after {
        content: '';
        position: absolute; top: -50%; right: -20%;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(30, 58, 95, 0.1), transparent 70%);
    }
    .cc-rank {
        font-size: 0.75rem; font-weight: 600; color: var(--navy);
        letter-spacing: 0.06em; text-transform: uppercase;
    }
    .cc-name {
        font-size: 1.45rem; font-weight: 700; color: var(--text);
        margin: 0.3rem 0 0.15rem; letter-spacing: -0.02em;
    }
    .cc-company { font-size: 0.85rem; color: var(--muted); margin-bottom: 1.25rem; }
    .cc-metrics {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem;
    }
    .cc-metric {
        background: rgba(255, 255, 255, 0.65);
        border: 1px solid var(--border);
        border-radius: 10px; padding: 0.65rem 0.8rem;
    }
    .cc-metric .ml { font-size: 0.68rem; color: var(--muted); }
    .cc-metric .mv { font-size: 1.05rem; font-weight: 700; color: var(--navy-dark); margin-top: 0.1rem; }
    .cc-metric.hero .mv { font-size: 1.25rem; color: var(--green); }
    .cc-summary {
        margin-top: 1rem; padding: 0.75rem 1rem;
        background: rgba(255, 255, 255, 0.55);
        border: 1px solid var(--border);
        border-radius: 10px; font-size: 0.85rem;
        line-height: 1.6; color: #444;
    }

    .benefit-list { margin-top: 0.5rem; }
    .benefit-item { margin-bottom: 0.65rem; }
    .benefit-hd {
        display: flex; justify-content: space-between;
        font-size: 0.8rem; margin-bottom: 0.25rem;
    }
    .benefit-hd .name { color: var(--text); font-weight: 500; }
    .benefit-hd .amt { color: var(--green); font-weight: 600; }
    .benefit-track {
        height: 6px; background: var(--surface-2);
        border-radius: 3px; overflow: hidden;
    }
    .benefit-fill {
        height: 100%; border-radius: 3px;
        background: linear-gradient(90deg, #3d9e56, var(--green));
    }

    .card-list-item {
        display: flex; align-items: center; gap: 1rem;
        padding: 0.85rem 1rem;
        background: var(--glass);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        margin-bottom: 0.5rem;
        box-shadow: var(--shadow);
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .card-list-item:hover {
        border-color: rgba(30, 58, 95, 0.28);
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    .cli-rank { font-size: 1.3rem; width: 36px; text-align: center; flex-shrink: 0; }
    .cli-info { flex: 1; min-width: 0; }
    .cli-info .title {
        font-size: 0.88rem; font-weight: 600; color: var(--text);
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .cli-info .fee { font-size: 0.72rem; color: var(--muted); margin-top: 0.1rem; }
    .cli-saving { text-align: right; flex-shrink: 0; }
    .cli-saving .val { font-size: 1rem; font-weight: 700; color: var(--green); }
    .cli-saving .lbl { font-size: 0.68rem; color: var(--muted); }

    .compare-row {
        display: flex; align-items: center; gap: 0.6rem;
        margin-bottom: 0.55rem; font-size: 0.78rem;
    }
    .compare-label {
        width: 130px; white-space: nowrap; overflow: hidden;
        text-overflow: ellipsis; color: var(--text); flex-shrink: 0;
    }
    .compare-track {
        flex: 1; height: 22px; background: var(--surface-2);
        border-radius: 5px; overflow: hidden;
    }
    .compare-fill { height: 100%; border-radius: 5px; }
    .compare-val {
        width: 80px; text-align: right; color: var(--navy-dark);
        font-weight: 600; font-size: 0.75rem; flex-shrink: 0;
    }

    .kw-pill, .kw-highlight {
        display: inline;
        background: var(--green-bg);
        color: var(--green);
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        font-weight: 600;
    }

    .ad-panel {
        background: var(--glass);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.25rem 1.5rem;
        box-shadow: var(--shadow);
        margin-top: 0.5rem;
    }
    .ad-panel h3 { margin: 0 0 0.5rem; color: var(--navy-dark); font-size: 1.1rem; }
    .ad-benefit-item {
        padding: 0.4rem 0.65rem; margin: 0.25rem 0;
        background: var(--glass-strong); border: 1px solid var(--border);
        border-radius: 8px; font-size: 0.82rem; color: var(--text);
    }
    .ad-disclaimer { text-align: right; font-size: 0.72rem; color: var(--muted); margin-top: 0.5rem; }

    div[data-testid="stMetric"] { display: none; }

    [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
        background: #ffffff !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] [data-baseweb="input"] > div {
        background: #fff !important; color: var(--text) !important;
        border: 1px solid var(--border) !important;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    label, .stSubheader { color: var(--text) !important; }
    .stCaption, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }

    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        background: var(--navy) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        box-shadow: 0 2px 8px rgba(30, 58, 95, 0.25) !important;
    }
    .stButton > button[kind="primary"] *,
    .stButton > button[data-testid="stBaseButton-primary"] *,
    .stButton > button[data-testid="baseButton-primary"] * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        fill: #ffffff !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {
        background: var(--navy-mid) !important;
        color: #ffffff !important;
    }
    .stButton > button[kind="primary"]:hover *,
    .stButton > button[data-testid="stBaseButton-primary"]:hover *,
    .stButton > button[data-testid="baseButton-primary"]:hover * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    .stLinkButton > a, [data-testid="stLinkButton"] a {
        background: #fff !important;
        color: var(--navy) !important;
        border: 1px solid rgba(30, 58, 95, 0.3) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }

    [data-testid="stFileUploader"] section {
        background: var(--glass) !important;
        border: 1px dashed var(--border-strong) !important;
        border-radius: var(--radius) !important;
    }
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small { color: var(--text) !important; }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--glass) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        box-shadow: var(--shadow) !important;
    }

    [data-testid="stTabs"] button {
        font-size: 0.82rem !important; font-weight: 500 !important;
        color: var(--muted) !important; background: transparent !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--text) !important;
        border-bottom-color: var(--navy) !important;
    }

    details[data-testid="stExpander"] {
        background: var(--glass) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        box-shadow: var(--shadow) !important;
    }
    details[data-testid="stExpander"] summary,
    details[data-testid="stExpander"] summary span { color: var(--text) !important; }

    [data-testid="stProgress"] > div > div { background-color: var(--navy) !important; }
    [data-testid="stProgress"] > div { background-color: var(--surface-2) !important; }

    [data-testid="stVegaLiteChart"],
    [data-testid="stArrowVegaLiteChart"] {
        background: var(--glass-strong) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        box-shadow: var(--shadow) !important;
        padding: 0.5rem !important;
    }

    [data-testid="stDataFrame"] {
        background: var(--glass-strong) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        box-shadow: var(--shadow) !important;
    }
    [data-testid="stDataFrame"] * { color: var(--text) !important; }

    hr { border-color: var(--border) !important; opacity: 1; }
    </style>
    """, unsafe_allow_html=True)


def _fmt_ui(n: int | float) -> str:
    return f"{int(n):,}"


def _highlight_saving_keywords(text: str) -> str:
    safe = html.escape(str(text))
    return re.sub(
        r"(월 평균 [\d,]+원(?: 절감)?)",
        r'<span class="kw-pill">\1</span>',
        safe,
    )


_UI_CATEGORY_KEYWORDS = {
    "쿠팡": "쇼핑", "무신사": "쇼핑", "네이버": "쇼핑", "11번가": "쇼핑",
    "스타벅스": "카페", "이디야": "카페", "메가커피": "카페",
    "배달의민족": "배달", "요기요": "배달", "쿠팡이츠": "배달",
    "이마트": "마트", "홈플러스": "마트", "GS25": "편의점", "CU": "편의점",
    "세븐일레븐": "편의점", "SK에너지": "주유", "GS칼텍스": "주유",
    "넷플릭스": "구독", "유튜브": "구독", "멜론": "구독", "티빙": "구독",
    "카카오T": "교통", "쏘카": "교통", "아고다": "여행",
}
_UI_CAT_COLORS = {
    "쇼핑": "#1e3a5f", "카페": "#c97b5a", "배달": "#7ec86e",
    "마트": "#6ba3d6", "편의점": "#b07cc6", "주유": "#e8a838",
    "구독": "#5bc4b0", "교통": "#8a9bae", "여행": "#d47b9f", "기타": "#666",
}


def _ui_category_stats(tx: list[dict]) -> dict[str, int]:
    cats: dict[str, int] = defaultdict(int)
    for t in tx:
        merchant = str(t.get("matched_merchant") or "")
        amount = _parse_amount(t.get("amount", 0))
        matched_cat = "기타"
        for kw, cat in _UI_CATEGORY_KEYWORDS.items():
            if kw in merchant:
                matched_cat = cat
                break
        cats[matched_cat] += amount
    return dict(cats)


def _ui_hbar_chart(merchants: list[tuple[str, int]], title: str) -> str:
    if not merchants:
        return ""
    max_val = max(v for _, v in merchants)
    rows = []
    for name, val in merchants:
        pct = (val / max_val * 100) if max_val else 0
        short = name if len(name) <= 8 else name[:7] + "…"
        rows.append(
            f'<div class="hbar-row">'
            f'<span class="hbar-label" title="{html.escape(name)}">{html.escape(short)}</span>'
            f'<div class="hbar-track"><div class="hbar-fill" style="width:{pct:.1f}%"></div></div>'
            f'<span class="hbar-val">{_fmt_ui(val)}원</span></div>'
        )
    return f'<div class="chart-panel"><h4>{html.escape(title)}</h4>{"".join(rows)}</div>'


def _ui_category_chips(cats: dict[str, int]) -> str:
    total = sum(cats.values()) or 1
    chips = []
    for cat, val in sorted(cats.items(), key=lambda x: -x[1]):
        pct = val / total * 100
        color = _UI_CAT_COLORS.get(cat, "#666")
        chips.append(
            f'<div class="cat-chip">'
            f'<span class="cat-dot" style="background:{color}"></span>'
            f'<span>{html.escape(cat)}</span>'
            f'<span class="pct">{pct:.0f}%</span>'
            f'<span style="color:var(--muted);font-size:0.72rem">{_fmt_ui(val)}원</span>'
            f'</div>'
        )
    return f'<div class="chart-panel"><h4>카테고리별 소비 비중</h4><div class="cat-grid">{"".join(chips)}</div></div>'


def _ui_compare_bars(top5: list[dict]) -> str:
    if not top5:
        return ""
    max_saving = max(c.get("net_saving", 0) for c in top5) or 1
    colors = ["#1e3a5f", "#2c4a7c", "#374151", "#4b5563", "#6b7280"]
    rows = []
    for i, card in enumerate(top5):
        saving = card.get("net_saving", 0)
        pct = saving / max_saving * 100
        label = f"{card.get('card_company', '')} {card.get('card_name', '')}"
        short = label if len(label) <= 16 else label[:15] + "…"
        rank = card.get("rank", i + 1)
        rows.append(
            f'<div class="compare-row">'
            f'<span class="compare-label" title="{html.escape(label)}">'
            f'{_RANK_EMOJI.get(rank, "💳")} {html.escape(short)}</span>'
            f'<div class="compare-track">'
            f'<div class="compare-fill" style="width:{pct:.1f}%;background:{colors[i % len(colors)]}"></div>'
            f'</div>'
            f'<span class="compare-val">{_fmt_ui(saving)}원</span></div>'
        )
    return f'<div class="chart-panel"><h4>Top 5 순 절감액 비교</h4>{"".join(rows)}</div>'


def _ui_benefit_bars(details: list[dict]) -> str:
    active = [b for b in details if b.get("estimated_saving", 0) > 0]
    if not active:
        return ""
    max_s = max(b["estimated_saving"] for b in active)
    items = []
    for b in sorted(active, key=lambda x: -x["estimated_saving"]):
        s = b["estimated_saving"]
        pct = s / max_s * 100
        items.append(
            f'<div class="benefit-item">'
            f'<div class="benefit-hd">'
            f'<span class="name">{html.escape(b.get("benefit_name", ""))}</span>'
            f'<span class="amt">{_fmt_ui(s)}원</span></div>'
            f'<div class="benefit-track">'
            f'<div class="benefit-fill" style="width:{pct:.1f}%"></div></div></div>'
        )
    return f'<div class="benefit-list">{"".join(items)}</div>'


def _ui_premium_card(rec: dict, discount: int) -> str:
    saving = rec.get("net_saving", 0)
    fee = rec.get("annual_fee", 0)
    summary = _highlight_saving_keywords(_display_summary(rec))
    benefits_html = _ui_benefit_bars(rec.get("benefit_details", []))
    return f"""
    <div class="credit-card">
        <div class="cc-rank">🥇 Best Recommendation</div>
        <div class="cc-name">{html.escape(rec.get("card_name", ""))}</div>
        <div class="cc-company">{html.escape(rec.get("card_company", ""))}</div>
        <div class="cc-metrics">
            <div class="cc-metric">
                <div class="ml">연회비</div>
                <div class="mv">{_fmt_ui(fee)}원</div>
            </div>
            <div class="cc-metric">
                <div class="ml">연간 총 할인</div>
                <div class="mv">{_fmt_ui(discount)}원</div>
            </div>
            <div class="cc-metric hero">
                <div class="ml">순 절감액</div>
                <div class="mv">{_fmt_ui(saving)}원/년</div>
            </div>
        </div>
        <div class="cc-summary">{summary}</div>
    </div>
    {benefits_html}
    """


def _ui_compact_cards(recs: list[dict], top5: list[dict]) -> str:
    items = []
    for rec in recs:
        rank = rec.get("rank", 0)
        if rank == 1:
            continue
        saving = rec.get("net_saving", 0)
        fee = rec.get("annual_fee", 0)
        discount = next((c.get("total_discount", 0) for c in top5 if c.get("rank") == rank), 0)
        emoji = _RANK_EMOJI.get(rank, "💳")
        items.append(
            f'<div class="card-list-item">'
            f'<div class="cli-rank">{emoji}</div>'
            f'<div class="cli-info">'
            f'<div class="title">{html.escape(rec.get("card_company", ""))} — {html.escape(rec.get("card_name", ""))}</div>'
            f'<div class="fee">연회비 {_fmt_ui(fee)}원 · 총할인 {_fmt_ui(discount)}원</div>'
            f'</div>'
            f'<div class="cli-saving">'
            f'<div class="val">{_fmt_ui(saving)}원</div>'
            f'<div class="lbl">순 절감/년</div></div></div>'
        )
    return "".join(items)


def _render_card_benefit_expanders(rec: dict) -> None:
    details = rec.get("benefit_details", [])
    if not details:
        return
    for b in details:
        matched = b.get("matched_merchants", "")
        is_match = (
            "이용하지 않으셨습니다" not in matched
            and "이용 내역 없음" not in matched
            and matched.strip() != ""
        )
        icon = "✅" if is_match else "ℹ️"
        saving_val = b.get("estimated_saving", 0)
        with st.expander(f"{icon} {b.get('benefit_name', '')} — {_fmt_ui(saving_val)}원"):
            condition = str(b.get("condition", "")).strip()
            if condition and not (PREVIEW_MODE and "실패" in condition):
                st.markdown(f"**조건** {condition}")
            st.markdown(f"**매칭 가맹점** {matched}")
            st.markdown(f"**내 소비** {b.get('my_spending', '')}")
            st.code(b.get("calculation", ""), language=None)


def _display_summary(rec: dict) -> str:
    summary = str(rec.get("summary", "")).strip()
    if PREVIEW_MODE and summary == "추천 설명 생성 실패":
        total = str(rec.get("total_summary", "")).strip()
        return total or "고객님의 소비 패턴을 분석해 절감 효과가 큰 카드를 선정했습니다."
    return summary


def _run_demo_analysis(progress_bar, status_text) -> None:
    steps = [
        (15, "[1/5] 거래내역 매칭 중..."),
        (30, "[2/5] 카드별 거래 매핑 중..."),
        (55, "[3/5] 100개 카드 시뮬레이션 중..."),
        (75, "[4/5] Top 5 선정 중..."),
        (90, "[5/5] 추천 설명 생성 중..."),
        (100, "✅ 분석 완료!"),
    ]
    for pct, msg in steps:
        status_text.text(msg)
        progress_bar.progress(pct)
        time.sleep(0.12)
    progress_bar.empty()
    status_text.empty()


# ── 헬퍼 ────────────────────────────────────────────────────
def extract_json_array(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if m:
        return m.group(1).strip()
    s, e = text.find("["), text.rfind("]")
    return text[s:e+1] if s != -1 and e != -1 else text

def extract_json_obj(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if m:
        return m.group(1).strip()
    s, e = text.find("{"), text.rfind("}")
    return text[s:e+1] if s != -1 and e != -1 else text

def get_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        st.error("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        st.stop()
    return OpenAI(api_key=key)


# ── 파이프라인 함수들 ─────────────────────────────────────────

def run_fuzzy_matching(client: OpenAI, transactions: list[dict],
                       merchants: list[str], model: str,
                       progress_bar, status_text) -> list[dict]:
    merchant_set = set(merchants)
    merchant_set_str = "\n".join(merchants)
    batches = [transactions[i:i+BATCH_SIZE] for i in range(0, len(transactions), BATCH_SIZE)]
    if not batches:
        return []
    results: list[dict] = []
    llm_failed_batches = 0

    system = ("You are a transaction matcher for Korean credit card benefits. "
              "Respond ONLY with a valid JSON array.")
    user_template = """\
아래 거래내역 {n}건의 merchant_name을 global_merchant_set과 매칭해라.

[global_merchant_set]
{mset}

[거래내역]
{txs}

[매칭 규칙]
1. 정확히 일치하지 않아도 같은 가맹점이면 매칭한다.
   - "씨유(CU) 분당효자촌 현대점" → "CU"
   - "스타벅스_주문-에스씨" → "스타벅스"
   - "우아한형제들" → "배달의민족"
   - "교통-버스25건" → "버스"
   - "교통-지하철2건" → "지하철"
2. 지점명, 번호, 특수문자, 법인표기는 무시하고 브랜드명으로 판단한다.
3. 운영사와 브랜드가 다른 경우 브랜드로 매칭한다.
4. 매칭 불가 또는 확신 없으면 matched_merchant를 null로 한다.

[출력 형식]
입력된 거래내역과 동일한 순서로 JSON 배열만 반환한다.
[{{"date":"...", "original_merchant":"...", "matched_merchant":"..." or null, "amount":숫자}}]"""

    for idx, batch in enumerate(batches):
        done = min((idx + 1) * BATCH_SIZE, len(transactions))
        status_text.text(f"[1/5] 거래 매칭 중... {done}/{len(transactions)}건")
        progress_bar.progress(int((idx + 1) / len(batches) * 20))

        tx_json = json.dumps(
            [{"date": t["date"], "original_merchant": t["merchant_name"], "amount": t["amount"]}
             for t in batch],
            ensure_ascii=False,
        )
        user_msg = user_template.format(n=len(batch), mset=merchant_set_str, txs=tx_json)

        batch_rows: list[dict] | None = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system},
                               {"role": "user",   "content": user_msg}],
                    temperature=0,
                )
                parsed = json.loads(extract_json_array(resp.choices[0].message.content or ""))
                if isinstance(parsed, list) and len(parsed) == len(batch):
                    batch_rows = [
                        _finalize_match_row(
                            {
                                **item,
                                "original_merchant": item.get("original_merchant") or tx["merchant_name"],
                                "amount": item.get("amount", tx["amount"]),
                            },
                            merchant_set,
                            tx["merchant_name"],
                        )
                        for item, tx in zip(parsed, batch)
                    ]
                    break
            except Exception:
                pass
            time.sleep(SLEEP_SEC)

        if batch_rows is None:
            llm_failed_batches += 1
            batch_rows = [
                _finalize_match_row(
                    {
                        "date": t["date"],
                        "original_merchant": t["merchant_name"],
                        "matched_merchant": None,
                        "amount": t["amount"],
                    },
                    merchant_set,
                    t["merchant_name"],
                )
                for t in batch
            ]

        results.extend(batch_rows)

    matched = [r for r in results if r.get("matched_merchant")]
    if llm_failed_batches and matched:
        status_text.text(
            f"[1/5] LLM 매칭 {llm_failed_batches}개 배치 실패 → 로컬 규칙으로 보완 ({len(matched)}건 매칭)"
        )
    return matched


def map_cards(filtered_tx: list[dict], structured: list[dict]) -> dict:
    ALIASES = {
        "MGC메가커피": "메가커피", "메가MGC커피": "메가커피",
        "투썸플레이트": "투썸플레이스", "파리바게트": "파리바게뜨",
        "파스쿠치": "파스쿠찌", "디즈니 플러스": "디즈니+",
        "디즈니플러스": "디즈니+", "SSG COM": "SSG.COM",
        "할리스커피": "할리스", "삼성 페이": "삼성페이",
        "SSGPAY": "SSG PAY", "SSG페이": "SSG PAY",
        "L.pay": "L.PAY", "L페이": "L.PAY",
    }
    def normalize(m: str) -> str:
        return ALIASES.get(m, m) if m else m

    card_map: dict = {}
    for card in structured:
        name = card["card_name"]
        has_all = any(
            b["merchant_type"] == "ALL" for b in card.get("benefits", [])
            if not any(kw in b.get("category", "") for kw in ["해외", "해외이용", "해외가맹점"])
        )
        eligible_norm = {normalize(m) for m in card.get("eligible_merchants", [])}
        txs = filtered_tx[:] if has_all else [
            tx for tx in filtered_tx
            if normalize(tx.get("matched_merchant", "")) in eligible_norm
        ]
        card_map[name] = {
            "card_company": card["card_company"],
            "has_all_type_benefit": has_all,
            "transactions": txs,
        }
    return card_map


def aggregate_transactions(transactions: list[dict], prev_req: int) -> dict:
    monthly: dict = defaultdict(
        lambda: {"total": 0, "merchants": defaultdict(lambda: {"count": 0, "total": 0})}
    )
    for tx in transactions:
        parts = tx["date"].replace(".", "-").split("-")
        mk = f"{parts[0]}-{parts[1]}"
        amt = _parse_amount(tx.get("amount", 0))
        m = tx.get("matched_merchant", "")
        monthly[mk]["total"] += amt
        monthly[mk]["merchants"][m]["count"] += 1
        monthly[mk]["merchants"][m]["total"] += amt

    summary = []
    prev = prev_req
    for mk in sorted(monthly.keys()):
        d = monthly[mk]
        summary.append({
            "month": mk,
            "month_total": d["total"],
            "prev_month_total": prev,
            "transactions_by_merchant": [
                {"matched_merchant": m, "count": v["count"], "total_amount": v["total"]}
                for m, v in sorted(d["merchants"].items(), key=lambda x: -x[1]["total"])
            ],
        })
        prev = d["total"]
    return {"monthly_summary": summary}


def run_simulation(client: OpenAI, structured: list[dict],
                   card_map: dict, model: str,
                   progress_bar, status_text) -> list[dict]:
    system = ("You are a Korean credit card benefit simulator. "
              "Calculate maximum savings over 1 year. "
              "Respond ONLY with a valid JSON object. No markdown.")
    user_template = """\
아래 카드 혜택과 사용자 월별 거래 집계로 연간 최대 절감액을 계산해라.

[카드 정보]
{card_json}

[월별 거래 집계]
{agg_json}

[절대 규칙]
- discount_value는 퍼센트(%). 7 → 7% / 1.2 → 1.2%
- 0.7 → 0.7%, 0.8 → 0.8%, 2.0 → 2.0%다. 7%, 8%, 20%로 키워서 계산하지 마라.
- monthly_breakdown discount 합계 = total_discount
- net_saving = total_discount - annual_fee
- benefit_breakdown은 benefit_id 기준 연간 합산 (월별 반복 금지)
- sum(benefit_breakdown[i].total_discount) = total_discount
- benefit_breakdown 합계와 monthly_breakdown 합계가 다르면 잘못된 결과다.
- discount_type amount의 discount_value는 원 단위 정액 혜택이다. 퍼센트로 계산하지 마라.
- discount_type point는 현금 절감액으로 환산하지 말고 보수적으로 0 처리해라.
- 주유 혜택의 "60원/L", "리터당 60원", "ℓ당 60원" 같은 리터당 원 단위 혜택은
  거래내역에 리터 수가 없으므로 계산하지 마라. 60%로 해석하면 안 된다.
- ALL 혜택이어도 "6개 영역 중 이용금액 1위/2위 영역", "가장 많이 쓴 영역"처럼
  소비 영역 순위를 먼저 판단해야 하는 혜택은 전체 거래에 적용하지 마라.
- ALL 혜택이어도 간편결제, 앱카드, FAN페이처럼 결제수단 조건이 붙은 혜택은
  거래내역에서 결제수단을 확인할 수 없으므로 보수적으로 0 처리해라.

[원칙]
1. 가장 유리한 방향으로 (SELECT형은 최적 옵션 선택)
2. 월별 독립 계산 (전월실적 조건, 월 한도 매월 초기화)
3. ALL: 전체 적용 / A/B/C: merchants 매칭 / SERVICE: 제외
4. offline_only: 쿠팡·배달앱·온라인쇼핑 → 온라인 취급하여 제외
5. 거래 없으면 net_saving: 0

[출력 JSON]
{{
  "card_name":"","card_company":"",
  "selected_options":{{"SELECT_1":null}},
  "annual_fee":0,"total_discount":0,"net_saving":0,
  "monthly_breakdown":[{{"month":"","month_total":0,"prev_month_total":0,"requirement_met":true,"discount":0,"limit_hit":false}}],
  "benefit_breakdown":[{{"benefit_id":"b1","category":"","merchant_type":"","total_applied_amount":0,"total_discount":0,"applied_count":0}}],
  "calculation_notes":""
}}"""

    results = []
    total = len(structured)
    for idx, card in enumerate(structured):
        name = card["card_name"]
        pct = int((idx + 1) / total * 55) + 20
        status_text.text(f"[3/5] 시뮬레이션 중... {idx+1}/{total} ({name[:20]})")
        progress_bar.progress(pct)

        txs = card_map.get(name, {}).get("transactions", [])
        if not txs:
            results.append(make_zero_result(card))
            continue

        prev_req = card.get("previous_month_requirement") or 0
        aggregated = aggregate_transactions(txs, prev_req)
        card_for_prompt = {k: v for k, v in card.items() if k != "eligible_merchants"}
        deterministic = deterministic_simulate_card(
            card,
            txs,
            note="LLM 결과 검산 실패 시 사용되는 코드 기반 보수 계산",
        )
        max_expected_discount = deterministic.get("total_discount", 0)

        user_msg = user_template.format(
            card_json=json.dumps(card_for_prompt, ensure_ascii=False, indent=2),
            agg_json=json.dumps(aggregated, ensure_ascii=False, indent=2),
        )

        sim_result = None
        for _ in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system},
                               {"role": "user",   "content": user_msg}],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                parsed = json.loads(extract_json_obj(resp.choices[0].message.content or ""))
                ok, _ = validate_and_fix(parsed, max_expected_discount=max_expected_discount)
                if ok:
                    ok, _ = validate_against_deterministic(parsed, deterministic)
                if ok:
                    sim_result = parsed
                    break
            except Exception:
                pass
            time.sleep(SLEEP_SEC)

        if sim_result is None:
            deterministic["calculation_notes"] = (
                "[코드 기반 보수 계산 사용] " + deterministic.get("calculation_notes", "")
            )
            sim_result = deterministic

        results.append(sim_result)
        time.sleep(SLEEP_SEC)

    results.sort(key=lambda x: x.get("net_saving", 0), reverse=True)
    return results


def extract_top5(sim_results: list[dict]) -> list[dict]:
    return [{"rank": i + 1, **r} for i, r in enumerate(sim_results[:5])]


def merge_breakdown(breakdown: list[dict]) -> list[dict]:
    merged: dict = {}
    for b in breakdown:
        bid = b.get("benefit_id", "x")
        if bid not in merged:
            merged[bid] = {
                **b,
                "total_applied_amount": 0,
                "total_discount": 0,
                "applied_count": 0,
                "matched_merchants": [],
            }
        merged[bid]["total_applied_amount"] += b.get("total_applied_amount", 0)
        merged[bid]["total_discount"]       += b.get("total_discount", 0)
        merged[bid]["applied_count"]        += b.get("applied_count", 0)
        merged[bid]["matched_merchants"].extend(b.get("matched_merchants", []))
    return list(merged.values())


def get_benefit_structure(card_name: str, structured_map: dict) -> str:
    card = structured_map.get(card_name)
    if not card:
        return "데이터 없음"
    lines = []
    for b in card.get("benefits", []):
        bid, mt, cat, dv, ms = (
            b.get("benefit_id", ""), b.get("merchant_type", ""),
            b.get("category", ""), b.get("discount_value", ""), b.get("merchants", []),
        )
        dtype = b.get("discount_type", "")
        notes = (b.get("conditions") or {}).get("notes")
        if dtype == "rate":
            benefit_desc = f"{dv}%"
        elif dtype == "amount":
            benefit_desc = f"{int(dv or 0):,}원 정액"
        elif dtype == "point":
            benefit_desc = "포인트/마일리지형 (현금 환산 불확실)"
        else:
            benefit_desc = "혜택값 불명확"
        note_desc = f" / 조건: {notes}" if notes else ""
        if mt == "SERVICE":
            lines.append(f"[{bid}] {cat} — SERVICE (가맹점 없음){note_desc}")
        elif mt == "ALL":
            lines.append(f"[{bid}] {cat} — ALL ({benefit_desc}){note_desc}")
        else:
            ms_str = ", ".join(ms)
            lines.append(f"[{bid}] {cat} {benefit_desc} — 대상: {ms_str}{note_desc}")
    return "\n".join(lines)


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
        merchant_summary = "; ".join(
            f"{row.get('merchant', '기타')} 연 {row.get('count', 0)}건 총 {row.get('total', 0):,}원"
            for row in merchants
        ) or "가맹점별 집계 없음"
        lines.append(
            f"[{bid}] {b.get('category', '')} / {b.get('merchant_type', '')}\n"
            f"- 시뮬레이션 적용 합계: 연 {b.get('applied_count', 0)}건, "
            f"총 {b.get('total_applied_amount', 0):,}원, "
            f"절감 {b.get('total_discount', 0):,}원\n"
            f"- 적용 가맹점 전체: {merchant_summary}"
        )

    return "\n\n".join(lines) if lines else "적용 거래 집계 없음"


def generate_recommendations(client: OpenAI, top5: list[dict], filtered_tx: list[dict],
                              structured_map: dict, model: str,
                              progress_bar, status_text) -> list[dict]:
    system = ("You are a Korean credit card recommendation expert. "
              "Write user-facing recommendation reports in friendly Korean. "
              "Respond ONLY with a valid JSON object.")
    user_template = """\
사용자에게 보여줄 카드 추천 상세 설명을 작성해라.

[카드 정보]
카드명: {name} | 카드사: {company}
연회비: {fee}원 | 순 절감액: {saving}원 | 총 할인액: {discount}원

[카드 혜택 상세 (가맹점별)]
{benefit_struct}

[혜택별 절감 내역]
{breakdown}

[혜택별 실제 적용 거래 집계 — 전체]
아래 데이터는 이 카드의 각 혜택에 실제로 매칭된 거래를 가맹점별로 전부 집계한 것이다.
my_spending과 matched_merchants는 반드시 이 데이터를 기준으로 작성해라.
{benefit_usage}

[월별 절감 내역]
{monthly}

[계산 근거]
{notes}

[규칙]
1. benefit_details: breakdown 항목 수와 동일하게 (생략 금지)
2. my_spending: [혜택별 실제 적용 거래 집계]의 해당 benefit_id 적용 거래 합계와 가맹점별 전체 집계 기준
3. matched_merchants: [혜택별 실제 적용 거래 집계]의 해당 benefit_id 가맹점 전체를 빠짐없이 요약
4. estimated_saving: breakdown의 total_discount 값 그대로
5. discount_type amount의 discount_value는 원 단위 정액 혜택이다. 절대 퍼센트로 쓰지 마라.
6. total_summary: 연간 총 할인액과 연회비 차감 후 순절감액을 구분해서 작성, 월 평균은 총 할인액 기준

[출력 JSON]
{{
  "card_name":"","card_company":"","rank":{rank},
  "net_saving":{saving},"annual_fee":{fee},
  "summary":"1~2문장 핵심 요약",
  "benefit_details":[{{
    "benefit_name":"","condition":"",
    "matched_merchants":"","my_spending":"",
    "calculation":"","estimated_saving":0
  }}],
  "total_summary":"연간 총 할인액과 순절감액을 구분해서 작성, 월 평균 포함",
  "fee_recovery":"연회비 회수 기간 또는 순수 이득 명시",
  "cautions":null
}}"""

    recommendations = []
    for idx, card in enumerate(top5):
        status_text.text(f"[5/5] 추천 설명 생성 중... {idx+1}/5 ({card['card_name'][:20]})")
        progress_bar.progress(85 + idx * 3)

        merged_bd = merge_breakdown(card.get("benefit_breakdown", []))
        if not merged_bd:
            monthly_total = sum(m.get("discount", 0) for m in card.get("monthly_breakdown", []))
            merged_bd = [{
                "benefit_id": "b_all", "category": "전체 혜택", "merchant_type": "ALL",
                "total_applied_amount": card.get("total_discount", 0),
                "total_discount": monthly_total, "applied_count": 0,
            }]
        merged_bd = [b for b in merged_bd if b.get("total_discount", 0) > 0]
        benefit_usage = build_benefit_usage_summary(
            card["card_name"],
            merged_bd,
            structured_map,
            filtered_tx,
        )
        expected_detail_count = len(merged_bd)
        expected_detail_sum = round(sum(b.get("total_discount", 0) for b in merged_bd))

        fee      = round(card.get("annual_fee") or 0)
        saving   = round(card.get("net_saving", 0))
        discount = round(card.get("total_discount", 0))
        rank     = card.get("rank", idx + 1)

        user_msg = user_template.format(
            name=card["card_name"], company=card["card_company"],
            fee=fee, saving=saving, discount=discount, rank=rank,
            benefit_struct=get_benefit_structure(card["card_name"], structured_map),
            breakdown=json.dumps(merged_bd, ensure_ascii=False, indent=2),
            benefit_usage=benefit_usage,
            monthly=json.dumps(card.get("monthly_breakdown", []), ensure_ascii=False, indent=2),
            notes=card.get("calculation_notes", ""),
        )

        result = None
        for _ in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system},
                               {"role": "user",   "content": user_msg}],
                    temperature=0.3,
                    response_format={"type": "json_object"},
                )
                parsed = json.loads(extract_json_obj(resp.choices[0].message.content or ""))
                if isinstance(parsed.get("summary"), str) and parsed["summary"].strip():
                    details = parsed.get("benefit_details", [])
                    detail_sum = round(sum(d.get("estimated_saving", 0) for d in details))
                    bad_percent = any(
                        re.search(r"\b\d{3,}%", str(d.get("condition", "")))
                        or re.search(r"\b\d{3,}%", str(d.get("calculation", "")))
                        for d in details
                    )
                    no_spend_with_saving = any(
                        (d.get("estimated_saving", 0) or 0) > 0
                        and "모든 가맹점" not in str(d.get("matched_merchants", ""))
                        and (
                            "이용 내역 없음" in str(d.get("matched_merchants", ""))
                            or "이용 내역 없음" in str(d.get("my_spending", ""))
                        )
                        for d in details
                    )
                    if (
                        len(details) != expected_detail_count
                        or detail_sum != expected_detail_sum
                        or bad_percent
                        or no_spend_with_saving
                    ):
                        raise ValueError("추천 상세 검증 실패")
                    parsed["rank"] = rank
                    parsed["net_saving"] = saving
                    parsed["annual_fee"] = fee
                    parsed["total_summary"] = (
                        f"연간 총 할인액은 {expected_detail_sum:,}원이며, "
                        f"연회비 {fee:,}원을 차감한 순절감액은 {saving:,}원입니다. "
                        f"월 평균 총 할인액은 약 {expected_detail_sum // 12:,}원입니다."
                    )
                    result = parsed
                    break
            except Exception:
                pass
            time.sleep(1)

        recommendations.append(
            result
            or make_failed_recommendation(card, rank, card["card_name"], card["card_company"])
        )
        time.sleep(1)

    return recommendations

# ── 고정 광고 카드 데이터 ─────────────────────────────────────
AD_CARDS = [
    {
        "card_company": "삼성카드",
        "card_name": "삼성 iD VITA",
        "annual_fee": 20000,
        "main_categories": ["병원", "약국", "건강", "헬스", "피트니스"],
        "benefits": [
            "병원 이용 혜택",
            "약국 이용 혜택",
            "헬스·피트니스 관련 혜택",
        ],
        "ad_text": "건강 관련 소비가 많은 고객님께 추천하는 카드입니다.",
        "link": "https://www.card-gorilla.com/card/detail/2534"
    },
    {
        "card_company": "하나카드",
        "card_name": "하나 MULTI Any 체크카드",
        "annual_fee": 0,
        "main_categories": ["카페", "커피", "편의점", "교통", "대중교통", "간편결제", "생활"],
        "benefits": [
            "카페·커피 이용 혜택",
            "편의점·생활 소비 혜택",
            "대중교통·간편결제 혜택",
        ],
        "ad_text": "연회비 부담 없이 생활 소비 혜택을 받고 싶은 고객님께 추천하는 체크카드입니다.",
        "link": "https://www.card-gorilla.com/card/detail/2643"
    },
    {
        "card_company": "현대카드",
        "card_name": "현대카드O",
        "annual_fee": 20000,
        "main_categories": ["주유", "자동차", "차량", "정기결제", "구독"],
        "benefits": [
            "주유소 이용 혜택",
            "정기결제 이용 혜택",
            "차량 관련 소비 혜택",
        ],
        "ad_text": "주유, 정기결제, 차량 관련 소비가 많은 고객님께 추천하는 카드입니다.",
        "link": "https://www.card-gorilla.com/card/detail/2880"
    },
]


MERCHANT_CATEGORY_MAP = {
    "스타벅스": "카페",
    "메가커피": "카페",
    "이디야": "카페",
    "투썸플레이스": "카페",
    "컴포즈커피": "카페",
    "빽다방": "카페",

    "CU": "편의점",
    "GS25": "편의점",
    "세븐일레븐": "편의점",
    "이마트24": "편의점",

    "버스": "교통",
    "지하철": "교통",
    "택시": "교통",
    "코레일": "교통",

    "카카오페이": "간편결제",
    "네이버페이": "간편결제",
    "토스페이": "간편결제",
    "삼성페이": "간편결제",

    "병원": "병원",
    "의원": "병원",
    "약국": "약국",
    "올리브영": "건강",
    "헬스": "헬스",
    "피트니스": "피트니스",

    "SK주유소": "주유",
    "GS칼텍스": "주유",
    "S-OIL": "주유",
    "현대오일뱅크": "주유",
    "주유": "주유",

    "넷플릭스": "구독",
    "유튜브": "구독",
    "멜론": "구독",
    "쿠팡와우": "구독",
}


def get_ad_categories_from_transactions(transactions: list[dict]) -> list[str]:
    categories = []

    for t in transactions:
        merchant = str(
            t.get("matched_merchant")
            or t.get("merchant_name")
            or t.get("original_merchant")
            or ""
        )

        for keyword, category in MERCHANT_CATEGORY_MAP.items():
            if keyword in merchant:
                categories.append(category)

    return categories


def select_best_ad_card(transactions: list[dict]):
    user_categories = get_ad_categories_from_transactions(transactions)

    best_card = AD_CARDS[1]  # 기본값: 하나 MULTI Any 체크카드
    best_score = -1

    for card in AD_CARDS:
        score = sum(
            user_categories.count(category)
            for category in card["main_categories"]
        )

        if score > best_score:
            best_score = score
            best_card = card

    return best_card


def render_ad_banner(transactions: list[dict]) -> None:
    ad_card = select_best_ad_card(transactions)

    st.markdown('<div class="section-hd"><span class="dot"></span>📢 회원님께 추천하는 특별 카드</div>', unsafe_allow_html=True)
    st.caption("본 내용은 유료 광고를 포함하고 있습니다.")

    st.markdown(
        f"""
        <div class="ad-panel">
            <h3>{html.escape(ad_card['card_company'])} — {html.escape(ad_card['card_name'])}</h3>
            <p style="color:var(--muted);font-size:0.88rem;line-height:1.6;margin:0 0 0.75rem 0;">
                {html.escape(ad_card['ad_text'])}
            </p>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.6rem;margin-bottom:0.75rem;">
                <div class="cc-metric">
                    <div class="ml">연회비</div>
                    <div class="mv">{_fmt_ui(ad_card['annual_fee'])}원</div>
                </div>
            </div>
            {''.join(f'<div class="ad-benefit-item">✔ {html.escape(b)}</div>' for b in ad_card['benefits'])}
            <div class="ad-disclaimer">본 내용은 유료 광고를 포함하고 있습니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.link_button("자세히 보기", ad_card["link"])


# ── 결과 렌더링 (공통) ────────────────────────────────────────
def render_results(recs: list[dict], top5: list[dict], tx: list[dict]) -> None:
    stats: dict = defaultdict(lambda: {"count": 0, "total": 0})
    for t in tx:
        m = t.get("matched_merchant", "")
        if m:
            stats[m]["count"] += 1
            stats[m]["total"] += _parse_amount(t.get("amount", 0))

    cats = _ui_category_stats(tx)
    top1 = next((r for r in recs if r.get("rank") == 1), recs[0] if recs else {})
    top1_discount = next((c.get("total_discount", 0) for c in top5 if c.get("rank") == 1), 0)
    total_spend = sum(s["total"] for s in stats.values())
    top_merchants = sorted(stats.items(), key=lambda x: -x[1]["total"])[:10]
    top_merchant_tuples = [(m, d["total"]) for m, d in top_merchants]

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card highlight">
            <div class="label">1위 순 절감액</div>
            <div class="value">{_fmt_ui(top1.get("net_saving", 0))}원</div>
            <div class="sub">월 평균 {_fmt_ui(top1_discount // 12)}원</div>
        </div>
        <div class="kpi-card">
            <div class="label">매칭 거래</div>
            <div class="value">{_fmt_ui(len(tx))}건</div>
            <div class="sub">{_fmt_ui(len(stats))}개 가맹점</div>
        </div>
        <div class="kpi-card">
            <div class="label">연간 대상 소비</div>
            <div class="value">{_fmt_ui(total_spend)}원</div>
            <div class="sub">혜택 적용 가능</div>
        </div>
        <div class="kpi-card">
            <div class="label">추천 카드 수</div>
            <div class="value">Top {len(recs)}</div>
            <div class="sub">시뮬레이션 완료</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_spend, tab_cards, tab_detail = st.tabs([
        "📊 소비 분석", "🏆 카드 추천", "📋 상세 비교",
    ])

    with tab_spend:
        if stats:
            col_l, col_r = st.columns([3, 2])
            with col_l:
                st.markdown(
                    _ui_hbar_chart(top_merchant_tuples, "Top 10 가맹점 연간 지출"),
                    unsafe_allow_html=True,
                )
            with col_r:
                st.markdown(_ui_category_chips(cats), unsafe_allow_html=True)
        else:
            st.info("매칭된 거래 데이터가 없습니다.")

        st.markdown(
            '<div class="section-hd"><span class="dot"></span>월별 할인 추이 (1위 카드)</div>',
            unsafe_allow_html=True,
        )
        top1_data = next((c for c in top5 if c.get("rank") == 1), None)
        if top1_data and top1_data.get("monthly_breakdown"):
            mb = top1_data["monthly_breakdown"]
            df_monthly = pd.DataFrame({
                "월": [m["month"] for m in mb],
                "월별 할인(원)": [m["discount"] for m in mb],
            })
            st.line_chart(df_monthly.set_index("월")["월별 할인(원)"], color="#1e3a5f")

    with tab_cards:
        col_card, col_rest = st.columns([3, 2])
        with col_card:
            st.markdown(
                '<div class="section-hd"><span class="dot"></span>1위 추천 카드</div>',
                unsafe_allow_html=True,
            )
            st.markdown(_ui_premium_card(top1, top1_discount), unsafe_allow_html=True)

            show_total_box = not (
                PREVIEW_MODE and str(top1.get("summary", "")).strip() == "추천 설명 생성 실패"
            )
            if show_total_box and (total_sum := top1.get("total_summary", "")):
                st.markdown(
                    f'<div class="accent-box">💰 {_highlight_saving_keywords(total_sum)}</div>',
                    unsafe_allow_html=True,
                )
            if fee_rec := top1.get("fee_recovery", ""):
                st.markdown(
                    f'<div class="accent-box">📅 {_highlight_saving_keywords(fee_rec)}</div>',
                    unsafe_allow_html=True,
                )
            if not PREVIEW_MODE and (cautions := top1.get("cautions", "")):
                st.warning(f"⚠️ {cautions}")

            if top1.get("benefit_details"):
                with st.expander("🎁 혜택별 상세 분석 (1위)", expanded=False):
                    _render_card_benefit_expanders(top1)

        with col_rest:
            st.markdown(
                '<div class="section-hd"><span class="dot"></span>2~5위 카드</div>',
                unsafe_allow_html=True,
            )
            st.markdown(_ui_compact_cards(recs, top5), unsafe_allow_html=True)
            st.markdown(_ui_compare_bars(top5), unsafe_allow_html=True)

        for rec in recs:
            rank = rec.get("rank", 0)
            if rank <= 1:
                continue
            emoji = _RANK_EMOJI.get(rank, "💳")
            with st.expander(
                f"{emoji} {rank}위 {rec.get('card_company', '')} — {rec.get('card_name', '')} 상세",
                expanded=False,
            ):
                summary = _display_summary(rec)
                if summary:
                    st.markdown(
                        f'<div class="summary-box">{_highlight_saving_keywords(summary)}</div>',
                        unsafe_allow_html=True,
                    )
                _render_card_benefit_expanders(rec)
                show_total_box = not (
                    PREVIEW_MODE and str(rec.get("summary", "")).strip() == "추천 설명 생성 실패"
                )
                if show_total_box and (total_sum := rec.get("total_summary", "")):
                    st.markdown(
                        f'<div class="accent-box">💰 {_highlight_saving_keywords(total_sum)}</div>',
                        unsafe_allow_html=True,
                    )
                if fee_rec := rec.get("fee_recovery", ""):
                    st.markdown(
                        f'<div class="accent-box">📅 {_highlight_saving_keywords(fee_rec)}</div>',
                        unsafe_allow_html=True,
                    )
                if not PREVIEW_MODE and (cautions := rec.get("cautions", "")):
                    st.warning(f"⚠️ {cautions}")

    with tab_detail:
        st.markdown(_ui_compare_bars(top5), unsafe_allow_html=True)
        rows = [
            {
                "순위": r.get("rank", i + 1),
                "카드사": r.get("card_company", ""),
                "카드명": r.get("card_name", ""),
                "총할인(원)": r.get("total_discount", 0),
                "연회비(원)": r.get("annual_fee", 0),
                "순절감(원)": r.get("net_saving", 0),
            }
            for i, r in enumerate(top5)
        ]
        if rows:
            st.dataframe(
                pd.DataFrame(rows).style.format({
                    "총할인(원)": "{:,.0f}",
                    "연회비(원)": "{:,.0f}",
                    "순절감(원)": "{:,.0f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown(
            '<div class="section-hd"><span class="dot"></span>카드별 월별 할인 비교</div>',
            unsafe_allow_html=True,
        )
        monthly_frames = []
        for card in top5:
            mb = card.get("monthly_breakdown", [])
            if mb:
                monthly_frames.append(pd.DataFrame({
                    "월": [m["month"] for m in mb],
                    card.get("card_name", f"rank{card.get('rank')}"): [m["discount"] for m in mb],
                }).set_index("월"))
        if monthly_frames:
            combined = monthly_frames[0]
            for df_m in monthly_frames[1:]:
                combined = combined.join(df_m, how="outer")
            st.line_chart(combined)


def _load_demo_results() -> tuple[list, list, list]:
    rec_path = DEBUG_OUTPUT_DIR / "recommendation.json"
    top5_path = DEBUG_OUTPUT_DIR / "top5_cards.json"
    tx_path = DEBUG_OUTPUT_DIR / "filtered_transactions.json"
    for path in (rec_path, top5_path, tx_path):
        if not path.exists():
            st.error(f"결과 파일 `{path.name}`을 찾을 수 없습니다.")
            st.stop()
    with open(rec_path, encoding="utf-8") as f:
        recommendations = json.load(f)["recommendations"]
    with open(top5_path, encoding="utf-8") as f:
        top5 = json.load(f)["top5"]
    with open(tx_path, encoding="utf-8") as f:
        filtered_tx = json.load(f)["transactions"]
    return recommendations, top5, filtered_tx


# ── UI 진입점 ────────────────────────────────────────────────

st.set_page_config(page_title="AI 맞춤 카드 추천", page_icon="💳", layout="wide")
_inject_styles()

# 사이드바
with st.sidebar:
    st.markdown("## ⚙️ 설정")
    st.markdown("---")
    st.markdown("**LLM 모델**")
    match_model = st.text_input("거래 매칭 모델", value=DEFAULT_MATCH_MODEL)
    sim_model = st.text_input("절감액 시뮬레이션 모델", value=DEFAULT_SIM_MODEL)
    rec_model = st.text_input("추천 문장 생성 모델", value=DEFAULT_REC_MODEL)

# 헤더
st.markdown("""
<div class="app-hero">
    <h1>💳 AI 맞춤 신용카드 추천</h1>
    <p>12개월 카드 이용 내역을 업로드하면, 소비 패턴에 맞는 최적의 카드를 추천해 드립니다.</p>
    <div class="accent-tags">
        <span class="accent-tag">📊 소비 패턴 분석</span>
        <span class="accent-tag">🔍 100개 카드 시뮬레이션</span>
        <span class="accent-tag">💰 실제 절감액 계산</span>
        <span class="accent-tag">🏆 Top 5 추천</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── CSV 업로드 → 파이프라인 ──────────────────────────────────
uploaded = st.file_uploader(
    "카드 이용 내역 CSV 업로드 (컬럼: date, merchant_name, amount)",
    type=["csv"],
    help="date, merchant_name, amount 컬럼이 있는 CSV를 업로드하세요.",
)

if not uploaded:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">📂</div>
        <p><strong>CSV 파일을 업로드해 주세요.</strong></p>
        <p style="color:var(--muted);font-size:0.88rem;margin-top:0.5rem;">
            date, merchant_name, amount 컬럼이 있는 CSV를 업로드하면 분석이 시작됩니다.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

try:
    df = pd.read_csv(uploaded, encoding="utf-8-sig")
    if not {"date", "merchant_name", "amount"}.issubset(df.columns):
        st.error("CSV 컬럼이 date, merchant_name, amount 여야 합니다.")
        st.stop()
except Exception as e:
    st.error(f"파일 읽기 오류: {e}")
    st.stop()

upload_sig = f"{uploaded.name}:{uploaded.size}"
if st.session_state.get("upload_sig") != upload_sig:
    for key in ("recommendations", "top5", "filtered_tx", "model_config"):
        st.session_state.pop(key, None)
    st.session_state["upload_sig"] = upload_sig

parsed_amounts = df["amount"].apply(_parse_amount)

with st.expander("📋 업로드 데이터 미리보기", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("총 거래 건수",    f"{len(df):,}건")
    c2.metric("연간 총 결제금액", f"{parsed_amounts.sum():,}원")

if st.button("🔍 카드 추천 받기", type="primary", use_container_width=True):
    if PREVIEW_MODE:
        progress_bar = st.progress(0)
        status_text = st.empty()
        _run_demo_analysis(progress_bar, status_text)
        recommendations, top5, filtered_tx = _load_demo_results()
        st.session_state["recommendations"] = recommendations
        st.session_state["top5"] = top5
        st.session_state["filtered_tx"] = filtered_tx
        st.session_state["model_config"] = {
            "match": match_model,
            "simulation": sim_model,
            "recommendation": rec_model,
        }
    elif not STRUCTURED_JSON.exists() or not MERCHANT_JSON.exists():
        st.error("output/ 디렉터리에 사전 처리 파일이 없습니다.")
        st.stop()
    else:
        with open(STRUCTURED_JSON, encoding="utf-8") as f:
            structured: list = json.load(f)
        with open(MERCHANT_JSON, encoding="utf-8") as f:
            gms: dict = json.load(f)
        merchants:      list[str] = gms["merchants"]
        structured_map: dict      = {c["card_name"]: c for c in structured}

        client       = get_client()
        transactions = _prepare_transactions(df)

        progress_bar = st.progress(0)
        status_text  = st.empty()

        status_text.text("[1/5] 거래내역 매칭 중...")
        filtered_tx = run_fuzzy_matching(
            client, transactions, merchants, match_model, progress_bar, status_text
        )
        progress_bar.progress(20)

        if not filtered_tx:
            st.error(
                "거래 매칭 결과가 없습니다. 사이드바의 LLM 모델 설정과 OPENAI_API_KEY를 확인한 뒤 "
                "**카드 추천 받기** 버튼을 다시 눌러 주세요."
            )
            st.stop()

        if DEBUG_MODE:
            DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            with open(DEBUG_OUTPUT_DIR / "filtered_transactions.json", "w", encoding="utf-8") as f:
                json.dump({
                    "saved_at": str(date.today()),
                    "total_transactions": len(transactions),
                    "matched_count": len(filtered_tx),
                    "transactions": filtered_tx,
                }, f, ensure_ascii=False, indent=2)

        status_text.text("[2/5] 카드별 거래 매핑 중...")
        card_map = map_cards(filtered_tx, structured)
        progress_bar.progress(25)

        status_text.text("[3/5] 100개 카드 시뮬레이션 중...")
        sim_results = run_simulation(
            client, structured, card_map, sim_model, progress_bar, status_text
        )
        progress_bar.progress(80)

        if DEBUG_MODE:
            with open(DEBUG_OUTPUT_DIR / "simulation_results.json", "w", encoding="utf-8") as f:
                json.dump({
                    "saved_at": str(date.today()),
                    "total_cards": len(sim_results),
                    "results": sim_results,
                }, f, ensure_ascii=False, indent=2)

        status_text.text("[4/5] Top 5 선정 중...")
        top5 = extract_top5(sim_results)
        progress_bar.progress(83)

        if DEBUG_MODE:
            with open(DEBUG_OUTPUT_DIR / "top5_cards.json", "w", encoding="utf-8") as f:
                json.dump({
                    "saved_at": str(date.today()),
                    "top5": top5,
                }, f, ensure_ascii=False, indent=2)

        recommendations = generate_recommendations(
            client, top5, filtered_tx, structured_map, rec_model, progress_bar, status_text
        )
        progress_bar.progress(100)
        status_text.text("✅ 분석 완료!")

        if DEBUG_MODE:
            with open(DEBUG_OUTPUT_DIR / "recommendation.json", "w", encoding="utf-8") as f:
                json.dump({
                    "saved_at": str(date.today()),
                    "recommendations": recommendations,
                }, f, ensure_ascii=False, indent=2)

        st.session_state["recommendations"] = recommendations
        st.session_state["top5"]            = top5
        st.session_state["filtered_tx"]     = filtered_tx
        st.session_state["model_config"]    = {
            "match": match_model,
            "simulation": sim_model,
            "recommendation": rec_model,
        }

if "recommendations" not in st.session_state:
    st.stop()

st.success("분석이 완료되었습니다! 아래에서 결과를 확인하세요.")
render_results(
    st.session_state["recommendations"],
    st.session_state["top5"],
    st.session_state.get("filtered_tx", []),
)

render_ad_banner(
    st.session_state.get("filtered_tx", [])
)