"""
UI 개선 예시 화면
실행: streamlit run ui_demo.py

기존 app.py 대비 개선 포인트:
  - 정보 밀도: KPI 컴팩트 배치, 2열 레이아웃
  - 차트: Top 10 가로 막대 + 카테고리 분포 + Top5 비교 + 월별 추이
  - 카드 UI: 1위 프리미엄 카드 비주얼, 2~5위 컴팩트 리스트
  - 혜택: 프로그레스 바로 비중 시각화
"""

from __future__ import annotations

import html
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
DEBUG_OUTPUT_DIR = BASE_DIR / "debug_output"

_RANK_EMOJI = {1: "🥇", 2: "🥈", 3: "🥉", 4: "4️⃣", 5: "5️⃣"}

MERCHANT_CATEGORY_MAP = {
    "쿠팡": "쇼핑", "무신사": "쇼핑", "네이버": "쇼핑", "11번가": "쇼핑",
    "스타벅스": "카페", "이디야": "카페", "메가커피": "카페",
    "배달의민족": "배달", "요기요": "배달", "쿠팡이츠": "배달",
    "이마트": "마트", "홈플러스": "마트", "GS25": "편의점", "CU": "편의점",
    "세븐일레븐": "편의점", "SK에너지": "주유", "GS칼텍스": "주유",
    "넷플릭스": "구독", "유튜브": "구독", "멜론": "구독", "티빙": "구독",
    "카카오T": "교통", "쏘카": "교통", "아고다": "여행",
}


def _fmt(n: int | float) -> str:
    return f"{int(n):,}"


def _highlight_saving(text: str) -> str:
    safe = html.escape(str(text))
    return re.sub(
        r"(월 평균 [\d,]+원(?: 절감)?)",
        r'<span class="kw-pill">\1</span>',
        safe,
    )


def _inject_demo_styles() -> None:
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
    .stApp, [data-testid="stAppViewContainer"], .main, .main .block-container {
        background: var(--bg) !important;
        color: var(--text) !important;
    }
    .block-container { max-width: 1280px; padding-top: 1rem; }

    /* ── 글래스 박스 공통 ── */
    .glass-box {
        background: var(--glass);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        box-shadow: var(--shadow);
    }

    /* ── Hero ── */
    .demo-hero {
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
    .demo-hero::before {
        content: '';
        position: absolute;
        top: -40%; right: -10%;
        width: 320px; height: 320px;
        background: radial-gradient(circle, var(--navy-glow) 0%, transparent 70%);
        pointer-events: none;
    }
    .demo-hero h1 {
        margin: 0 0 0.35rem;
        font-size: 1.65rem;
        font-weight: 700;
        color: var(--navy-dark);
        letter-spacing: -0.02em;
    }
    .demo-hero p { margin: 0; color: var(--muted); font-size: 0.92rem; }
    .demo-badge {
        display: inline-block;
        margin-top: 0.75rem;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(30, 58, 95, 0.08);
        border: 1px solid rgba(30, 58, 95, 0.22);
        color: var(--navy);
    }

    /* ── KPI row ── */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.75rem;
        margin-bottom: 1.25rem;
    }
    @media (max-width: 768px) {
        .kpi-grid { grid-template-columns: repeat(2, 1fr); }
    }
    .kpi-card {
        background: var(--glass);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.85rem 1rem;
        box-shadow: var(--shadow);
    }
    .kpi-card .label {
        font-size: 0.72rem;
        color: var(--muted);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .kpi-card .value {
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--text);
        margin-top: 0.15rem;
        line-height: 1.2;
    }
    .kpi-card .sub {
        font-size: 0.72rem;
        color: var(--green);
        margin-top: 0.2rem;
    }
    .kpi-card.highlight {
        background: linear-gradient(135deg, rgba(240,245,252,0.92), rgba(255,255,255,0.78));
        border-color: rgba(30, 58, 95, 0.22);
    }
    .kpi-card.highlight .value { color: var(--navy-dark); }

    /* ── Section header ── */
    .section-hd {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 1.5rem 0 0.85rem;
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text);
    }
    .section-hd .dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--navy);
    }

    /* ── Chart panel ── */
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
        margin: 0 0 0.85rem;
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--muted);
    }

    /* ── Horizontal bar chart (HTML) ── */
    .hbar-row {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.45rem;
        font-size: 0.78rem;
    }
    .hbar-label {
        width: 72px;
        text-align: right;
        color: var(--text);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        flex-shrink: 0;
    }
    .hbar-track {
        flex: 1;
        height: 18px;
        background: var(--surface-2);
        border-radius: 4px;
        overflow: hidden;
    }
    .hbar-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, rgba(30,58,95,0.45), var(--navy));
        transition: width 0.4s ease;
    }
    .hbar-val {
        width: 72px;
        text-align: right;
        color: var(--muted);
        font-size: 0.72rem;
        flex-shrink: 0;
    }

    /* ── Category chips ── */
    .cat-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .cat-chip {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.45rem 0.75rem;
        background: var(--glass-strong);
        border: 1px solid var(--border);
        border-radius: 10px;
        font-size: 0.78rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .cat-chip .pct {
        font-weight: 700;
        color: var(--navy-dark);
    }
    .cat-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    /* ── Premium credit card ── */
    .credit-card {
        background: linear-gradient(135deg, rgba(248,250,255,0.95) 0%, rgba(240,245,252,0.9) 50%, rgba(255,255,255,0.85) 100%);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(30, 58, 95, 0.22);
        border-radius: 18px;
        padding: 1.5rem 1.75rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(30, 58, 95, 0.1), var(--shadow);
    }
    .credit-card::after {
        content: '';
        position: absolute;
        top: -50%; right: -20%;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(30, 58, 95, 0.1), transparent 70%);
    }
    .cc-rank {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--navy);
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .cc-name {
        font-size: 1.45rem;
        font-weight: 700;
        color: var(--text);
        margin: 0.3rem 0 0.15rem;
        letter-spacing: -0.02em;
    }
    .cc-company {
        font-size: 0.85rem;
        color: var(--muted);
        margin-bottom: 1.25rem;
    }
    .cc-metrics {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.75rem;
    }
    .cc-metric {
        background: rgba(255, 255, 255, 0.65);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.65rem 0.8rem;
    }
    .cc-metric .ml { font-size: 0.68rem; color: var(--muted); }
    .cc-metric .mv { font-size: 1.05rem; font-weight: 700; color: var(--navy-dark); margin-top: 0.1rem; }
    .cc-metric.hero .mv { font-size: 1.25rem; color: var(--green); }
    .cc-summary {
        margin-top: 1rem;
        padding: 0.75rem 1rem;
        background: rgba(255, 255, 255, 0.55);
        border: 1px solid var(--border);
        border-radius: 10px;
        font-size: 0.85rem;
        line-height: 1.6;
        color: #444;
    }

    /* ── Benefit bars ── */
    .benefit-list { margin-top: 0.5rem; }
    .benefit-item {
        margin-bottom: 0.65rem;
    }
    .benefit-hd {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        margin-bottom: 0.25rem;
    }
    .benefit-hd .name { color: var(--text); font-weight: 500; }
    .benefit-hd .amt { color: var(--green); font-weight: 600; }
    .benefit-track {
        height: 6px;
        background: var(--surface-2);
        border-radius: 3px;
        overflow: hidden;
    }
    .benefit-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #3d9e56, var(--green));
    }

    /* ── Compact card list (2~5위) ── */
    .card-list-item {
        display: flex;
        align-items: center;
        gap: 1rem;
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
    .cli-rank {
        font-size: 1.3rem;
        width: 36px;
        text-align: center;
        flex-shrink: 0;
    }
    .cli-info { flex: 1; min-width: 0; }
    .cli-info .title {
        font-size: 0.88rem;
        font-weight: 600;
        color: var(--text);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .cli-info .fee { font-size: 0.72rem; color: var(--muted); margin-top: 0.1rem; }
    .cli-saving {
        text-align: right;
        flex-shrink: 0;
    }
    .cli-saving .val {
        font-size: 1rem;
        font-weight: 700;
        color: var(--green);
    }
    .cli-saving .lbl { font-size: 0.68rem; color: var(--muted); }

    .kw-pill {
        display: inline;
        background: var(--green-bg);
        color: var(--green);
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        font-weight: 600;
    }

    /* ── Compare bars (Top5) ── */
    .compare-row {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.55rem;
        font-size: 0.78rem;
    }
    .compare-label {
        width: 130px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        color: var(--text);
        flex-shrink: 0;
    }
    .compare-track {
        flex: 1;
        height: 22px;
        background: var(--surface-2);
        border-radius: 5px;
        overflow: hidden;
    }
    .compare-fill {
        height: 100%;
        border-radius: 5px;
    }
    .compare-val {
        width: 80px;
        text-align: right;
        color: var(--navy-dark);
        font-weight: 600;
        font-size: 0.75rem;
        flex-shrink: 0;
    }

    /* ── Sidebar note ── */
    .improve-note {
        font-size: 0.82rem;
        line-height: 1.65;
        color: var(--muted);
    }
    .improve-note li { margin-bottom: 0.35rem; }

    /* ── Streamlit 화이트 테마 오버라이드 ── */
    div[data-testid="stMetric"] { display: none; }

    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {
        background: #ffffff !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    label, .stCaption {
        color: var(--text) !important;
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--muted) !important;
    }

    [data-testid="stTabs"] button {
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: var(--muted) !important;
        background: transparent !important;
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
    details[data-testid="stExpander"] summary span {
        color: var(--text) !important;
    }

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
    [data-testid="stDataFrame"] * {
        color: var(--text) !important;
    }
    </style>
    """, unsafe_allow_html=True)


def _load_data() -> tuple[list, list, list]:
    rec_path = DEBUG_OUTPUT_DIR / "recommendation.json"
    top5_path = DEBUG_OUTPUT_DIR / "top5_cards.json"
    tx_path = DEBUG_OUTPUT_DIR / "filtered_transactions.json"
    for path in (rec_path, top5_path, tx_path):
        if not path.exists():
            st.error(f"`{path.name}` 파일이 없습니다. debug_output/ 폴더를 확인해주세요.")
            st.stop()
    with open(rec_path, encoding="utf-8") as f:
        recs = json.load(f)["recommendations"]
    with open(top5_path, encoding="utf-8") as f:
        top5 = json.load(f)["top5"]
    with open(tx_path, encoding="utf-8") as f:
        tx = json.load(f)["transactions"]
    return recs, top5, tx


def _merchant_stats(tx: list[dict]) -> dict[str, dict]:
    stats: dict = defaultdict(lambda: {"count": 0, "total": 0})
    for t in tx:
        m = t.get("matched_merchant", "")
        if m:
            stats[m]["count"] += 1
            stats[m]["total"] += int(t.get("amount", 0))
    return stats


def _category_stats(tx: list[dict]) -> dict[str, int]:
    cats: dict[str, int] = defaultdict(int)
    for t in tx:
        merchant = str(t.get("matched_merchant") or "")
        amount = int(t.get("amount", 0))
        matched_cat = "기타"
        for kw, cat in MERCHANT_CATEGORY_MAP.items():
            if kw in merchant:
                matched_cat = cat
                break
        cats[matched_cat] += amount
    return dict(cats)


_CAT_COLORS = {
    "쇼핑": "#1e3a5f", "카페": "#c97b5a", "배달": "#7ec86e",
    "마트": "#6ba3d6", "편의점": "#b07cc6", "주유": "#e8a838",
    "구독": "#5bc4b0", "교통": "#8a9bae", "여행": "#d47b9f", "기타": "#666",
}


def _render_hbar_chart(merchants: list[tuple[str, int]], title: str) -> str:
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
            f'<span class="hbar-val">{_fmt(val)}원</span></div>'
        )
    return f'<div class="chart-panel"><h4>{html.escape(title)}</h4>{"".join(rows)}</div>'


def _render_category_chips(cats: dict[str, int]) -> str:
    total = sum(cats.values()) or 1
    sorted_cats = sorted(cats.items(), key=lambda x: -x[1])
    chips = []
    for cat, val in sorted_cats:
        pct = val / total * 100
        color = _CAT_COLORS.get(cat, "#666")
        chips.append(
            f'<div class="cat-chip">'
            f'<span class="cat-dot" style="background:{color}"></span>'
            f'<span>{html.escape(cat)}</span>'
            f'<span class="pct">{pct:.0f}%</span>'
            f'<span style="color:var(--muted);font-size:0.72rem">{_fmt(val)}원</span>'
            f'</div>'
        )
    return f'<div class="chart-panel"><h4>카테고리별 소비 비중</h4><div class="cat-grid">{"".join(chips)}</div></div>'


def _render_compare_bars(top5: list[dict]) -> str:
    if not top5:
        return ""
    max_saving = max(c.get("net_saving", 0) for c in top5) or 1
    colors = ["#1e3a5f", "#2c4a7c", "#374151", "#4b5563", "#6b7280"]
    rows = []
    for i, card in enumerate(top5):
        saving = card.get("net_saving", 0)
        pct = saving / max_saving * 100
        label = f"{card.get('card_company','')} {card.get('card_name','')}"
        short = label if len(label) <= 16 else label[:15] + "…"
        rows.append(
            f'<div class="compare-row">'
            f'<span class="compare-label" title="{html.escape(label)}">'
            f'{_RANK_EMOJI.get(card.get("rank", i+1), "💳")} {html.escape(short)}</span>'
            f'<div class="compare-track">'
            f'<div class="compare-fill" style="width:{pct:.1f}%;background:{colors[i % len(colors)]}"></div>'
            f'</div>'
            f'<span class="compare-val">{_fmt(saving)}원</span></div>'
        )
    return f'<div class="chart-panel"><h4>Top 5 순 절감액 비교</h4>{"".join(rows)}</div>'


def _render_benefit_bars(details: list[dict]) -> str:
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
            f'<span class="amt">{_fmt(s)}원</span></div>'
            f'<div class="benefit-track">'
            f'<div class="benefit-fill" style="width:{pct:.1f}%"></div></div></div>'
        )
    return f'<div class="benefit-list">{"".join(items)}</div>'


def _render_premium_card(rec: dict, discount: int) -> str:
    saving = rec.get("net_saving", 0)
    fee = rec.get("annual_fee", 0)
    summary = _highlight_saving(rec.get("summary", ""))
    benefits_html = _render_benefit_bars(rec.get("benefit_details", []))
    return f"""
    <div class="credit-card">
        <div class="cc-rank">🥇 Best Recommendation</div>
        <div class="cc-name">{html.escape(rec.get("card_name", ""))}</div>
        <div class="cc-company">{html.escape(rec.get("card_company", ""))}</div>
        <div class="cc-metrics">
            <div class="cc-metric">
                <div class="ml">연회비</div>
                <div class="mv">{_fmt(fee)}원</div>
            </div>
            <div class="cc-metric">
                <div class="ml">연간 총 할인</div>
                <div class="mv">{_fmt(discount)}원</div>
            </div>
            <div class="cc-metric hero">
                <div class="ml">순 절감액</div>
                <div class="mv">{_fmt(saving)}원/년</div>
            </div>
        </div>
        <div class="cc-summary">{summary}</div>
    </div>
    {benefits_html}
    """


def _render_compact_cards(recs: list[dict], top5: list[dict]) -> str:
    items = []
    for rec in recs:
        rank = rec.get("rank", 0)
        if rank == 1:
            continue
        saving = rec.get("net_saving", 0)
        fee = rec.get("annual_fee", 0)
        emoji = _RANK_EMOJI.get(rank, "💳")
        items.append(
            f'<div class="card-list-item">'
            f'<div class="cli-rank">{emoji}</div>'
            f'<div class="cli-info">'
            f'<div class="title">{html.escape(rec.get("card_company",""))} — {html.escape(rec.get("card_name",""))}</div>'
            f'<div class="fee">연회비 {_fmt(fee)}원 · 총할인 {_fmt(next((c.get("total_discount",0) for c in top5 if c.get("rank")==rank),0))}원</div>'
            f'</div>'
            f'<div class="cli-saving">'
            f'<div class="val">{_fmt(saving)}원</div>'
            f'<div class="lbl">순 절감/년</div></div></div>'
        )
    return "".join(items)


def render_demo(recs: list[dict], top5: list[dict], tx: list[dict]) -> None:
    stats = _merchant_stats(tx)
    cats = _category_stats(tx)
    top1 = next((r for r in recs if r.get("rank") == 1), recs[0] if recs else {})
    top1_discount = next((c.get("total_discount", 0) for c in top5 if c.get("rank") == 1), 0)
    total_spend = sum(s["total"] for s in stats.values())

    top_merchants = sorted(stats.items(), key=lambda x: -x[1]["total"])[:10]
    top_merchant_tuples = [(m, d["total"]) for m, d in top_merchants]

    st.markdown("""
    <div class="demo-hero">
        <h1>💳 AI 맞춤 신용카드 추천</h1>
        <p>12개월 소비 데이터 기반 최적 카드 시뮬레이션 · UI 개선 예시 화면</p>
        <span class="demo-badge">✨ UI DEMO v2 — 기존 대비 레이아웃·차트·카드 비주얼 개선</span>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card highlight">
            <div class="label">1위 순 절감액</div>
            <div class="value">{_fmt(top1.get("net_saving", 0))}원</div>
            <div class="sub">월 평균 {_fmt(top1_discount // 12)}원</div>
        </div>
        <div class="kpi-card">
            <div class="label">매칭 거래</div>
            <div class="value">{_fmt(len(tx))}건</div>
            <div class="sub">{_fmt(len(stats))}개 가맹점</div>
        </div>
        <div class="kpi-card">
            <div class="label">연간 대상 소비</div>
            <div class="value">{_fmt(total_spend)}원</div>
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
        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.markdown(
                _render_hbar_chart(top_merchant_tuples, "Top 10 가맹점 연간 지출"),
                unsafe_allow_html=True,
            )
        with col_r:
            st.markdown(_render_category_chips(cats), unsafe_allow_html=True)

        st.markdown('<div class="section-hd"><span class="dot"></span>월별 할인 추이 (1위 카드)</div>', unsafe_allow_html=True)
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
            st.markdown('<div class="section-hd"><span class="dot"></span>1위 추천 카드</div>', unsafe_allow_html=True)
            st.markdown(_render_premium_card(top1, top1_discount), unsafe_allow_html=True)

            details = top1.get("benefit_details", [])
            if details:
                with st.expander("🎁 혜택별 상세 분석 (클릭하여 펼치기)", expanded=False):
                    for b in details:
                        saving_val = b.get("estimated_saving", 0)
                        matched = b.get("matched_merchants", "")
                        is_match = (
                            "이용하지 않으셨습니다" not in matched
                            and "이용 내역 없음" not in matched
                            and matched.strip() != ""
                        )
                        icon = "✅" if is_match else "ℹ️"
                        with st.expander(f"{icon} {b.get('benefit_name','')} — {_fmt(saving_val)}원", expanded=False):
                            st.markdown(f"**조건** {b.get('condition','')}")
                            st.markdown(f"**매칭 가맹점** {matched}")
                            st.markdown(f"**내 소비** {b.get('my_spending','')}")
                            st.code(b.get("calculation", ""), language=None)

        with col_rest:
            st.markdown('<div class="section-hd"><span class="dot"></span>2~5위 카드</div>', unsafe_allow_html=True)
            st.markdown(_render_compact_cards(recs, top5), unsafe_allow_html=True)
            st.markdown(_render_compare_bars(top5), unsafe_allow_html=True)

    with tab_detail:
        st.markdown(_render_compare_bars(top5), unsafe_allow_html=True)
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

        st.markdown('<div class="section-hd"><span class="dot"></span>카드별 월별 할인 비교</div>', unsafe_allow_html=True)
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
            for df in monthly_frames[1:]:
                combined = combined.join(df, how="outer")
            st.line_chart(combined)


def main() -> None:
    st.set_page_config(
        page_title="UI 개선 예시 — AI 카드 추천",
        page_icon="✨",
        layout="wide",
    )
    _inject_demo_styles()

    with st.sidebar:
        st.header("🎨 UI 개선 포인트")
        st.markdown("""
        <div class="improve-note">
        <ul>
        <li><b>정보 밀도</b> — KPI 4칸 컴팩트 배치, 탭으로 섹션 분리</li>
        <li><b>차트</b> — 78개 세로 막대 → Top 10 가로 막대</li>
        <li><b>카테고리</b> — 소비 비중 칩으로 한눈에 파악</li>
        <li><b>1위 카드</b> — 실물 카드 느낌의 프리미엄 비주얼</li>
        <li><b>2~5위</b> — 컴팩트 리스트 + 비교 막대</li>
        <li><b>혜택</b> — 프로그레스 바로 절감 비중 표시</li>
        <li><b>추이</b> — 월별 할인 라인 차트 추가</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        st.divider()
        st.caption("기존 화면: `streamlit run app.py`")
        st.caption("개선 예시: `streamlit run ui_demo.py`")

    recs, top5, tx = _load_data()
    render_demo(recs, top5, tx)


if __name__ == "__main__":
    main()
