"""
Streamlit UI (v2)
v2 변경점:
  - 절감액·순위는 calculator.py가 결정론적으로 계산 (LLM 산수 오류 차단)
  - LLM은 자연어 설명(reason / key_benefits / overall_summary)만 담당

result 구조:
    result["spending_profile"]    → 소비 패턴 dict
    result["recommendations"]     → 추천 카드 리스트 (계산 결과 + LLM 설명)
    result["overall_summary"]     → 전체 요약 문자열
    result["rag_candidates"]      → RAG 후보 카드 리스트
    result["parsed_candidates"]   → 파싱된 카드 구조 (디버깅용)
"""
import streamlit as st
import pandas as pd

import recommender
from openai_config import format_api_error, reload_env, resolve_api_key

# ── 고정 광고 카드 후보 3개 ─────────────────────────────
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
        "ad_text": "건강 관련 소비가 많은 고객님께 추천하는 카드이다.",
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
        "ad_text": "연회비 부담 없이 생활 소비 혜택을 받고 싶은 고객님께 추천하는 체크카드이다.",
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
        "ad_text": "주유, 정기결제, 차량 관련 소비가 많은 고객님께 추천하는 카드이다.",
        "link": "https://www.card-gorilla.com/card/detail/2880"
    },
]

# 순위별 이모지 (표시용)
_RANK_EMOJI = {1: "🥇", 2: "🥈", 3: "🥉", 4: "4️⃣", 5: "5️⃣"}
_DEFAULT_RANK_EMOJI = "💳"


def select_best_ad_card(spending_profile):
    categories = spending_profile["categories"]

    best_card = None
    best_score = -1

    for card in AD_CARDS:
        score = 0

        for user_category, data in categories.items():
            for ad_category in card["main_categories"]:
                if ad_category in user_category or user_category in ad_category:
                    score += data["ratio"]

        if score > best_score:
            best_score = score
            best_card = card

    return best_card, best_score


def _inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Noto Sans KR', sans-serif;
        }

        .block-container {
            padding-top: 1.5rem;
            max-width: 1100px;
        }

        [data-testid="stSidebar"] {
            background: #0f172a;
        }
        [data-testid="stSidebar"] * {
            color: #f8fafc !important;
        }
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stSlider label {
            font-weight: 600;
            font-size: 0.95rem;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background-color: rgba(255,255,255,0.12) !important;
            border-color: rgba(255,255,255,0.25) !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
            color: #94a3b8 !important;
            font-size: 1.1rem;
            letter-spacing: -0.02em;
        }

        .app-hero {
            background: #0f172a;
            border-radius: 20px;
            padding: 2rem 2.25rem;
            margin-bottom: 1.75rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.35);
            color: white;
        }
        .app-hero h1 {
            margin: 0 0 0.5rem 0;
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.03em;
        }
        .app-hero p {
            margin: 0;
            opacity: 0.95;
            font-size: 1.05rem;
        }
        .hero-tags {
            margin-top: 1rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        .hero-tag {
            background: rgba(255,255,255,0.15);
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 999px;
            padding: 0.25rem 0.75rem;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .section-label {
            display: inline-block;
            background: #0f172a;
            color: white;
            font-weight: 700;
            font-size: 0.85rem;
            padding: 0.45rem 0.9rem;
            border-radius: 4px;
            margin-bottom: 0.75rem;
        }

        .empty-state {
            text-align: center;
            padding: 2.5rem 1rem;
            color: #64748b;
        }
        .empty-state .icon {
            font-size: 3rem;
            margin-bottom: 0.5rem;
        }

        .rec-header {
            background: #e8eef4;
            border-left: 4px solid #1e3a5f;
            border-radius: 4px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.75rem;
        }
        .rec-header h3 {
            margin: 0 0 0.35rem 0;
            font-size: 1.25rem;
            color: #0f172a;
        }
        .rec-badge {
            display: inline-block;
            font-size: 0.78rem;
            font-weight: 700;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            margin-left: 0.35rem;
            vertical-align: middle;
        }
        .badge-ok {
            background: #fef9c3;
            color: #92400e;
            border: 1px solid #fde047;
        }
        .badge-no {
            background: #fee2e2;
            color: #991b1b;
        }

        .summary-box {
            background: #e8eef4;
            border-left: 4px solid #1e3a5f;
            border-radius: 12px;
            padding: 1rem 1.25rem;
            margin-bottom: 1.25rem;
            color: #1e293b;
            line-height: 1.6;
        }

        .benefit-chip {
            display: inline-block;
            background: #fef9c3;
            color: #92400e;
            border: 1px solid #fde047;
            border-radius: 8px;
            padding: 0.35rem 0.65rem;
            margin: 0.2rem 0.35rem 0.2rem 0;
            font-size: 0.88rem;
        }

        .ad-banner {
            background: #1e293b;
            border-radius: 16px;
            padding: 1.5rem;
            color: #f8fafc;
            margin-top: 0.5rem;
        }
        .ad-banner h3 { color: #fde68a !important; margin-top: 0; }
        .ad-label {
            display: inline-block;
            background: #fbbf24;
            color: #78350f;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            margin-bottom: 0.5rem;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 0.65rem 0.85rem;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }

        /* metric delta — 월 평균 XX원 절감 등 */
        [data-testid="stMetricDelta"] {
            color: #a16207 !important;
        }
        [data-testid="stMetricDelta"] > div {
            background-color: #fef9c3 !important;
            color: #92400e !important;
            border: 1px solid #fde047;
            border-radius: 6px;
            padding: 0.1rem 0.4rem;
        }
        [data-testid="stMetricDelta"] svg {
            fill: #ca8a04 !important;
            stroke: #ca8a04 !important;
        }

        /* 분석 완료 등 success 알림 (오류 알림은 제외) */
        div[data-testid="stAlert"]:has([data-testid="stIconMaterialCheck"]) {
            background-color: #fef9c3 !important;
            color: #92400e !important;
            border: 1px solid #fde047 !important;
        }
        div[data-testid="stAlert"]:has([data-testid="stIconMaterialCheck"]) svg {
            fill: #ca8a04 !important;
        }

        .stButton > button[kind="primary"] {
            background: #0f172a !important;
            border: none !important;
            font-weight: 700 !important;
            border-radius: 12px !important;
            padding: 0.65rem 1.25rem !important;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.35) !important;
        }
        .stButton > button[kind="primary"]:hover {
            background: #1e293b !important;
            box-shadow: 0 6px 18px rgba(30, 41, 59, 0.45) !important;
        }

        /* 결과 텍스트 복사 허용 */
        .block-container, [data-testid="stMarkdownContainer"] {
            user-select: text !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _inject_copy_helper():
    """Streamlit 단축키 C(캐시 삭제)와 텍스트 선택 충돌 완화."""
    st.components.v1.html(
        """
        <script>
        (function () {
          const block = (e) => {
            if ((e.key === "c" || e.key === "C") && !e.ctrlKey && !e.metaKey && !e.altKey) {
              const sel = window.getSelection();
              if (sel && sel.toString().length > 0) {
                e.stopImmediatePropagation();
              }
            }
          };
          window.addEventListener("keydown", block, true);
          const doc = window.parent.document;
          doc.addEventListener("keydown", block, true);
        })();
        </script>
        """,
        height=0,
    )


def _rank_emoji(rank: int) -> str:
    return _RANK_EMOJI.get(rank, _DEFAULT_RANK_EMOJI)


# ── 페이지 설정 ──────────────────────────────────────────────────
st.set_page_config(
    page_title="AI 맞춤 카드 추천",
    page_icon="💳",
    layout="wide",
)

reload_env()
_inject_styles()
_inject_copy_helper()

# ── 사이드바 ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 분석 설정")
    st.markdown(
        "<p style='opacity:0.85; font-size:0.9rem; margin-top:-0.5rem;'>"
        "모델과 추천 개수를 선택한 뒤 메인 화면에서 CSV를 업로드한다."
        "</p>",
        unsafe_allow_html=True,
    )
    gpt_model = st.selectbox("GPT 모델", ["gpt-4o-mini", "gpt-4o"], index=0)
    top_k = st.slider("추천 카드 수", min_value=1, max_value=5, value=5)

# ── 메인 ─────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-hero">
        <h1>💳 AI 맞춤 신용카드 추천</h1>
        <p>1년치 카드 이용 내역을 업로드하면, 소비 패턴에 맞는 최적의 카드를 추천한다.</p>
        <div class="hero-tags">
            <span class="hero-tag">📊 소비 패턴 분석</span>
            <span class="hero-tag">🔍 맞춤 카드 검색</span>
            <span class="hero-tag">💰 절감액 시뮬레이션</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "카드 이용 내역 CSV 업로드",
    type=["csv"],
    help="거래일시, 카드번호, 가맹점명, 카테고리, 결제금액, 승인번호 컬럼이 포함된 CSV",
)

if not uploaded:
    st.markdown(
        """
        <div class="empty-state">
            <div class="icon">📂</div>
            <p><strong>CSV 파일을 업로드해야 한다.</strong></p>
            <p style="font-size:0.9rem;">업로드 후 미리보기를 확인하고 「카드 추천 받기」를 누르면 분석이 시작된다.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if uploaded:
    df = pd.read_csv(uploaded)

    with st.expander("📋 업로드 데이터 미리보기", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("총 거래 건수", f"{len(df):,}건")
        c2.metric("연간 총 결제 금액", f"{df['결제금액'].sum():,}원")
        c3.metric("카테고리 수", f"{df['카테고리'].nunique()}개")

    if st.button("🔍 카드 추천 받기", type="primary", use_container_width=True):
        if not resolve_api_key():
            st.error(
                "OpenAI API 키가 설정되지 않았다. "
                "프로젝트 루트의 .env 파일에 OPENAI_API_KEY=sk-... 를 설정해야 한다."
            )
            st.stop()

        with st.spinner("소비 패턴을 분석하고 맞춤 카드를 찾는 중…"):
            try:
                result = recommender.run(
                    df,
                    top_recommend=top_k,
                    gpt_model=gpt_model,
                )
            except Exception as e:
                st.error(f"오류 발생: {format_api_error(e)}")
                st.stop()

        st.success("분석이 완료되었다. 아래에서 결과를 확인한다.")

        # ── 소비 패턴 ──────────────────────────────────────────
        st.markdown('<span class="section-label">📊 소비 패턴 분석</span>', unsafe_allow_html=True)

        sp = result["spending_profile"]
        col1, col2 = st.columns(2)
        col1.metric("연간 총 지출", f"{sp['total_annual']:,}원")
        col2.metric("월 평균 지출", f"{sp['monthly_avg']:,}원 / 월")

        cats = sp["categories"]

        chart_df = pd.DataFrame(
            [{"카테고리": k,
              "월 평균 지출(원)": v["monthly_avg"],
              "연간 합계(원)": v["total"],
              "비율(%)": v["ratio"]}
             for k, v in sorted(cats.items(), key=lambda x: x[1]["total"], reverse=True)]
        ).set_index("카테고리")

        st.caption("카테고리별 월 평균 지출 (해당 카테고리 이용 월 기준)")
        st.bar_chart(chart_df["월 평균 지출(원)"])

        with st.expander("카테고리별 상세 (월 평균 / 연간 합계)"):
            st.dataframe(
                chart_df.style.format({
                    "월 평균 지출(원)": "{:,.0f}",
                    "연간 합계(원)":    "{:,.0f}",
                    "비율(%)":         "{:.1f}",
                }),
                use_container_width=True,
            )

        st.divider()

        # ── 추천 결과 ──────────────────────────────────────────
        st.markdown('<span class="section-label">🏆 AI 추천 카드</span>', unsafe_allow_html=True)

        overall = result.get("overall_summary", "")
        if overall:
            st.markdown(f'<div class="summary-box">{overall}</div>', unsafe_allow_html=True)

        for rec in result.get("recommendations", []):
            rank = rec.get("rank", 0)
            emoji = _rank_emoji(rank)
            achievable = rec.get("monthly_req_achievable", True)
            badge = "✅ 전월실적 달성 가능" if achievable else "❌ 전월실적 달성 불가"
            badge_class = "badge-ok" if achievable else "badge-no"

            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="rec-header">
                        <h3>
                            {emoji} {rank}위 &nbsp; {rec['card_company']} — {rec['card_name']}
                            <span class="rec-badge {badge_class}">{badge}</span>
                        </h3>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                req_note = rec.get("monthly_req_note", "")
                if req_note:
                    st.caption(f"💡 {req_note}")

                m1, m2, m3 = st.columns(3)
                m1.metric(
                    "연회비 (연간)",
                    f"{rec['annual_fee']:,}원 / 년",
                )
                m2.metric(
                    "예상 연간 절감액",
                    f"{rec.get('total_annual_savings', 0):,}원 / 년",
                    delta=f"월 평균 {rec.get('total_annual_savings', 0) // 12:,}원 절감",
                )
                m3.metric(
                    "순 혜택 (연간 절감 − 연회비)",
                    f"{rec.get('net_benefit', 0):,}원 / 년",
                )

                st.markdown("**✨ 추천 이유**")
                st.write(rec.get("reason", ""))

                st.markdown("**🎁 주요 혜택 (수치 기준)**")
                benefits_html = "".join(
                    f'<span class="benefit-chip">{b}</span>'
                    for b in rec.get("key_benefits", [])
                )
                if benefits_html:
                    st.markdown(benefits_html, unsafe_allow_html=True)

                breakdown = rec.get("savings_breakdown", [])
                if breakdown:
                    with st.expander("💰 카테고리별 월/연간 절감액 계산"):
                        rows = []
                        for b in breakdown:
                            spend = b.get("monthly_avg_spend", b.get("monthly_spend", 0))
                            rate = b.get("discount_rate", "-")
                            m_save = b.get("monthly_saving", 0)
                            a_save = b.get("annual_saving", 0)
                            formula = (
                                b.get("formula")
                                or f"월 평균 {spend:,}원 × {rate} = 월 {m_save:,}원 → 연간 {a_save:,}원"
                            )
                            rows.append({
                                "카테고리":           b.get("category", ""),
                                "월 평균 지출(원)":   spend,
                                "할인율":             rate,
                                "월 절감액(원)":      m_save,
                                "연간 절감액(원)":    a_save,
                                "계산식":             formula,
                            })
                        bd_df = pd.DataFrame(rows)
                        st.dataframe(
                            bd_df.style.format({
                                "월 평균 지출(원)":  "{:,.0f}",
                                "월 절감액(원)":     "{:,.0f}",
                                "연간 절감액(원)":   "{:,.0f}",
                            }),
                            use_container_width=True,
                            hide_index=True,
                        )

        st.divider()

        # ── 고정 광고 카드 배너 ─────────────────────────────────
        ad_card, ad_score = select_best_ad_card(sp)

        if ad_card:
            st.markdown('<span class="section-label">📢 특별 추천</span>', unsafe_allow_html=True)
            st.caption("본 내용은 유료 광고를 포함하고 있다.")

            st.markdown(
                f"""
                <div class="ad-banner">
                    <span class="ad-label">AD</span>
                    <h3>{ad_card['card_company']} — {ad_card['card_name']}</h3>
                    <p style="opacity:0.9; margin-bottom:0;">{ad_card['ad_text']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.container(border=True):
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.metric("연회비", f"{ad_card['annual_fee']:,}원 / 년")

                with col2:
                    st.markdown("**주요 혜택 TOP 3**")
                    for benefit in ad_card["benefits"]:
                        st.markdown(f"✔ {benefit}")

                st.link_button("보러가기 >", ad_card["link"])

                st.markdown(
                    """
                    <div style='text-align: right; color: #0f172a; font-size: 12px; margin-top: 8px;'>
                        본 내용은 유료 광고를 포함하고 있다.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── RAG 후보 ───────────────────────────────────────────
        with st.expander("🔎 RAG 후보 카드 전체 보기 (소비 카테고리 매칭 점수 순)"):
            st.caption("매칭 점수: 사용자 지출 카테고리와 카드 주요 카테고리의 소비 비율 가중 일치도 (높을수록 적합)")
            rows = []
            for c in result.get("rag_candidates", []):
                rows.append({
                    "카드사":            c.get("card_company", ""),
                    "카드명":            c.get("card_name", ""),
                    "연회비(원/년)":     int(c.get("annual_fee", 0)),
                    "전월실적조건(원/월)": int(c.get("previous_month_requirement", 0)),
                    "매칭 점수":         c.get("similarity_score", 0),
                })
            cand_df = pd.DataFrame(rows)
            st.dataframe(
                cand_df.style.format({
                    "연회비(원/년)":      "{:,.0f}",
                    "전월실적조건(원/월)": "{:,.0f}",
                    "매칭 점수":          "{:.1f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        # ── 파싱 결과 (v2 디버깅용) ─────────────────────────────
        with st.expander("🧪 파서 결과 보기 (raw_benefits → 카테고리:할인율 변환)"):
            st.caption(
                "benefit_parser가 추출한 카테고리-할인율 매핑 "
                "(outputs/card_parsed_benefits.json 우선). 비어 있으면 해당 카드 혜택 미매칭."
            )
            parsed_rows = []
            for p in result.get("parsed_candidates", []):
                cd = p.get("category_discounts", {})
                parsed_rows.append({
                    "카드사":      p.get("card_company", ""),
                    "카드명":      p.get("card_name", ""),
                    "혜택 매핑":   ", ".join(f"{k}({v}%)" for k, v in cd.items()) or "(파싱 실패)",
                    "혜택 개수":   len(cd),
                })
            st.dataframe(
                pd.DataFrame(parsed_rows),
                use_container_width=True,
                hide_index=True,
            )
