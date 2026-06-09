# AI 맞춤 신용카드 추천 시스템

1년치 카드 이용 내역을 분석해 소비 패턴에 맞는 신용카드를 추천하고, 예상 절감액을 시뮬레이션하는 AI 추천 시스템입니다.

이 저장소에는 **서로 다른 접근 방식의 두 버전**이 공존합니다.

| 버전 | 폴더 | 접근 방식 | 실행 |
|------|------|-----------|------|
| **v2.1** | [`v2/`](v2/) | RAG + LLM 하이브리드, **카테고리** 기반 매칭 | `streamlit run v2/gpt/app.py` |
| **v3** | [`v3/`](v3/) | 가맹점 **퍼지매칭** + LLM 검산, 전체 100종 시뮬레이션 | `streamlit run v3/app.py` |

> v3는 v2를 점진적으로 개선한 것이 아니라, **파이프라인 전체를 재설계한 버전**입니다. 두 버전 모두 독립적으로 실행할 수 있습니다.

📄 **발표자료**: [소비패턴기반_AI_카드추천시스템_발표자료.pdf](소비패턴기반_AI_카드추천시스템_발표자료.pdf) — 프로젝트 개요, 파이프라인, 시연 결과를 정리한 PDF입니다.

---

## 버전별 요약

### v2.1 — RAG + LLM (카테고리 기반)

- CSV에 **카테고리 컬럼**이 포함된 6컬럼 형식
- 소비 비율 가중 **RAG 키워드 매칭**으로 Top-15 후보 선별
- 100종 카드 혜택 **사전 파싱 JSON** → 추천 시 즉시 조회
- 절감액·순위는 **Python 결정론적 계산**, LLM은 설명만 담당

자세한 내용: **[v2/README.md](v2/README.md)**

### v3 — 가맹점 매칭 (전면 재설계)

- CSV **3컬럼** (date, merchant_name, amount) — 영수증 가맹점명 그대로 업로드
- GPT **퍼지매칭**으로 가맹점명 정규화 → 카드별 혜택 매핑
- 100종 전체 **절감액 시뮬레이션** (LLM + Python 검산)
- Top 5 선정 후 GPT가 추천 설명 생성

자세한 내용: **[v3/README.md](v3/README.md)**

---

## 공통 설치

Python 3.10+ 권장. OpenAI API 키가 필요합니다.

```bash
# 1. 저장소 클론 후 루트에서 API 키 설정
copy .env.example .env          # Windows
# cp .env.example .env          # macOS / Linux

# .env 파일 편집
# OPENAI_API_KEY=your-api-key-here

# 2-A. v2 실행
cd v2
pip install -r requirements.txt
streamlit run gpt/app.py

# 2-B. v3 실행
cd v3
pip install -r requirements.txt
streamlit run app.py
```

**`.env` 파일은 git에 커밋하지 않습니다.**

---

## 저장소 구조

```
CardRecommendationSystem/
├── README.md              ← 이 파일 (버전 개요)
├── 소비패턴기반_AI_카드추천시스템_발표자료.pdf  ← 발표자료
├── .env.example           ← API 키 템플릿 (루트 공통)
├── .gitignore
│
├── v2/                    ← v2.1 RAG + LLM 버전
│   ├── gpt/               # Streamlit UI + 파이프라인
│   ├── rag/               # 소비 분석 + 후보 검색
│   ├── scripts/           # 혜택 사전 파싱
│   ├── outputs/           # 카드 DB + 파싱 JSON
│   ├── examples/          # 샘플 거래내역 (1,326건)
│   └── README.md
│
└── v3/                    ← v3 가맹점 매칭 버전
    ├── app.py             # Streamlit UI + 5단계 파이프라인
    ├── output/            # 카드 DB + 가맹점 목록
    ├── examples/          # 샘플 거래내역
    ├── ui_reference/      # UI 목업
    └── README.md
```

---

## 기술 스택

- **Language**: Python 3.10+
- **LLM**: OpenAI GPT-4o / GPT-4o-mini
- **UI**: Streamlit
- **Data**: Pandas, NumPy
- **Card DB**: 100종 국내 신용카드

---

## 보안 · 개인정보

- API 키는 **환경변수 또는 `.env` 파일**로만 설정합니다. 코드에 하드코딩하지 않습니다.
- 실제 거래내역 CSV, `.xlsx` 파일, `debug_output/` JSON은 `.gitignore`에 등록되어 커밋되지 않습니다.
- 테스트 시 각 버전의 `examples/` 샘플 CSV를 사용하세요.
