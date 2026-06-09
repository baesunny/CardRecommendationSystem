# v3 — AI 맞춤 신용카드 추천 시스템 (가맹점 매칭)

> 카드 거래내역 CSV를 올리면 GPT가 소비 패턴을 분석해서 **국내 100개 카드 중 연간 절감액 Top 5**를 추천해주는 Streamlit 앱

v2와 달리 **카테고리가 아닌 가맹점 단위**로 혜택을 매칭하고, LLM 퍼지매칭 + Python 검산을 결합한 파이프라인입니다.

> 프로젝트 발표자료: [../소비패턴기반_AI_카드추천시스템_발표자료.pdf](../소비패턴기반_AI_카드추천시스템_발표자료.pdf)

---

## 빠른 시작

Python 3.10 이상이 필요합니다. (`python3 --version` 으로 확인)

**1. 가상환경 생성 & 활성화** (권장)

```bash
cd v3
python -m venv venv
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate
```

**2. 라이브러리 설치**

```bash
pip install -r requirements.txt
```

**3. OpenAI API 키 설정**

```bash
copy ..\.env.example ..\.env   # Windows (저장소 루트)
# cp ../.env.example ../.env     # macOS / Linux
```

`.env` 파일에 API 키를 입력합니다. **`.env`는 git에 커밋하지 않습니다.**

```
OPENAI_API_KEY=your-api-key-here
```

또는 환경변수로 직접 설정할 수 있습니다.

```bash
# macOS / Linux
export OPENAI_API_KEY="your-api-key-here"

# Windows (PowerShell)
$env:OPENAI_API_KEY="your-api-key-here"
```

**4. 앱 실행**

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속 → CSV 업로드 → **카드 추천 받기** 클릭

> `output/` 폴더에 카드 DB가 이미 준비되어 있어, 크롤링·전처리는 다시 돌릴 필요가 없습니다.
> 테스트용 샘플 CSV: `examples/sample_transactions.csv`

---

## 전체 흐름

파이프라인은 **사전 준비(1회성, 이미 완료)** 와 **런타임(CSV 업로드마다 실행)** 두 단계로 나뉩니다.

```
━━━━━━━━━━━━━━ 사전 준비 (1회성, 이미 완료) ━━━━━━━━━━━━━━

  preprocess_benefits.py  →  structure_benefits.py
  혜택 텍스트 정제              GPT-4o로 구조화
                                                                  ↓
                                                  output/card_benefits_structured.json  ★ 카드 DB
                                                                  ↓
                                                  build_merchant_set.py
                                                  가맹점명 정규화 · 중복제거
                                                                  ↓
                                                  output/global_merchant_set.json


━━━━━━━━━━━━━ 런타임 (CSV 업로드마다 실행) ━━━━━━━━━━━━━

  사용자 CSV
      │
      ▼
  [1/5]  거래 매칭      영수증 가맹점명을 브랜드명으로 정규화 (GPT-4o, 50건씩 배치)
      │
      ▼
  [2/5]  카드 매핑      카드별로 혜택 받을 수 있는 거래만 분리
      │
      ▼
  [3/5]  시뮬레이션     카드 100개의 연간 최대 절감액 계산 (GPT-4o-mini)
      │                 LLM 결과를 코드로 검산 → 이상하면 보수값으로 자동 대체
      ▼
  [4/5]  Top 5 선정     순절감액(net_saving) 내림차순 상위 5개
      │
      ▼
  [5/5]  추천 설명      Top 5 카드별 상세 추천 문장 생성 (GPT-4o, 병렬)
      │
      ▼
  Streamlit 결과 화면
```

---

## 파일 구조

```
v3/
├── app.py                       ← 진입점. 이 파일만 실행하면 됨
├── requirements.txt
├── .env.example
│
├── preprocess_benefits.py       ← 혜택 텍스트 정제
├── structure_benefits.py        ← GPT-4o로 혜택 구조화
├── build_merchant_set.py        ← 전체 가맹점 집합 생성
│
├── filter_transactions.py       ← [1/5] 거래 퍼지매칭
├── map_card_transactions.py     ← [2/5] 카드별 거래 매핑
├── simulate_savings.py          ← [3/5] 절감액 시뮬레이션 + 검산
├── extract_top5.py              ← [4/5] Top 5 추출
├── generate_recommendation.py   ← [5/5] 추천 설명 생성
│
├── output/                          ← 고정 데이터 (수정 금지)
│   ├── card_benefits_cleaned.json     혜택 텍스트 정제 결과
│   ├── card_benefits_structured.json  ★ 카드 혜택 DB
│   └── global_merchant_set.json       ★ 가맹점 목록
│
├── examples/
│   └── sample_transactions.csv      ← 테스트용 익명 샘플
│
├── ui_reference/                    ← UI 목업 HTML
├── ui_demo.py
└── debug_output/                    ← DEBUG_MODE=True 일 때 단계별 결과 (git 제외)
    └── .gitkeep
```

### output/ 와 debug_output/ 의 차이

| | 성격 | 위치 |
|---|---|---|
| **고정 데이터** | 누가 실행하든 항상 동일 (카드 DB, 가맹점 목록) | `output/` |
| **사용자 종속** | 업로드한 CSV에 따라 매번 달라짐 (매칭·시뮬레이션·추천 결과) | `debug_output/` |

> `debug_output/` 안의 JSON은 **개인 거래내역 기반 산출물**이라 git에 포함하지 않습니다.
> 앱을 `DEBUG_MODE=True`로 실행하면 단계별 JSON이 로컬에 생성됩니다.

---

## 입력 CSV 형식

`date`, `merchant_name`, `amount` 세 컬럼이 필요합니다.

```csv
date,merchant_name,amount
2025.01.05,편의점 ○○점,3000
2025.01.12,카페 ○○점,5500
2025.02.03,통신요금 자동이체,55000
```

| 컬럼 | 형식 | 설명 |
|------|------|------|
| `date` | `YYYY.MM.DD` 또는 `YYYY-MM-DD` | 거래일 |
| `merchant_name` | 문자열 | 영수증 가맹점명 그대로 (지점명·법인표기 포함 OK — GPT가 브랜드명으로 정규화) |
| `amount` | 정수 | 결제금액 (원) |

> **12개월치 데이터를 권장합니다.** 기간이 짧으면 절감액이 실제보다 낮게 계산됩니다.

---

## v2와의 차이

| 항목 | v2 (RAG + LLM) | v3 (가맹점 매칭) |
|------|----------------|------------------|
| 매칭 단위 | 소비 **카테고리** (12종) | **가맹점/브랜드** (퍼지매칭) |
| 후보 검색 | RAG 키워드 매칭 Top-15 | 전체 100종 시뮬레이션 |
| 혜택 파싱 | 사전 파싱 JSON (카테고리별 할인율) | 구조화 JSON (가맹점·조건별 혜택) |
| 절감액 계산 | Python 결정론적 | LLM + Python 검산 (이상값 자동 대체) |
| 입력 CSV | 6컬럼 (카테고리 포함) | 3컬럼 (date, merchant, amount) |

---

## 설계 포인트

### LLM 결과 검산

시뮬레이션에서 GPT가 계산한 절감액을 Python 코드(`deterministic_simulate_card()`)로 다시 계산해 비교합니다. LLM 결과가 비정상이면 코드 계산 결과로 자동 대체합니다.

### 가맹점 퍼지매칭

실제 영수증 가맹점명에는 지점명·법인표기가 붙어 있습니다. GPT가 이를 브랜드명으로 정규화해 가맹점 목록과 대조합니다.

### 모델 역할 분리

| 단계 | 모델 | 이유 |
|------|------|------|
| 거래 매칭 | `gpt-4o` | 매칭 오류가 전체 추천에 영향 → 정확도 우선 |
| 절감액 시뮬레이션 | `gpt-4o-mini` | 카드 100개 반복 호출 → 비용 절감 |
| 추천 설명 생성 | `gpt-4o` | 자연스러운 한국어 품질 우선 |

> 모델은 앱 사이드바에서 직접 바꿀 수 있습니다.

---

## 사전 준비 스크립트 재실행 (선택)

카드 데이터를 최신화할 때만 순서대로 실행합니다.

```bash
python preprocess_benefits.py
python structure_benefits.py   # GPT-4o 호출, 비용 발생
python build_merchant_set.py
```
