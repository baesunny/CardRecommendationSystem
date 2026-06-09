# v2.1 — AI 맞춤 신용카드 추천 시스템 (RAG + LLM)

1년치 카드 이용 내역을 분석해 소비 패턴에 맞는 신용카드를 추천하고, 예상 절감액을 시뮬레이션하는 **RAG + LLM 하이브리드 추천 시스템**이다.

> LLM은 혜택 파싱(오프라인 사전 처리)과 자연어 설명만 담당하고, 절감액·순위 계산은 Python으로 결정론적으로 처리해 산수 오류를 방지한다.

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| 소비 패턴 분석 | CSV 업로드 → 카테고리별 월/연간 지출, 비율 산출 |
| RAG 후보 검색 | 100종 카드 DB에서 소비 비율 가중 키워드 매칭으로 Top-15 후보 선별 |
| 사전 혜택 파싱 | 100종 카드 혜택을 오프라인에서 LLM으로 구조화해 JSON에 저장 (추천 시 즉시 조회) |
| 절감액 시뮬레이션 | 전월 실적 조건, 월 한도, 연회비를 반영한 순 혜택 계산 |
| AI 추천 설명 | GPT가 계산 결과를 바탕으로 추천 이유·핵심 혜택·전체 요약 생성 |
| Streamlit UI | 업로드 → 분석 → 결과 시각화를 한 화면에서 제공 |

---

## v2.1 성능 개선 — 사전 혜택 파싱

이전에는 추천 요청마다 RAG 후보(최대 15장)의 `raw_benefits`를 LLM으로 실시간 파싱했다.  
**v2.1**에서는 100종 전체 카드 혜택을 미리 파싱해 `outputs/card_parsed_benefits.json`에 저장하고, 추천 시에는 이 파일에서 즉시 조회한다.

| 구분 | v2 (실시간 파싱) | v2.1 (사전 파싱) |
|------|------------------|------------------|
| 파싱 시점 | 추천 요청마다 LLM 호출 | 오프라인 1회 (`scripts/preparse_benefits.py`) |
| 추천 지연 | 후보 수 × LLM 응답 시간 | JSON 조회 (거의 즉시) |
| LLM 호출 | 파싱 + 설명 | 설명만 |

조회 우선순위 (`benefit_parser.py`):

1. `outputs/card_parsed_benefits.json` — 전체 사전 파싱 (기본)
2. `gpt/.parser_cache.json` — 런타임 LLM 캐시 (로컬, git 제외)
3. LLM API — 사전 파싱·캐시 모두 없을 때만 폴백

---

## 시스템 아키텍처

```
[CSV 거래내역]
      │
      ▼
 spending_analyzer ──→ SpendingProfile (소비 패턴)
      │
      ▼
 rag_retriever ──────→ Top-15 후보 카드 (키워드 매칭)
      │
      ▼
 benefit_parser ─────→ {카테고리: 할인율} 조회  [사전 파싱 JSON / LLM 폴백]
      │
      ▼
 calculator ─────────→ 절감액·순위 계산          [Python]
      │
      ▼
 recommender ────────→ 자연어 설명 생성            [LLM]
      │
      ▼
 Streamlit UI
```

### 설계 포인트

- **v1 문제**: GPT 한 번 호출로 파싱·계산·설명을 모두 처리 → 산수 오류 빈번
- **v2 해결**: 역할 3단 분리 — LLM(파싱) → Python(계산) → LLM(설명)
- **v2.1 개선**: 100종 카드 혜택 사전 파싱으로 추천 응답 속도 대폭 단축

---

## 기술 스택

- **Language**: Python 3.9+
- **LLM**: OpenAI GPT-4o / GPT-4o-mini
- **UI**: Streamlit
- **Data**: Pandas, NumPy
- **Card DB**: 100종 국내 신용카드 (전처리된 CSV + 메타데이터 + 사전 파싱 JSON)

---

## 프로젝트 구조

```
v2/
├── gpt/                        # 핵심 애플리케이션 모듈
│   ├── app.py                  # Streamlit UI
│   ├── recommender.py          # 전체 파이프라인 오케스트레이션
│   ├── benefit_parser.py       # 혜택 조회 (사전 파싱 JSON 우선)
│   ├── calculator.py           # 결정론적 절감액·순위 계산
│   └── openai_config.py        # API 키 로딩 및 클라이언트 설정
├── rag/                        # 검색·분석 모듈
│   ├── spending_analyzer.py    # 거래내역 → 소비 프로필 변환
│   └── rag_retriever.py        # 소비 패턴 기반 카드 후보 검색
├── scripts/
│   └── preparse_benefits.py    # 100종 카드 혜택 오프라인 일괄 파싱
├── outputs/                    # 카드 DB 및 사전 파싱 결과
│   ├── card_processed.csv      # 100종 카드 혜택 원문
│   ├── card_parsed_benefits.json  # 사전 파싱된 {카테고리: 할인율} (100종)
│   ├── embedding_metadata.json # 카드 메타 (연회비, 전월실적 등)
│   └── card_embeddings.npy     # 임베딩 (향후 벡터 검색 확장용)
├── examples/                   # 샘플 데이터
│   ├── card_history_2024.csv   # 1년치 가상 거래내역 (1,326건)
│   └── generate_card_history.py
├── requirements.txt
└── README.md
```

---

## 설치 및 실행

### 1. 의존성 설치

```bash
cd v2
pip install -r requirements.txt
```

### 2. API 키 설정

```bash
copy ..\.env.example ..\.env   # Windows (저장소 루트)
# cp ../.env.example ../.env     # macOS / Linux
```

`.env` 파일에 OpenAI API 키를 입력한다. **`.env`는 git에 커밋하지 않는다.**

```
OPENAI_API_KEY=your-api-key-here
```

### 3. 앱 실행

```bash
streamlit run gpt/app.py
```

브라우저에서 `examples/card_history_2024.csv`를 업로드해 테스트할 수 있다.

### 4. (선택) 혜택 사전 파싱 재생성

카드 DB(`outputs/card_processed.csv`)를 수정했거나 파싱 결과를 갱신할 때:

```bash
py -3 scripts/preparse_benefits.py
py -3 scripts/preparse_benefits.py --model gpt-4o-mini
py -3 scripts/preparse_benefits.py --skip-llm   # 기존 캐시/파일만 병합
```

---

## CSV 입력 형식

| 컬럼 | 설명 | 예시 |
|------|------|------|
| 거래일시 | `YYYY-MM-DD HH:MM` | 2024-03-15 14:30 |
| 카드번호 | 마스킹 가능 | 5234-****-****-1234 |
| 가맹점명 | 가맹점 이름 | 스타벅스 홍대입구점 |
| 카테고리 | 소비 카테고리 | 카페 |
| 결제금액 | 정수 (원) | 5500 |
| 승인번호 | 승인 번호 | 12345678 |

지원 카테고리: 온라인쇼핑, 음식점, 마트/슈퍼, 배달앱, 주유소, 편의점, 운동/스포츠, 의료/약국, 카페, 대중교통, 통신, OTT/구독

---

## 파이프라인 상세

### Stage 1 — 소비 패턴 분석 (`spending_analyzer`)

- 연간/월 평균 지출, 카테고리별 비율·월 평균 산출
- 전월 실적 조건 비교에 사용

### Stage 2 — RAG 후보 검색 (`rag_retriever`)

- 사용자 소비 비율을 가중치로 카드 `main_categories` 키워드 매칭
- Top-15 후보 선별 (향후 벡터 임베딩 검색 확장 가능)

### Stage 3 — 혜택 조회 (`benefit_parser`)

- 카드별 `raw_benefits` → `{카테고리: 할인율, 월한도}` 구조화 데이터 조회
- `outputs/card_parsed_benefits.json`에서 100종 전체 사전 파싱 결과를 즉시 로드
- 미스 시 `gpt/.parser_cache.json` 또는 LLM API 폴백

### Stage 4 — 절감액 계산 (`calculator`, Python)

- 전월 실적 달성 여부 판단
- 카테고리별 절감액 (월 한도 적용) → 연간 총 절감 − 연회비 = **순 혜택**
- 순 혜택 내림차순 Top-K 선정

### Stage 5 — 추천 설명 (`recommender`, LLM)

- 계산 결과를 입력으로 `reason`, `key_benefits`, `overall_summary` 생성
- 수치는 입력값 그대로 인용 (재계산 금지)
