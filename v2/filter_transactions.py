"""
사용자 거래 데이터 + output/global_merchant_set.json
→ OpenAI 퍼지매칭 결과 생성
"""

import csv
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

from openai import OpenAI

# ── 경로 설정 ───────────────────────────────────────────────
BASE_DIR         = Path(__file__).resolve().parent
OUTPUT_DIR       = BASE_DIR / "output"        # 고정 데이터 (가맹점 집합 등)
DEBUG_OUTPUT_DIR = BASE_DIR / "debug_output"  # 디버그 모드 산출물
MERCHANT_JSON    = OUTPUT_DIR / "global_merchant_set.json"

# ── 디버그 모드 ─────────────────────────────────────────────
# True : 독립 실행 가능. ex_data/ CSV 읽고 filtered_transactions.json 저장
# False: app.py 메모리 파이프라인용 (파일 입출력 없음)
DEBUG_MODE = True

MIN_MERCHANT_COUNT = 100

MODEL       = "gpt-4o"
BATCH_SIZE  = 50
MAX_RETRIES = 2
SLEEP_SEC   = 0.5

# ── 이슈 2 수정: 가맹점명 정규화 테이블 ────────────────────
# LLM이 같은 가맹점을 다른 이름으로 매칭한 경우 하나로 통일
MERCHANT_ALIASES: dict[str, str] = {
    # 메가커피 계열
    "MGC메가커피":   "메가커피",
    "메가MGC커피":   "메가커피",
    # 투썸
    "투썸플레이트":   "투썸플레이스",
    # 파리바게뜨
    "파리바게트":    "파리바게뜨",
    # 파스쿠찌
    "파스쿠치":     "파스쿠찌",
    # 디즈니+
    "디즈니 플러스":  "디즈니+",
    "디즈니플러스":   "디즈니+",
    # SSG
    "SSG COM":    "SSG.COM",
    # L페이
    "L.pay":      "L.PAY",
    "L페이":       "L.PAY",
    # 삼성페이
    "삼성 페이":    "삼성페이",
    # SSG페이
    "SSGPAY":     "SSG PAY",
    "SSG페이":     "SSG PAY",
    # 할리스
    "할리스커피":    "할리스",
}


def normalize_merchant(name: str | None) -> str | None:
    if name is None:
        return None
    return MERCHANT_ALIASES.get(name, name)


SYSTEM_PROMPT = """\
You are a transaction matcher for Korean credit card benefits.
Your job is to match merchant names from transaction records
to the global merchant set of a card benefit system.
You must respond ONLY with a valid JSON array. No explanation,
no markdown, no code blocks."""

USER_PROMPT_TEMPLATE = """\
아래는 사용자의 카드 거래내역 {batch_size}건이다.
각 거래의 merchant_name을 분석해서 global_merchant_set과 매칭해라.

[global_merchant_set]
{merchant_set_str}

[거래내역]
{transactions_json}

[매칭 규칙]
1. 정확히 일치하지 않아도 같은 가맹점이면 매칭한다.
   - "씨유(CU) 분당효자촌 현대점" → "CU"
   - "스타벅스_주문-에스씨" → "스타벅스"
   - "이디야커피 분당효자촌점" → "이디야"
   - "한국맥도날드(유) 분당효자점" → "맥도날드"
   - "우아한형제들" → "배달의민족"
   - "교통-버스25건" → "버스"
   - "교통-지하철2건" → "지하철"
   - "카카오_택시_0" → "카카오T"
   - "LG U+ 통신요금자동이체" → "LG U+"
2. 지점명, 번호, 특수문자, 법인표기는 무시하고 브랜드명으로 판단한다.
3. 운영사와 브랜드가 다른 경우 브랜드로 매칭한다.
   - "우아한형제들" → "배달의민족"
   - "쿠팡 주식회사" → "쿠팡"
4. 어떤 가맹점과도 매칭되지 않으면 matched_merchant를 null로 한다.
5. 확신이 없으면 null로 한다. 억지 매칭 금지.

[출력 형식]
입력된 거래내역과 동일한 순서로 아래 형식의 JSON 배열을 반환해라.
[
  {{
    "date": "2025.01.08",
    "original_merchant": "씨유(CU) 분당효자촌 현대점",
    "matched_merchant": "CU",
    "amount": 2000
  }},
  {{
    "date": "2025.01.15",
    "original_merchant": "동네 세탁소",
    "matched_merchant": null,
    "amount": 15000
  }}
]
반드시 JSON 배열만 반환. 다른 텍스트 절대 금지."""


def load_transactions(data_dir: Path) -> list[dict]:
    rows: list[dict] = []
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"CSV 파일 없음: {data_dir}")
    for path in csv_files:
        with open(path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                rows.append({
                    "date":          row["date"].strip(),
                    "merchant_name": row["merchant_name"].strip(),
                    "amount":        int(str(row["amount"]).replace(",", "").strip()),
                })
        print(f"  CSV 로드: {path.name}  ({len(rows)}건 누적)")
    return rows


def extract_json_array(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if m:
        return m.group(1).strip()
    start = text.find("[")
    end   = text.rfind("]")
    if start != -1 and end != -1:
        return text[start:end + 1]
    return text


# ── 이슈 1 수정: 재시도 로직 명확화 ────────────────────────
def process_batch(
    client: OpenAI,
    batch: list[dict],
    merchant_set_str: str,
) -> list[dict] | None:
    tx_json = json.dumps(
        [{"date": t["date"], "original_merchant": t["merchant_name"], "amount": t["amount"]}
         for t in batch],
        ensure_ascii=False,
        indent=2,
    )
    user_msg = USER_PROMPT_TEMPLATE.format(
        batch_size=len(batch),
        merchant_set_str=merchant_set_str,
        transactions_json=tx_json,
    )

    last_error = ""
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0,
            )
            raw    = resp.choices[0].message.content or ""
            parsed = json.loads(extract_json_array(raw))

            if not isinstance(parsed, list) or len(parsed) != len(batch):
                # 길이 불일치 → 명시적으로 재시도 경로로 진입
                last_error = f"응답 길이 불일치 (기대 {len(batch)}, 실제 {len(parsed) if isinstance(parsed, list) else 'N/A'})"
                raise ValueError(last_error)

            return parsed

        except (json.JSONDecodeError, ValueError) as e:
            last_error = str(e)
        except Exception as e:
            last_error = f"API 오류: {e}"

        if attempt < MAX_RETRIES:
            print(f"      재시도 {attempt + 1}/{MAX_RETRIES} ({last_error})")
            time.sleep(SLEEP_SEC)

    print(f"      최종 실패: {last_error}")
    return None


def main():
    if not DEBUG_MODE:
        print("디버그 모드 비활성화 — app.py 메모리 파이프라인으로 실행하세요.")
        print("독립 실행하려면 DEBUG_MODE = True 로 변경하세요.")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

    client = OpenAI(api_key=api_key)

    print(f"\n{'='*58}")
    print("  거래내역 퍼지매칭 시작")
    print(f"{'='*58}")

    with open(MERCHANT_JSON, encoding="utf-8") as f:
        gms = json.load(f)
    merchants: list[str] = gms["merchants"]

    if len(merchants) < MIN_MERCHANT_COUNT:
        raise ValueError(
            f"[오류] global_merchant_set 가맹점 수가 너무 적습니다: {len(merchants)}개\n"
            f"  최소 {MIN_MERCHANT_COUNT}개 이상이어야 합니다.\n"
            f"  파일 경로: {MERCHANT_JSON}"
        )

    merchant_set_str = "\n".join(merchants)
    print(f"  글로벌 가맹점 집합: {len(merchants)}개")

    transactions = load_transactions(EX_DATA_DIR)
    total = len(transactions)
    print(f"  거래내역 합계: {total}건\n")

    batches       = [transactions[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    total_batches = len(batches)
    all_results:   list[dict] = []
    failed_batches: list[int] = []

    for batch_idx, batch in enumerate(batches, 1):
        print(
            f"  배치 [{batch_idx:>2}/{total_batches}]  "
            f"({(batch_idx-1)*BATCH_SIZE + 1}~"
            f"{min(batch_idx*BATCH_SIZE, total)}건)",
            end=" ... ",
            flush=True,
        )
        result = process_batch(client, batch, merchant_set_str)

        if result is None:
            print("실패 — 원본 보존")
            failed_batches.append(batch_idx)
            for t in batch:
                all_results.append({
                    "date":              t["date"],
                    "original_merchant": t["merchant_name"],
                    "matched_merchant":  None,
                    "amount":            t["amount"],
                })
        else:
            # ── 이슈 2 수정: 정규화 적용 ──────────────────────
            for r in result:
                r["matched_merchant"] = normalize_merchant(r.get("matched_merchant"))

            matched_in_batch = sum(1 for r in result if r.get("matched_merchant"))
            print(f"OK  (매칭 {matched_in_batch}/{len(batch)}건)")
            all_results.extend(result)

        time.sleep(SLEEP_SEC)

    matched       = [r for r in all_results if r.get("matched_merchant")]
    unmatched     = [r for r in all_results if not r.get("matched_merchant")]
    matched_cnt   = len(matched)
    unmatched_cnt = len(unmatched)

    match_rate = matched_cnt / total * 100 if total else 0

    print(f"\n{'='*58}")
    print(f"  처리 완료")
    print(f"  전체 거래:    {total:>5}건")
    print(f"  매칭 성공:    {matched_cnt:>5}건  ({match_rate:.1f}%)")
    print(f"  매칭 실패:    {unmatched_cnt:>5}건")
    # DEBUG_MODE일 때만 파일 저장
    output_path = DEBUG_OUTPUT_DIR / "filtered_transactions.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"total_transactions": total, "matched_count": matched_cnt,
                   "unmatched_count": unmatched_cnt, "transactions": matched},
                  f, ensure_ascii=False, indent=2)
    print(f"  저장 위치:    {output_path}")

    if failed_batches:
        print(f"\n  [실패 배치: {failed_batches}]")

    counter = Counter(r["matched_merchant"] for r in matched)
    print(f"\n  {'─'*54}")
    print(f"  매칭된 가맹점별 거래 건수 Top 20")
    print(f"  {'─'*54}")
    for rank, (merchant, cnt) in enumerate(counter.most_common(20), 1):
        bar = "■" * min(cnt, 35)
        print(f"  {rank:>2}. {merchant:<22} {cnt:>3}건  {bar}")

    print(f"\n  {'─'*54}")
    print(f"  매칭 안 된 거래 샘플 (최대 20건, 육안 확인용)")
    print(f"  {'─'*54}")
    for r in unmatched[:20]:
        print(f"  {r['date']}  {r['original_merchant']:<35}  {r['amount']:>7,}원")

    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
