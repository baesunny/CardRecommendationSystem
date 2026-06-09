"""
simulation_results 데이터 → Top 5 카드 추출
net_saving 상위 5개 카드 추출
"""

import json
from datetime import date
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent

# ── 디버그 모드 ─────────────────────────────────────────────
# True : 독립 실행 가능. simulation_results.json 읽고 top5_cards.json 저장
# False: app.py 메모리 파이프라인용 (파일 입출력 없음)
DEBUG_MODE = True


def main():
    if not DEBUG_MODE:
        print("디버그 모드 비활성화 — app.py 메모리 파이프라인으로 실행하세요.")
        print("독립 실행하려면 DEBUG_MODE = True 로 변경하세요.")
        return

    input_json  = BASE_DIR / "debug_output" / "simulation_results.json"
    output_path = BASE_DIR / "debug_output" / "top5_cards.json"

    with open(input_json, encoding="utf-8") as f:
        sim: dict = json.load(f)

    results: list[dict] = sim["results"]   # 이미 net_saving 내림차순 정렬
    top5 = [r for r in results if r.get("rank", 99) <= 5]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"extracted_at": str(date.today()), "top5": top5},
                  f, ensure_ascii=False, indent=2)

    print(f"\n{'='*52}")
    print(f"  Top 5 카드 추출 완료")
    print(f"  저장 위치: {output_path}")
    print(f"  {'─'*50}")
    print(f"  {'순위':<4} {'카드명':<36} {'카드사':<10} {'순절감':>9}")
    print(f"  {'─'*50}")
    for r in top5:
        print(
            f"  {r['rank']:<4} {r.get('card_name','')[:34]:<34} "
            f"{r.get('card_company','')[:9]:<9} "
            f"{r.get('net_saving', 0):>9,}원"
        )
    print(f"{'='*52}\n")


if __name__ == "__main__":
    main()
