"""
전체 카드 혜택 오프라인 파싱 → outputs/card_parsed_benefits.json

사용법 (프로젝트 루트 또는 scripts/ 에서):
  py -3 scripts/preparse_benefits.py
  py -3 scripts/preparse_benefits.py --model gpt-4o-mini
  py -3 scripts/preparse_benefits.py --skip-llm   # 기존 캐시/파일만 병합
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_GPT = _ROOT / "gpt"
if str(_GPT) not in sys.path:
    sys.path.insert(0, str(_GPT))

import benefit_parser as bp  # noqa: E402

CARD_CSV = _ROOT / "outputs" / "card_processed.csv"
CARD_META = _ROOT / "outputs" / "embedding_metadata.json"
PREPARSED_FILE = bp.PREPARSED_FILE
RUNTIME_CACHE = bp.CACHE_FILE


def _load_cards() -> list[dict]:
    df = pd.read_csv(CARD_CSV)
    prev_req_map: dict[str, int] = {}
    if CARD_META.exists():
        meta = json.loads(CARD_META.read_text(encoding="utf-8"))
        prev_req_map = {
            c["card_name"]: int(c["previous_month_requirement"])
            for c in meta.get("cards", [])
        }
    cards = []
    for _, row in df.iterrows():
        cards.append({
            "card_company": str(row["card_company"]),
            "card_name": str(row["card_name"]),
            "annual_fee": int(row["annual_fee"]),
            "previous_month_requirement": int(
                prev_req_map.get(str(row["card_name"]), 0)
            ),
            "main_categories": str(row["main_categories"]),
            "raw_benefits": str(row["raw_benefits"]),
        })
    return cards


def _load_existing_entries() -> dict[str, dict]:
    entries: dict[str, dict] = {}
    if PREPARSED_FILE.exists():
        data = json.loads(PREPARSED_FILE.read_text(encoding="utf-8"))
        entries.update(data.get("entries", {}))
    if RUNTIME_CACHE.exists():
        try:
            entries.update(json.loads(RUNTIME_CACHE.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    return entries


def run(model: str = "gpt-4o-mini", skip_llm: bool = False) -> None:
    cards = _load_cards()
    entries = _load_existing_entries()

    keys = [
        bp.cache_key(c["card_name"], c["raw_benefits"]) for c in cards
    ]
    miss_indices = [i for i, k in enumerate(keys) if k not in entries]

    print(f"cards={len(cards)}  already_parsed={len(cards) - len(miss_indices)}  to_parse={len(miss_indices)}")

    if miss_indices and not skip_llm:
        client = bp.create_client()
        total = len(miss_indices)
        done = 0

        def _job(idx: int):
            card = cards[idx]
            parsed = bp.parse_single_card_llm(
                card["card_name"],
                card["raw_benefits"],
                client,
                model,
            )
            return idx, card, parsed

        with ThreadPoolExecutor(max_workers=bp._MAX_WORKERS) as ex:
            for idx, card, parsed in ex.map(_job, miss_indices):
                done += 1
                if parsed:
                    entries[keys[idx]] = parsed
                print(
                    f"[{done}/{total}] {card['card_company']} - {card['card_name']}"
                    + (" OK" if parsed else " (empty)")
                )
    elif miss_indices and skip_llm:
        print(f"skip-llm: {len(miss_indices)} cards still missing (run without --skip-llm)")

    payload = {
        "version": 1,
        "source": "card_processed.csv",
        "model": model,
        "parsed_at": datetime.now(timezone.utc).isoformat(),
        "num_cards": len(cards),
        "num_entries": len(entries),
        "entries": entries,
    }
    PREPARSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    PREPARSED_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    covered = sum(1 for k in keys if k in entries)
    print(f"saved → {PREPARSED_FILE}")
    print(f"coverage: {covered}/{len(cards)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-parse all card benefits to JSON")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Only merge existing pre-parsed file and .parser_cache.json",
    )
    args = parser.parse_args()
    run(model=args.model, skip_llm=args.skip_llm)
