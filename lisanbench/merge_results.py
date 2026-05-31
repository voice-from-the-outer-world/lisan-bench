#!/usr/bin/env python3
"""
merge_results.py

Merge two JSON result files produced by your benchmarking pipeline.

Usage:
    uv run python merge_results.py file_a.json file_b.json merged.json
    # add -y / --yes to skip interactive confirmation if temperatures differ
"""

import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _iso(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp (handles Python 3.11+ microsecond format)."""
    return datetime.fromisoformat(ts)


def _confirm(msg: str, auto_yes: bool) -> None:
    if auto_yes:
        return
    ans = input(f"{msg} [y/N]: ").strip().lower()
    if ans not in ("y", "yes"):
        print("Aborted.")
        raise SystemExit(1)


def _starting_word_set(metadata: Dict[str, Any]) -> set[str]:
    value = metadata.get("starting_words_tested")
    words: List[str] = []

    if isinstance(value, dict):
        for model_words in value.values():
            if not isinstance(model_words, list):
                raise ValueError("starting_words_tested must contain word lists")
            words.extend(str(word).strip() for word in model_words if str(word).strip())
    elif isinstance(value, list):
        words.extend(str(word).strip() for word in value if str(word).strip())
    else:
        raise ValueError("starting_words_tested must be a list or model-to-list mapping")

    return set(words)


def merge_metadata(meta_a: Dict[str, Any], meta_b: Dict[str, Any], auto_yes: bool) -> Dict[str, Any]:
    """Merge the two metadata blocks according to the spec."""
    # 1. temperature must match
    if meta_a["temperature"] != meta_b["temperature"]:
        _confirm(
            f"Temperatures differ ({meta_a['temperature']} vs {meta_b['temperature']}). Merge anyway?",
            auto_yes,
        )

    # 2. starting word sets must match exactly, including model-specific reduced metadata.
    if _starting_word_set(meta_a) != _starting_word_set(meta_b):
        raise ValueError("starting_words_tested do not match!")

    if meta_a["num_trials"] != meta_b["num_trials"]:
        raise ValueError(
            f"num_trials do not match ({meta_a['num_trials']} vs {meta_b['num_trials']})!"
        )

    merged: Dict[str, Any] = {
        # take latest of the two
        "timestamp": max(meta_a["timestamp"], meta_b["timestamp"]),
        "temperature": meta_a["temperature"],
        # sums
        "total_api_cost_usd": meta_a["total_api_cost_usd"] + meta_b["total_api_cost_usd"],
        "total_input_tokens": meta_a["total_input_tokens"] + meta_b["total_input_tokens"],
        "total_output_tokens": meta_a["total_output_tokens"] + meta_b["total_output_tokens"],
        "total_reasoning_tokens": int(meta_a.get("total_reasoning_tokens", 0) or 0)
        + int(meta_b.get("total_reasoning_tokens", 0) or 0),
        # union without duplicates
        "models_tested": sorted(set(meta_a["models_tested"]).union(meta_b["models_tested"])),
        "starting_words_tested": meta_a["starting_words_tested"],
        # latest update
        "last_updated": max(meta_a["last_updated"], meta_b["last_updated"]),
        # add up trials, keep smaller thread count
        "num_trials": meta_a["num_trials"],
        "threads": min(meta_a["threads"], meta_b["threads"]),
        # later completion time
        "completion_time": max(meta_a["completion_time"], meta_b["completion_time"]),
        # keep status from whichever finished later (optional, theyâ€™re usually identical)
        "status": meta_a["status"] if _iso(meta_a["completion_time"]) >= _iso(meta_b["completion_time"]) else meta_b["status"],
    }

    # copy through any extra keys not explicitly handled, preferring the newer file
    newer = meta_a if _iso(meta_a["timestamp"]) >= _iso(meta_b["timestamp"]) else meta_b
    for k, v in newer.items():
        if k not in merged:
            merged[k] = v

    return merged


def merge_results(res_a: Dict[str, List[Any]], res_b: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """Merge the results dict (lists concatenated per model, no duplicate model keys)."""
    merged = {model: list(entries) for model, entries in res_a.items()}
    for model, entries in res_b.items():
        merged.setdefault(model, []).extend(entries)
    return merged


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON through a same-directory temp file, then atomically replace."""
    out_path = path.resolve()
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{out_path.name}.",
        suffix=".tmp",
        dir=out_dir,
        text=True,
    )
    tmp_path = Path(tmp_name)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        with tmp_path.open("r", encoding="utf-8") as f:
            json.load(f)

        os.replace(tmp_path, out_path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def merge_files(path_a: Path, path_b: Path, out_path: Path, auto_yes: bool) -> None:
    with path_a.open("r", encoding="utf-8") as f:
        data_a = json.load(f)
    with path_b.open("r", encoding="utf-8") as f:
        data_b = json.load(f)

    merged = {
        "metadata": merge_metadata(data_a["metadata"], data_b["metadata"], auto_yes),
        "results": merge_results(data_a["results"], data_b["results"]),
    }

    atomic_write_json(out_path, merged)
    print(f"Merged file written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge two JSON result files.")
    parser.add_argument("file_a", type=Path, help="First JSON file")
    parser.add_argument("file_b", type=Path, help="Second JSON file")
    parser.add_argument("output", type=Path, help="Output file")
    parser.add_argument("-y", "--yes", action="store_true", help="Automatically confirm mismatched temps")
    args = parser.parse_args()

    merge_files(args.file_a, args.file_b, args.output, args.yes)


if __name__ == "__main__":
    main()
