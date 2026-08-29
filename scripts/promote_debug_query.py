#!/usr/bin/env python3
"""Promote a saved debug query image into regression fixtures."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _upsert_case(path: Path, case: dict[str, Any]) -> None:
    data = _load_json(path, {"cases": []})
    cases = list(data.get("cases", []))
    case_id = str(case["id"])
    for idx, existing in enumerate(cases):
        if str(existing.get("id", "")) == case_id:
            cases[idx] = case
            break
    else:
        cases.append(case)
    data["cases"] = cases
    _dump_json(path, data)


def _upsert_label(path: Path, query_image: str, style_code: str) -> None:
    data = _load_json(path, {"labels": []})
    labels = list(data.get("labels", []))
    row = {"query_image": query_image, "style_code": style_code}
    for idx, existing in enumerate(labels):
        if str(existing.get("query_image", "")) == query_image:
            labels[idx] = row
            break
    else:
        labels.append(row)
    data["labels"] = labels
    _dump_json(path, data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("debug_image", type=Path, help="Path from logs: saved=outputs/debug_queries/...")
    parser.add_argument("--id", required=True, help="Regression case id, e.g. gz24_0067_region_120")
    parser.add_argument("--expected", required=True, help="Expected style code.")
    parser.add_argument("--description", default="")
    parser.add_argument("--ext", default="", help="Destination extension. Defaults to source suffix.")
    parser.add_argument("--max-rank", type=int, default=10)
    parser.add_argument("--label-memory", action="store_true", help="Also add to data/query_labels.json.")
    parser.add_argument("--samples-dir", type=Path, default=Path("data/test_samples"))
    parser.add_argument("--cases", type=Path, default=Path("data/search_regression_cases.json"))
    parser.add_argument("--labels", type=Path, default=Path("data/query_labels.json"))
    args = parser.parse_args()

    src = args.debug_image
    if not src.exists() or not src.is_file():
        raise SystemExit(f"missing debug image: {src}")
    ext = args.ext if args.ext else src.suffix
    if not ext.startswith("."):
        ext = "." + ext
    args.samples_dir.mkdir(parents=True, exist_ok=True)
    dest = args.samples_dir / f"{args.id}{ext.lower()}"
    shutil.copy2(src, dest)

    query_image = str(dest)
    case = {
        "id": args.id,
        "query_image": query_image,
        "expected_style_code": args.expected,
        "max_rank": args.max_rank,
        "description": args.description or f"Regression promoted from {src.name}.",
    }
    _upsert_case(args.cases, case)
    if args.label_memory:
        _upsert_label(args.labels, query_image, args.expected)
    print(query_image)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
