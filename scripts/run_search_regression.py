#!/usr/bin/env python3
"""Run image-search regression cases against a running API server."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def _multipart_body(fields: dict[str, str], file_field: str, file_path: Path) -> tuple[bytes, str]:
    boundary = "----search-regression-" + uuid.uuid4().hex
    mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    parts: list[bytes] = []
    for key, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
        parts.append(str(value).encode())
        parts.append(b"\r\n")
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        (
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{file_path.name}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode()
    )
    parts.append(file_path.read_bytes())
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _post_search(base_url: str, image_path: Path, top_k: int, crop: dict[str, Any] | None) -> dict[str, Any]:
    fields = {"result_top_k": str(top_k)}
    if crop:
        for key in ("crop_x", "crop_y", "crop_w", "crop_h"):
            if key in crop:
                fields[key] = str(crop[key])
    body, content_type = _multipart_body(fields, "file", image_path)
    req = urllib.request.Request(
        base_url.rstrip("/") + "/search",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _matches(actual: str, expected: str) -> bool:
    return actual == expected or actual.startswith(expected + "-")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("data/search_regression_cases.json"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--max-rank", type=int, default=10)
    parser.add_argument("--case", action="append", default=[], help="Run only matching case id(s).")
    args = parser.parse_args()

    data = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = list(data.get("cases", []))
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if str(case.get("id", "")) in wanted]
    if not cases:
        print("no cases selected", file=sys.stderr)
        return 2

    ok = True
    for case in cases:
        case_id = str(case.get("id", ""))
        image_path = Path(str(case.get("query_image", "")))
        expected = str(case.get("expected_style_code", ""))
        max_rank = int(case.get("max_rank", args.max_rank))
        if not image_path.exists():
            print(f"FAIL {case_id}: missing {image_path}")
            ok = False
            continue
        try:
            response = _post_search(args.base_url, image_path, args.top_k, case.get("crop"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"FAIL {case_id}: request failed: {exc}")
            ok = False
            continue
        rows = response.get("topk_style_codes") or []
        codes = [str(row.get("style_code", "")) for row in rows]
        rank = next((idx + 1 for idx, code in enumerate(codes) if _matches(code, expected)), None)
        status = "PASS" if rank is not None and rank <= max_rank else "FAIL"
        if status == "FAIL":
            ok = False
        rank_text = str(rank) if rank is not None else "-"
        print(f"{status} {case_id}: expected={expected} rank={rank_text} top10={','.join(codes[:10])}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
