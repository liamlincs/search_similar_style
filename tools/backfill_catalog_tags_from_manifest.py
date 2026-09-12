#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.catalog_store import CatalogStore, filename_to_style_code, make_typed_tag, parse_catalog_tag


def split_values(value: Any) -> list[str]:
    if value is None:
        return []
    raw_items: Iterable[Any]
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = re.split(r"[、,，\s]+", str(value or ""))
    out: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        clean = str(raw or "").strip()
        if not clean:
            continue
        key = clean.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def normalize_tags(tags: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        clean = str(tag or "").strip()
        if not clean:
            continue
        key = clean.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def tags_from_manifest_item(item: dict[str, Any]) -> list[str]:
    raw_tags: list[str] = []
    raw_tags.extend(split_values(item.get("tags")))
    raw_tags.extend(split_values(item.get("raw_tags")))
    for year in split_values(item.get("year_tag") or item.get("year") or item.get("years")):
        raw_tags.append(year if str(year).startswith("year:") else make_typed_tag("year", year))
    for category in split_values(item.get("category") or item.get("category_tags")):
        raw_tags.append(category if str(category).startswith("category:") else make_typed_tag("category", category))
    for subcategory in split_values(item.get("subcategory") or item.get("subcategory_tags")):
        raw_tags.append(subcategory if str(subcategory).startswith("subcategory:") else make_typed_tag("subcategory", subcategory))
    return normalize_tags(raw_tags)


def tags_from_source_rel_path(source_rel_path: str) -> list[str]:
    parts = [part.strip() for part in Path(str(source_rel_path or "")).parts if part.strip()]
    if len(parts) < 4:
        return []
    year_dir, category_dir, subcategory_dir = parts[0], parts[1], parts[2]
    tags = []
    if re.fullmatch(r"20\d{2}", year_dir):
        tags.append(make_typed_tag("year", year_dir))
    tags.append(make_typed_tag("category", category_dir))
    tags.append(make_typed_tag("subcategory", subcategory_dir))
    return normalize_tags(tag for tag in tags if tag)


def discover_manifests(paths: list[str], standard_dir: Path | None) -> list[Path]:
    result: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            result.extend(sorted(path.glob("*.jsonl")))
            result.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            result.append(path)
    if standard_dir:
        result.extend(p for p in [standard_dir / "_nas_import_manifest.jsonl"] if p.is_file())
        result.extend(sorted(p for p in standard_dir.glob("_nas_import_manifest_*.jsonl") if p.is_file()))
        done_dir = standard_dir / "_nas_import_manifests_done"
        if done_dir.is_dir():
            result.extend(sorted(p for p in done_dir.glob("_nas_import_manifest*.jsonl") if p.is_file()))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in result:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def iter_manifest_rows(path: Path) -> Iterable[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
        elif isinstance(data, dict):
            items = data.get("items")
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        yield item
            else:
                yield data
        return
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
        except Exception:
            continue
        if isinstance(item, dict):
            yield item


def summarize_store(store: CatalogStore) -> Counter:
    data = store.list_products(limit=1000000, offset=0, include_images=False, exclude_owner=True)
    rows = data.get("products", data) if isinstance(data, dict) else data
    summary: Counter = Counter()
    for item in rows:
        groups = item.get("tag_groups") or {}
        if not (groups.get("year") or []):
            summary["missing_year"] += 1
        if not (groups.get("category") or []):
            summary["missing_category"] += 1
        subs = [x for x in (groups.get("subcategory") or []) if x and x != "暂无"]
        if not subs:
            summary["missing_subcategory"] += 1
    summary["products"] = len(rows)
    return summary


def existing_style_codes(db_path: Path) -> set[str]:
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute("SELECT style_code FROM products").fetchall()
    return {str(row[0]).strip() for row in rows if str(row[0]).strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill catalog year/category/subcategory tags from import manifests.")
    parser.add_argument("--db", required=True, help="Path to product_catalog.db")
    parser.add_argument("--standard-dir", help="Path to standard_samples. Used to find _nas_import_manifest*.jsonl")
    parser.add_argument("--manifest", action="append", default=[], help="Manifest json/jsonl file or directory. Can be repeated")
    parser.add_argument("--apply", action="store_true", help="Write tags to the database. Omit for dry-run")
    parser.add_argument("--no-backup", action="store_true", help="Do not create a .bak file before --apply")
    args = parser.parse_args()

    db_path = Path(args.db).expanduser()
    standard_dir = Path(args.standard_dir).expanduser() if args.standard_dir else None
    manifests = discover_manifests(args.manifest, standard_dir)
    if not manifests:
        print("未找到 manifest/jsonl。请检查 standard_samples 下是否有 _nas_import_manifest*.jsonl 或 _nas_import_manifests_done/")
        return 2

    store = CatalogStore(db_path)
    before = summarize_store(store)
    existing_codes = existing_style_codes(db_path)
    style_tags: dict[str, list[str]] = defaultdict(list)
    stats: Counter = Counter()
    examples: list[tuple[str, list[str], str]] = []

    for manifest in manifests:
        stats["manifest_files"] += 1
        for item in iter_manifest_rows(manifest):
            stats["rows"] += 1
            image_name = Path(str(item.get("target_filename") or item.get("image_name") or "")).name
            style_code = str(item.get("style_code") or "").strip() or filename_to_style_code(image_name).strip()
            if not style_code:
                stats["missing_style_code"] += 1
                continue
            tags = normalize_tags([*tags_from_manifest_item(item), *tags_from_source_rel_path(str(item.get("source_rel_path") or ""))])
            wanted = [
                tag
                for tag in tags
                if parse_catalog_tag(tag).get("type") in {"year", "category", "subcategory"}
            ]
            if not wanted:
                stats["rows_without_tags"] += 1
                continue
            style_tags[style_code].extend(wanted)
            if len(examples) < 10:
                examples.append((style_code, wanted, str(item.get("source_rel_path") or image_name)))

    all_style_tags = {code: normalize_tags(tags) for code, tags in style_tags.items()}
    normalized_style_tags = {code: tags for code, tags in all_style_tags.items() if code in existing_codes}
    tag_count = sum(len(tags) for tags in normalized_style_tags.values())
    print("模式:", "写入数据库" if args.apply else "预览，不写入")
    print("manifest 文件数:", stats["manifest_files"])
    print("manifest 行数:", stats["rows"])
    print("manifest 中有标签款数:", len(all_style_tags))
    print("当前库可补标签款数:", len(normalized_style_tags))
    print("可补标签条数:", tag_count)
    print("补前缺年份/类别/细类:", before["missing_year"], before["missing_category"], before["missing_subcategory"])
    print("示例:")
    for code, tags, source in examples:
        print(f"  {code}: {', '.join(tags)}  <- {source}")

    if not args.apply:
        print("确认示例无误后，加 --apply 执行写入。")
        return 0

    if not args.no_backup:
        backup = db_path.with_name(f"{db_path.name}.bak_{time.strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(db_path, backup)
        print("已备份:", backup)

    for code, tags in sorted(normalized_style_tags.items()):
        store.add_product_tags(code, tags)
    after = summarize_store(store)
    print("已写入。")
    print("补后缺年份/类别/细类:", after["missing_year"], after["missing_category"], after["missing_subcategory"])
    mapping = store.list_subcategories_by_category()
    print("类别-细类关系:")
    for category in sorted(mapping):
        if mapping[category]:
            print(f"  {category} => {', '.join(mapping[category])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
