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
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.catalog_store import CatalogStore, make_typed_tag, parse_catalog_tag

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TAG_TYPES = {"year", "category", "subcategory"}


def split_dir_tags(value: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in re.split(r"[、,，\s]+", str(value or "")):
        clean = raw.strip()
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


def clean_style_code_from_filename(path: Path) -> str:
    stem = path.stem.strip()
    stem = re.sub(r"\s*[\(（]\d+[\)）]\s*$", "", stem).strip()
    stem = re.sub(r"[#＃]+$", "", stem).strip()
    stem = re.sub(r"(?:_\d{3})+$", "", stem).strip()
    stem = re.sub(r"[\s_]+", "-", stem).strip("-")
    return stem


def discover_year_dirs(year_parent_dir: Path) -> list[Path]:
    root = year_parent_dir.expanduser().resolve()
    if re.fullmatch(r"20\d{2}", root.name):
        return [root]
    return sorted(path for path in root.iterdir() if path.is_dir() and re.fullmatch(r"20\d{2}", path.name))


def tags_from_image_path(year_dir: Path, image_path: Path) -> list[str]:
    try:
        rel_parent = image_path.parent.relative_to(year_dir)
    except Exception:
        return []
    parts = [part.strip() for part in rel_parent.parts if str(part).strip()]
    tags: list[str] = [make_typed_tag("year", year_dir.name)]
    if len(parts) >= 1:
        tags.append(make_typed_tag("category", parts[0]))
    if len(parts) >= 2:
        subcategory_tags: list[str] = []
        for part in parts[1:]:
            subcategory_tags.extend(split_dir_tags(part))
        tags.extend(make_typed_tag("subcategory", tag) for tag in subcategory_tags)
    return normalize_tags(tag for tag in tags if tag)


def scan_year_parent_dir(year_parent_dir: Path) -> tuple[dict[str, list[str]], dict[str, list[str]], Counter]:
    base_dir = year_parent_dir.expanduser().resolve()
    style_tags: dict[str, list[str]] = defaultdict(list)
    style_examples: dict[str, list[str]] = defaultdict(list)
    stats: Counter = Counter()
    for year_dir in discover_year_dirs(base_dir):
        stats["year_dirs"] += 1
        for path in sorted(year_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            stats["images"] += 1
            style_code = clean_style_code_from_filename(path)
            if not style_code:
                stats["missing_style_code"] += 1
                continue
            tags = tags_from_image_path(year_dir, path)
            if not any(parse_catalog_tag(tag).get("type") == "category" for tag in tags):
                stats["missing_category_from_path"] += 1
            if not any(parse_catalog_tag(tag).get("type") == "subcategory" for tag in tags):
                stats["missing_subcategory_from_path"] += 1
            style_tags[style_code].extend(tags)
            if len(style_examples[style_code]) < 3:
                try:
                    rel = str(path.relative_to(base_dir)).replace("\\", "/")
                except Exception:
                    rel = str(path)
                style_examples[style_code].append(rel)
    return {code: normalize_tags(tags) for code, tags in style_tags.items()}, style_examples, stats


def load_current_structured_tags(db_path: Path) -> dict[str, list[str]]:
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(
            """
            SELECT p.style_code, t.name
            FROM products p
            LEFT JOIN product_tags pt ON pt.style_code = p.style_code
            LEFT JOIN tags t ON t.id = pt.tag_id
            ORDER BY p.style_code ASC, t.name COLLATE NOCASE ASC
            """
        ).fetchall()
    result: dict[str, list[str]] = defaultdict(list)
    for code, tag in rows:
        style_code = str(code or "").strip()
        if not style_code:
            continue
        if tag is None:
            result.setdefault(style_code, [])
            continue
        clean = str(tag or "").strip()
        if parse_catalog_tag(clean).get("type") in TAG_TYPES:
            result[style_code].append(clean)
    return {code: normalize_tags(tags) for code, tags in result.items()}


def build_hierarchy(style_tags: dict[str, list[str]]) -> dict[str, list[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    for tags in style_tags.values():
        categories = []
        subcategories = []
        for tag in tags:
            parsed = parse_catalog_tag(tag)
            if parsed.get("type") == "category":
                categories.append(str(parsed.get("name") or "").strip())
            elif parsed.get("type") == "subcategory":
                name = str(parsed.get("name") or "").strip()
                if name and name != "暂无":
                    subcategories.append(name)
        for category in categories:
            for subcategory in subcategories:
                mapping[category].add(subcategory)
    return {category: sorted(values) for category, values in sorted(mapping.items())}


def write_tag_sync_jsonl(output_path: Path, style_tags: dict[str, list[str]], examples_by_code: dict[str, list[str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for code in sorted(style_tags):
            row = {
                "style_code": code,
                "tags": style_tags[code],
                "source_rel_path": (examples_by_code.get(code) or [""])[0],
                "source_examples": examples_by_code.get(code) or [],
                "generated_by": "sync_catalog_tags_from_year_dirs",
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_tag_sync_jsonl(paths: list[str]) -> tuple[dict[str, list[str]], dict[str, list[str]], Counter]:
    style_tags: dict[str, list[str]] = defaultdict(list)
    examples_by_code: dict[str, list[str]] = defaultdict(list)
    stats: Counter = Counter()
    for raw in paths:
        path = Path(raw).expanduser()
        candidates = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
        for candidate in candidates:
            stats["jsonl_files"] += 1
            for line in candidate.read_text(encoding="utf-8", errors="ignore").splitlines():
                text = line.strip()
                if not text:
                    continue
                stats["jsonl_rows"] += 1
                try:
                    item = json.loads(text)
                except Exception:
                    stats["invalid_json_rows"] += 1
                    continue
                code = str(item.get("style_code") or "").strip()
                if not code:
                    stats["missing_style_code"] += 1
                    continue
                tags = [
                    str(tag or "").strip()
                    for tag in (item.get("tags") or [])
                    if parse_catalog_tag(str(tag or "").strip()).get("type") in TAG_TYPES
                ]
                if not tags:
                    stats["rows_without_tags"] += 1
                    continue
                style_tags[code].extend(tags)
                examples = item.get("source_examples")
                if isinstance(examples, list):
                    for example in examples:
                        clean = str(example or "").strip()
                        if clean and clean not in examples_by_code[code] and len(examples_by_code[code]) < 3:
                            examples_by_code[code].append(clean)
                else:
                    source = str(item.get("source_rel_path") or "").strip()
                    if source and source not in examples_by_code[code]:
                        examples_by_code[code].append(source)
    return {code: normalize_tags(tags) for code, tags in style_tags.items()}, examples_by_code, stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan Windows/NAS year directories, export tag-sync JSONL, compare generated catalog tags with the DB, and optionally update tags."
    )
    parser.add_argument("--db", help="Path to product_catalog.db. Required for compare/apply, not required when only exporting JSONL on Windows")
    parser.add_argument("--year-parent-dir", help="Directory containing 2020/2021/... folders, or one year folder itself")
    parser.add_argument("--input-jsonl", action="append", default=[], help="Tag-sync JSONL file or directory generated on Windows. Can be repeated")
    parser.add_argument("--output-jsonl", help="Write scanned tags to this JSONL file without using the DB")
    parser.add_argument("--apply", action="store_true", help="Write changes. Omit for dry-run")
    parser.add_argument("--mode", choices=["replace", "add"], default="replace", help="replace updates year/category/subcategory; add only adds missing tags")
    parser.add_argument("--no-backup", action="store_true", help="Do not create a .bak file before --apply")
    parser.add_argument("--limit-examples", type=int, default=20, help="Number of changed style examples to print")
    args = parser.parse_args()

    if args.input_jsonl:
        scanned_tags, examples_by_code, scan_stats = load_tag_sync_jsonl(args.input_jsonl)
    else:
        if not args.year_parent_dir:
            parser.error("--year-parent-dir is required when --input-jsonl is not provided")
        year_parent_dir = Path(args.year_parent_dir).expanduser()
        scanned_tags, examples_by_code, scan_stats = scan_year_parent_dir(year_parent_dir)
    if args.output_jsonl:
        output_path = Path(args.output_jsonl).expanduser()
        write_tag_sync_jsonl(output_path, scanned_tags, examples_by_code)
        print("已生成标签同步 JSONL:", output_path)

    if not args.db:
        print("模式: 仅扫描/导出，不连接数据库")
        if args.input_jsonl:
            print("JSONL 文件数:", scan_stats["jsonl_files"])
            print("JSONL 行数:", scan_stats["jsonl_rows"])
        else:
            print("年份目录数:", scan_stats["year_dirs"])
            print("扫描图片数:", scan_stats["images"])
        print("扫描到款数:", len(scanned_tags))
        print("目录/JSONL 生成的类别-细类关系:")
        for category, subcategories in build_hierarchy(scanned_tags).items():
            print(f"  {category} => {', '.join(subcategories)}")
        if not args.output_jsonl:
            print("如需给服务器更新标签，请加 --output-jsonl 生成文件，再拷贝到服务器运行对比/更新。")
        return 0

    db_path = Path(args.db).expanduser()
    current_tags = load_current_structured_tags(db_path)
    current_codes = set(current_tags)
    matched_tags = {code: tags for code, tags in scanned_tags.items() if code in current_codes}
    unmatched_codes = sorted(set(scanned_tags) - current_codes)

    changes: list[tuple[str, list[str], list[str], list[str], list[str]]] = []
    warnings: Counter = Counter()
    for code, target in sorted(matched_tags.items()):
        current = current_tags.get(code, [])
        target_set = set(target)
        current_set = set(current)
        added = sorted(target_set - current_set)
        removed = sorted(current_set - target_set) if args.mode == "replace" else []
        if added or removed:
            changes.append((code, current, target, added, removed))
        groups = defaultdict(list)
        for tag in target:
            parsed = parse_catalog_tag(tag)
            if parsed.get("type") in TAG_TYPES:
                groups[parsed["type"]].append(parsed["name"])
        if len(groups["year"]) > 1:
            warnings["multi_year"] += 1
        if len(groups["category"]) > 1:
            warnings["multi_category"] += 1

    target_hierarchy = build_hierarchy(matched_tags)
    print("模式:", "写入数据库" if args.apply else "预览，不写入")
    print("更新方式:", "替换年份/类别/细类" if args.mode == "replace" else "只追加缺失标签")
    if args.input_jsonl:
        print("JSONL 文件数:", scan_stats["jsonl_files"])
        print("JSONL 行数:", scan_stats["jsonl_rows"])
    else:
        print("年份目录数:", scan_stats["year_dirs"])
        print("扫描图片数:", scan_stats["images"])
    print("扫描到款数:", len(scanned_tags))
    print("匹配当前库款数:", len(matched_tags))
    print("当前库未匹配扫描目录款数:", len(current_codes - set(scanned_tags)))
    print("扫描目录不在当前库款数:", len(unmatched_codes))
    print("会变更款数:", len(changes))
    print("路径缺类别图片数:", scan_stats["missing_category_from_path"])
    print("路径缺细类图片数:", scan_stats["missing_subcategory_from_path"])
    if warnings:
        print("提示:", dict(warnings))

    print("变更示例:")
    for code, current, target, added, removed in changes[: max(0, args.limit_examples)]:
        print(f"  {code}")
        print(f"    当前: {', '.join(current) or '(空)'}")
        print(f"    目录: {', '.join(target) or '(空)'}")
        print(f"    增加: {', '.join(added) or '(无)'}")
        if args.mode == "replace":
            print(f"    删除: {', '.join(removed) or '(无)'}")
        if examples_by_code.get(code):
            print(f"    来源: {examples_by_code[code][0]}")

    print("目录生成的类别-细类关系:")
    for category, subcategories in target_hierarchy.items():
        print(f"  {category} => {', '.join(subcategories)}")

    if not args.apply:
        print("确认无误后，加 --apply 执行更新。")
        return 0

    if not args.no_backup:
        backup = db_path.with_name(f"{db_path.name}.bak_{time.strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(db_path, backup)
        print("已备份:", backup)

    store = CatalogStore(db_path)
    for code, target in sorted(matched_tags.items()):
        if args.mode == "replace":
            store.replace_product_tags_by_types(code, target, TAG_TYPES)
        else:
            store.add_product_tags(code, target)
    print("已更新。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
