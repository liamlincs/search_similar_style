#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove one catalog tag from products.")
    parser.add_argument("--db", required=True, help="Path to product_catalog.db")
    parser.add_argument("--tag", required=True, help="Exact tag name, e.g. category:单品")
    parser.add_argument("--apply", action="store_true", help="Write changes. Omit for dry-run")
    parser.add_argument("--no-backup", action="store_true", help="Do not create a .bak file before --apply")
    args = parser.parse_args()

    db_path = Path(args.db).expanduser()
    tag = str(args.tag or "").strip()
    if not tag:
        print("tag 不能为空")
        return 2

    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(
            """
            SELECT p.style_code
            FROM products p
            JOIN product_tags pt ON pt.style_code = p.style_code
            JOIN tags t ON t.id = pt.tag_id
            WHERE t.name = ?
            ORDER BY p.style_code ASC
            """,
            (tag,),
        ).fetchall()
        codes = [str(row[0]) for row in rows]

    print("模式:", "写入数据库" if args.apply else "预览，不写入")
    print("标签:", tag)
    print("涉及款数:", len(codes))
    print("示例:", codes[:30])
    if not args.apply:
        print("确认无误后，加 --apply 执行删除。")
        return 0

    if not args.no_backup:
        backup = db_path.with_name(f"{db_path.name}.bak_{time.strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(db_path, backup)
        print("已备份:", backup)

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            DELETE FROM product_tags
            WHERE tag_id IN (SELECT id FROM tags WHERE name = ?)
            """,
            (tag,),
        )
        conn.execute(
            """
            DELETE FROM tags
            WHERE name = ?
              AND id NOT IN (SELECT DISTINCT tag_id FROM product_tags)
            """,
            (tag,),
        )
        conn.commit()
    print("已删除。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
