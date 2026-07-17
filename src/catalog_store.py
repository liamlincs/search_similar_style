import sqlite3
import re
import datetime as dt
from pathlib import Path
from typing import Any, Dict, Iterable, List

TAG_TYPE_PREFIXES = {
    "year": "year:",
    "category": "category:",
    "subcategory": "subcategory:",
}
DEFAULT_CATEGORY_TAGS = ["单品", "罗纹", "毛织配件", "布匹"]
DEFAULT_SUBCATEGORY_TAGS = ["暂无"]
MIN_AUTO_YEAR_TAG = 2000
MAX_AUTO_YEAR_AHEAD = 3


def filename_to_style_code(img_name: str) -> str:
    stem = Path(img_name).stem
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


def make_typed_tag(kind: str, name: str) -> str:
    prefix = TAG_TYPE_PREFIXES.get(str(kind or "").strip())
    value = str(name or "").strip()
    if not prefix or not value:
        return ""
    return f"{prefix}{value}"


def parse_catalog_tag(tag: str) -> Dict[str, str]:
    raw = str(tag or "").strip()
    for kind, prefix in TAG_TYPE_PREFIXES.items():
        if raw.startswith(prefix):
            return {"type": kind, "name": raw[len(prefix):].strip(), "raw": raw}
    if re.fullmatch(r"20\d{2}", raw):
        return {"type": "year", "name": raw, "raw": raw}
    return {"type": "other", "name": raw, "raw": raw}


def derive_year_from_style_code(style_code: str) -> str:
    code = str(style_code or "").strip()
    if not code:
        return ""
    prefix = code.split("-", 1)[0].strip()
    match = re.search(r"(\d{2})$", prefix)
    if not match:
        return ""
    year = int(f"20{match.group(1)}")
    max_year = dt.date.today().year + MAX_AUTO_YEAR_AHEAD
    if year < MIN_AUTO_YEAR_TAG or year > max_year:
        return ""
    return str(year)


class CatalogStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS products (
                    style_code TEXT PRIMARY KEY,
                    cover_image TEXT NOT NULL DEFAULT '',
                    note TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS product_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    style_code TEXT NOT NULL,
                    image_name TEXT NOT NULL UNIQUE,
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(style_code) REFERENCES products(style_code) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS product_tags (
                    style_code TEXT NOT NULL,
                    tag_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(style_code, tag_id),
                    FOREIGN KEY(style_code) REFERENCES products(style_code) ON DELETE CASCADE,
                    FOREIGN KEY(tag_id) REFERENCES tags(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_product_images_style_code
                ON product_images(style_code, sort_order, image_name);

                CREATE INDEX IF NOT EXISTS idx_product_tags_tag_id
                ON product_tags(tag_id, style_code);
                """
            )

    def sync_from_standard_dir(self, standard_dir: Path, image_exts: Iterable[str]) -> Dict[str, int]:
        allow = {f".{str(ext).lower().lstrip('.')}" for ext in image_exts}
        image_files = [
            path for path in sorted(Path(standard_dir).glob("*"))
            if path.is_file() and path.suffix.lower() in allow and not path.name.startswith("MY-")
        ]

        added_products = 0
        added_images = 0
        touched_products = set()

        with self._connect() as conn:
            for path in image_files:
                style_code = filename_to_style_code(path.name).strip()
                if not style_code:
                    continue
                cur = conn.execute(
                    """
                    INSERT INTO products(style_code, cover_image)
                    VALUES (?, ?)
                    ON CONFLICT(style_code) DO NOTHING
                    """,
                    (style_code, path.name),
                )
                if cur.rowcount > 0:
                    added_products += 1

                cur = conn.execute(
                    """
                    INSERT INTO product_images(style_code, image_name, sort_order)
                    VALUES (?, ?, ?)
                    ON CONFLICT(image_name) DO UPDATE SET
                        style_code=excluded.style_code,
                        sort_order=excluded.sort_order
                    """,
                    (style_code, path.name, self._infer_sort_order(path.name)),
                )
                if cur.rowcount > 0:
                    added_images += 1
                touched_products.add(style_code)

            for style_code in touched_products:
                cover = conn.execute(
                    """
                    SELECT image_name
                    FROM product_images
                    WHERE style_code=?
                    ORDER BY sort_order ASC, image_name ASC
                    LIMIT 1
                    """,
                    (style_code,),
                ).fetchone()
                if cover:
                    conn.execute(
                        """
                        UPDATE products
                        SET cover_image=?, updated_at=CURRENT_TIMESTAMP
                        WHERE style_code=?
                        """,
                        (str(cover["image_name"]), style_code),
                    )
            conn.commit()

        year_tags_added = 0
        for style_code in sorted(touched_products):
            year = derive_year_from_style_code(style_code)
            typed_year = make_typed_tag("year", year)
            if not typed_year:
                continue
            product_before = self.get_product(style_code)
            before = set(product_before.get("raw_tags", []) if product_before else [])
            if typed_year in before:
                continue
            self.add_product_tags(style_code, [typed_year])
            product_after = self.get_product(style_code)
            after = set(product_after.get("raw_tags", []) if product_after else [])
            if typed_year not in before and typed_year in after:
                year_tags_added += 1

        return {
            "products_total": len(touched_products),
            "products_added": added_products,
            "images_added_or_updated": added_images,
            "year_tags_added": year_tags_added,
        }

    def list_tags(self) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT name FROM tags ORDER BY name COLLATE NOCASE ASC").fetchall()
        return [str(row["name"]) for row in rows]

    def list_used_tags(self) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT t.name
                FROM tags t
                JOIN product_tags pt ON pt.tag_id = t.id
                ORDER BY t.name COLLATE NOCASE ASC
                """
            ).fetchall()
        return [str(row["name"]) for row in rows]

    def list_tag_groups(self) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {
            "year": [],
            "category": list(DEFAULT_CATEGORY_TAGS),
            "subcategory": list(DEFAULT_SUBCATEGORY_TAGS),
        }
        for raw in self.list_tags():
            parsed = parse_catalog_tag(raw)
            kind = parsed["type"]
            name = parsed["name"]
            if kind not in groups or not name:
                continue
            if name not in groups[kind]:
                groups[kind].append(name)
        groups["year"] = sorted(groups["year"], key=lambda item: str(item).lower())
        return groups

    def create_tag(self, name: str) -> str:
        tag = self._clean_tag(name)
        if not tag:
            raise ValueError("tag name is empty")
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO tags(name) VALUES (?)", (tag,))
            conn.commit()
        return tag

    def delete_tag(self, name: str) -> str:
        tag = self._clean_tag(name)
        if not tag:
            raise ValueError("tag name is empty")
        with self._connect() as conn:
            conn.execute("DELETE FROM tags WHERE name=?", (tag,))
            conn.commit()
        return tag

    def replace_product_tags(self, style_code: str, tags: Iterable[str]) -> List[str]:
        code = style_code.strip()
        if not code:
            raise ValueError("style_code is empty")
        cleaned = self._normalize_tags(tags)
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM products WHERE style_code=? LIMIT 1",
                (code,),
            ).fetchone()
            if not exists:
                raise ValueError("style_code not found")
            for tag in cleaned:
                conn.execute("INSERT OR IGNORE INTO tags(name) VALUES (?)", (tag,))
            conn.execute("DELETE FROM product_tags WHERE style_code=?", (code,))
            for tag in cleaned:
                tag_row = conn.execute("SELECT id FROM tags WHERE name=?", (tag,)).fetchone()
                if tag_row:
                    conn.execute(
                        "INSERT OR IGNORE INTO product_tags(style_code, tag_id) VALUES (?, ?)",
                        (code, int(tag_row["id"])),
                    )
            conn.execute(
                "UPDATE products SET updated_at=CURRENT_TIMESTAMP WHERE style_code=?",
                (code,),
            )
            conn.commit()
        return cleaned

    def add_product_tags(self, style_code: str, tags: Iterable[str]) -> List[str]:
        code = style_code.strip()
        if not code:
            raise ValueError("style_code is empty")
        extra_tags = self._normalize_tags(tags)
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM products WHERE style_code=? LIMIT 1",
                (code,),
            ).fetchone()
            if not exists:
                raise ValueError("style_code not found")
            current_rows = conn.execute(
                """
                SELECT t.name
                FROM product_tags pt
                JOIN tags t ON t.id = pt.tag_id
                WHERE pt.style_code=?
                ORDER BY t.name COLLATE NOCASE ASC
                """,
                (code,),
            ).fetchall()
            current_tags = [str(row["name"]) for row in current_rows]
        merged = self._normalize_tags([*current_tags, *extra_tags])
        return self.replace_product_tags(code, merged)

    def upsert_product(
        self,
        style_code: str,
        image_names: Iterable[str],
        tags: Iterable[str] | None = None,
        note: str = "",
    ) -> Dict[str, Any]:
        code = style_code.strip()
        if not code:
            raise ValueError("style_code is empty")
        images = [str(name).strip() for name in image_names if str(name).strip()]
        if not images:
            raise ValueError("image_names is empty")
        cleaned_tags = self._normalize_tags(tags or [])
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO products(style_code, cover_image, note)
                VALUES (?, ?, ?)
                ON CONFLICT(style_code) DO UPDATE SET
                    note=excluded.note,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (code, images[0], str(note or "")),
            )
            existing = conn.execute(
                """
                SELECT COALESCE(MAX(sort_order), -1) AS max_sort
                FROM product_images
                WHERE style_code=?
                """,
                (code,),
            ).fetchone()
            next_sort = int(existing["max_sort"] if existing and existing["max_sort"] is not None else -1) + 1
            for image_name in images:
                conn.execute(
                    """
                    INSERT INTO product_images(style_code, image_name, sort_order)
                    VALUES (?, ?, ?)
                    ON CONFLICT(image_name) DO UPDATE SET
                        style_code=excluded.style_code
                    """,
                    (code, image_name, next_sort),
                )
                next_sort += 1
            cover = conn.execute(
                """
                SELECT image_name
                FROM product_images
                WHERE style_code=?
                ORDER BY sort_order ASC, image_name ASC
                LIMIT 1
                """,
                (code,),
            ).fetchone()
            if cover:
                conn.execute(
                    "UPDATE products SET cover_image=?, updated_at=CURRENT_TIMESTAMP WHERE style_code=?",
                    (str(cover["image_name"]), code),
                )
            for tag in cleaned_tags:
                conn.execute("INSERT OR IGNORE INTO tags(name) VALUES (?)", (tag,))
                tag_row = conn.execute("SELECT id FROM tags WHERE name=?", (tag,)).fetchone()
                if tag_row:
                    conn.execute(
                        "INSERT OR IGNORE INTO product_tags(style_code, tag_id) VALUES (?, ?)",
                        (code, int(tag_row["id"])),
                    )
            conn.commit()
        product = self.get_product(code)
        if not product:
            raise ValueError("product upsert failed")
        return product

    def get_product(self, style_code: str) -> Dict[str, Any] | None:
        rows = self.get_products_by_codes([style_code])
        return rows[0] if rows else None

    def delete_product(self, style_code: str) -> List[str]:
        code = style_code.strip()
        if not code:
            raise ValueError("style_code is empty")
        with self._connect() as conn:
            image_rows = conn.execute(
                "SELECT image_name FROM product_images WHERE style_code=?",
                (code,),
            ).fetchall()
            exists = conn.execute("SELECT 1 FROM products WHERE style_code=? LIMIT 1", (code,)).fetchone()
            if not exists:
                raise ValueError("style_code not found")
            conn.execute("DELETE FROM products WHERE style_code=?", (code,))
            conn.commit()
        return [str(row["image_name"]) for row in image_rows]

    def delete_product_image(self, style_code: str, image_name: str) -> Dict[str, Any]:
        code = style_code.strip()
        name = image_name.strip()
        if not code:
            raise ValueError("style_code is empty")
        if not name:
            raise ValueError("image_name is empty")
        with self._connect() as conn:
            exists = conn.execute("SELECT 1 FROM products WHERE style_code=? LIMIT 1", (code,)).fetchone()
            if not exists:
                raise ValueError("style_code not found")
            image_row = conn.execute(
                "SELECT image_name FROM product_images WHERE style_code=? AND image_name=? LIMIT 1",
                (code, name),
            ).fetchone()
            if not image_row:
                raise ValueError("image_name not found")
            conn.execute("DELETE FROM product_images WHERE style_code=? AND image_name=?", (code, name))
            remaining_rows = conn.execute(
                """
                SELECT image_name
                FROM product_images
                WHERE style_code=?
                ORDER BY sort_order ASC, image_name ASC
                """,
                (code,),
            ).fetchall()
            remaining = [str(row["image_name"]) for row in remaining_rows]
            if remaining:
                conn.execute(
                    "UPDATE products SET cover_image=?, updated_at=CURRENT_TIMESTAMP WHERE style_code=?",
                    (remaining[0], code),
                )
            else:
                conn.execute("DELETE FROM products WHERE style_code=?", (code,))
            conn.commit()
        return {"deleted": name, "remaining": remaining}

    def get_products_by_codes(self, style_codes: Iterable[str]) -> List[Dict[str, Any]]:
        codes = [str(code).strip() for code in style_codes if str(code).strip()]
        if not codes:
            return []
        placeholders = ",".join(["?"] * len(codes))
        with self._connect() as conn:
            product_rows = conn.execute(
                f"""
                SELECT style_code, cover_image, note, created_at, updated_at
                FROM products
                WHERE style_code IN ({placeholders})
                ORDER BY style_code ASC
                """,
                codes,
            ).fetchall()
            image_rows = conn.execute(
                f"""
                SELECT style_code, image_name, sort_order
                FROM product_images
                WHERE style_code IN ({placeholders})
                ORDER BY style_code ASC, sort_order ASC, image_name ASC
                """,
                codes,
            ).fetchall()
            tag_rows = conn.execute(
                f"""
                SELECT pt.style_code, t.name
                FROM product_tags pt
                JOIN tags t ON t.id = pt.tag_id
                WHERE pt.style_code IN ({placeholders})
                ORDER BY pt.style_code ASC, t.name COLLATE NOCASE ASC
                """,
                codes,
            ).fetchall()
        return self._assemble_products(product_rows, image_rows, tag_rows)

    def list_products(
        self,
        style_code: str = "",
        tags: Iterable[str] | None = None,
        limit: int = 200,
        offset: int = 0,
        exclude_owner: bool = False,
        include_images: bool = True,
    ) -> List[Dict[str, Any]]:
        filters: List[str] = []
        filter_params: List[Any] = []
        join_params: List[Any] = []
        query_code = style_code.strip()
        normalized_tags = self._normalize_tags(tags or [])
        if query_code:
            filters.append(
                """
                (
                    UPPER(p.style_code) LIKE UPPER(?)
                    OR (
                        INSTR(p.style_code, '-') > 0
                        AND UPPER(SUBSTR(p.style_code, INSTR(p.style_code, '-') + 1)) LIKE UPPER(?)
                    )
                )
                """
            )
            filter_params.extend([f"%{query_code}%", f"%{query_code}%"])
        if exclude_owner:
            filters.append(
                """
                (
                    p.style_code NOT LIKE 'MY-%'
                    AND NOT EXISTS (
                        SELECT 1
                        FROM product_tags opt
                        JOIN tags ot ON ot.id = opt.tag_id
                        WHERE opt.style_code = p.style_code
                          AND ot.name LIKE 'owner:%'
                    )
                )
                """
            )

        from_clause = "FROM products p"
        tag_match_groups = self._tag_match_groups(normalized_tags)
        if tag_match_groups:
            where_parts = []
            having_parts = []
            for index, alternatives in enumerate(tag_match_groups):
                placeholders = ",".join(["?"] * len(alternatives))
                where_parts.append(f"t.name IN ({placeholders})")
                join_params.extend(alternatives)
                having_parts.append(f"SUM(CASE WHEN t.name IN ({placeholders}) THEN 1 ELSE 0 END) > 0")
            for alternatives in tag_match_groups:
                join_params.extend(alternatives)
            from_clause += """
                JOIN (
                    SELECT pt.style_code
                    FROM product_tags pt
                    JOIN tags t ON t.id = pt.tag_id
                    WHERE {}
                    GROUP BY pt.style_code
                    HAVING {}
                ) matched ON matched.style_code = p.style_code
            """.format(" OR ".join(where_parts), " AND ".join(having_parts))

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        params = [
            *join_params,
            *filter_params,
            max(1, min(int(limit), 500)),
            max(0, int(offset)),
        ]

        with self._connect() as conn:
            product_rows = conn.execute(
                f"""
                SELECT p.style_code, p.cover_image, p.note, p.created_at, p.updated_at
                {from_clause}
                {where_clause}
                ORDER BY p.style_code DESC
                LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()
            if not product_rows:
                return []
            codes = [str(row["style_code"]) for row in product_rows]
            placeholders = ",".join(["?"] * len(codes))
            if include_images:
                image_rows = conn.execute(
                    f"""
                    SELECT style_code, image_name, sort_order
                    FROM product_images
                    WHERE style_code IN ({placeholders})
                    ORDER BY style_code ASC, sort_order ASC, image_name ASC
                    """,
                    codes,
                ).fetchall()
            else:
                image_rows = conn.execute(
                    f"""
                    SELECT style_code, COUNT(*) AS image_count
                    FROM product_images
                    WHERE style_code IN ({placeholders})
                    GROUP BY style_code
                    """,
                    codes,
                ).fetchall()
            tag_rows = conn.execute(
                f"""
                SELECT pt.style_code, t.name
                FROM product_tags pt
                JOIN tags t ON t.id = pt.tag_id
                WHERE pt.style_code IN ({placeholders})
                ORDER BY pt.style_code ASC, t.name COLLATE NOCASE ASC
                """,
                codes,
            ).fetchall()
        return self._assemble_products(product_rows, image_rows, tag_rows)

    def _assemble_products(
        self,
        product_rows: Iterable[sqlite3.Row],
        image_rows: Iterable[sqlite3.Row],
        tag_rows: Iterable[sqlite3.Row],
    ) -> List[Dict[str, Any]]:
        products: Dict[str, Dict[str, Any]] = {}
        ordered_codes: List[str] = []
        for row in product_rows:
            style_code = str(row["style_code"])
            ordered_codes.append(style_code)
            products[style_code] = {
                "style_code": style_code,
                "cover_image": str(row["cover_image"] or ""),
                "note": str(row["note"] or ""),
                "created_at": str(row["created_at"] or ""),
                "updated_at": str(row["updated_at"] or ""),
                "images": [],
                "image_count": 0,
                "tags": [],
                "raw_tags": [],
                "tag_groups": {
                    "year": [],
                    "category": [],
                    "subcategory": [],
                },
            }
        for row in image_rows:
            style_code = str(row["style_code"])
            product = products.get(style_code)
            if product is None:
                continue
            if "image_count" in row.keys() and "image_name" not in row.keys():
                product["image_count"] = int(row["image_count"] or 0)
                continue
            product["images"].append(
                {
                    "image_name": str(row["image_name"]),
                    "sort_order": int(row["sort_order"] or 0),
                }
            )
            product["image_count"] += 1
        for row in tag_rows:
            style_code = str(row["style_code"])
            product = products.get(style_code)
            if product is None:
                continue
            raw = str(row["name"])
            parsed = parse_catalog_tag(raw)
            kind = parsed["type"]
            name = parsed["name"]
            if not name:
                continue
            product["raw_tags"].append(raw)
            if kind in product["tag_groups"]:
                if name not in product["tag_groups"][kind]:
                    product["tag_groups"][kind].append(name)
                if name not in product["tags"]:
                    product["tags"].append(name)
        return [products[code] for code in ordered_codes if code in products]

    def _tag_match_groups(self, tags: Iterable[str]) -> List[List[str]]:
        groups: List[List[str]] = []
        for tag in tags:
            clean = self._clean_tag(tag)
            if not clean:
                continue
            parsed = parse_catalog_tag(clean)
            alternatives = [clean]
            if parsed["type"] == "year":
                typed = make_typed_tag("year", parsed["name"])
                if typed and typed not in alternatives:
                    alternatives.append(typed)
                if parsed["name"] and parsed["name"] not in alternatives:
                    alternatives.append(parsed["name"])
            groups.append(alternatives)
        return groups

    def _normalize_tags(self, tags: Iterable[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for tag in tags:
            clean = self._clean_tag(tag)
            if not clean:
                continue
            if clean in seen:
                continue
            seen.add(clean)
            out.append(clean)
        return out

    def _clean_tag(self, tag: str) -> str:
        return str(tag or "").strip()

    def _infer_sort_order(self, image_name: str) -> int:
        stem = Path(image_name).stem
        if "_" not in stem:
            return 0
        suffix = stem.rsplit("_", 1)[-1]
        return int(suffix) if suffix.isdigit() else 0
