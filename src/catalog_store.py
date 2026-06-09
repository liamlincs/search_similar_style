import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List


def filename_to_style_code(img_name: str) -> str:
    stem = Path(img_name).stem
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


class CatalogStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
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
            if path.is_file() and path.suffix.lower() in allow
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

        return {
            "products_total": len(touched_products),
            "products_added": added_products,
            "images_added_or_updated": added_images,
        }

    def list_tags(self) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT name FROM tags ORDER BY name COLLATE NOCASE ASC").fetchall()
        return [str(row["name"]) for row in rows]

    def create_tag(self, name: str) -> str:
        tag = self._clean_tag(name)
        if not tag:
            raise ValueError("tag name is empty")
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO tags(name) VALUES (?)", (tag,))
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

    def get_product(self, style_code: str) -> Dict[str, Any] | None:
        rows = self.get_products_by_codes([style_code])
        return rows[0] if rows else None

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
    ) -> List[Dict[str, Any]]:
        filters: List[str] = []
        params: List[Any] = []
        query_code = style_code.strip()
        normalized_tags = self._normalize_tags(tags or [])
        if query_code:
            filters.append("p.style_code LIKE ?")
            params.append(f"%{query_code}%")

        from_clause = "FROM products p"
        if normalized_tags:
            placeholders = ",".join(["?"] * len(normalized_tags))
            from_clause += """
                JOIN (
                    SELECT pt.style_code
                    FROM product_tags pt
                    JOIN tags t ON t.id = pt.tag_id
                    WHERE t.name IN ({})
                    GROUP BY pt.style_code
                    HAVING COUNT(DISTINCT t.name) = ?
                ) matched ON matched.style_code = p.style_code
            """.format(placeholders)
            params.extend(normalized_tags)
            params.append(len(normalized_tags))

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        params.extend([max(1, min(int(limit), 500)), max(0, int(offset))])

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
                "tags": [],
            }
        for row in image_rows:
            style_code = str(row["style_code"])
            product = products.get(style_code)
            if product is None:
                continue
            product["images"].append(
                {
                    "image_name": str(row["image_name"]),
                    "sort_order": int(row["sort_order"] or 0),
                }
            )
        for row in tag_rows:
            style_code = str(row["style_code"])
            product = products.get(style_code)
            if product is None:
                continue
            product["tags"].append(str(row["name"]))
        return [products[code] for code in ordered_codes if code in products]

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
