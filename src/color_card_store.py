import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List


def clean_library_id(value: str) -> str:
    raw = str(value or "").strip().lower()
    raw = "".join(ch if ch.isalnum() else "_" for ch in raw)
    raw = "_".join(part for part in raw.split("_") if part)
    return raw[:80]


def lab_to_rgb_hex(l: float, a: float, b: float) -> str:
    y = (float(l) + 16.0) / 116.0
    x = float(a) / 500.0 + y
    z = y - float(b) / 200.0

    def pivot(v: float) -> float:
        return v ** 3 if v > 6 / 29 else (v - 16 / 116) / 7.787

    x = pivot(x) * 0.95047
    y = pivot(y)
    z = pivot(z) * 1.08883

    r = 3.2406 * x - 1.5372 * y - 0.4986 * z
    g = -0.9689 * x + 1.8758 * y + 0.0415 * z
    bl = 0.0557 * x - 0.2040 * y + 1.0570 * z

    def gamma(v: float) -> int:
        v = 1.055 * (v ** (1 / 2.4)) - 0.055 if v > 0.0031308 else 12.92 * v
        return max(0, min(255, round(v * 255)))

    return f"{gamma(r):02X}{gamma(g):02X}{gamma(bl):02X}"


def delta_e_ciede2000(lab1: tuple[float, float, float], lab2: tuple[float, float, float]) -> float:
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2
    c1 = math.sqrt(a1 * a1 + b1 * b1)
    c2 = math.sqrt(a2 * a2 + b2 * b2)
    c_bar = (c1 + c2) / 2.0
    g = 0.5 * (1 - math.sqrt((c_bar ** 7) / ((c_bar ** 7) + (25 ** 7)))) if c_bar else 0.0
    a1p = (1 + g) * a1
    a2p = (1 + g) * a2
    c1p = math.sqrt(a1p * a1p + b1 * b1)
    c2p = math.sqrt(a2p * a2p + b2 * b2)

    def hp(ap: float, bp: float) -> float:
        if ap == 0 and bp == 0:
            return 0.0
        angle = math.degrees(math.atan2(bp, ap))
        return angle + 360 if angle < 0 else angle

    h1p = hp(a1p, b1)
    h2p = hp(a2p, b2)
    dlp = l2 - l1
    dcp = c2p - c1p
    if c1p * c2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    elif h2p <= h1p:
        dhp = h2p - h1p + 360
    else:
        dhp = h2p - h1p - 360
    dhp_term = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(dhp / 2.0))

    l_bar_p = (l1 + l2) / 2.0
    c_bar_p = (c1p + c2p) / 2.0
    if c1p * c2p == 0:
        h_bar_p = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        h_bar_p = (h1p + h2p) / 2.0
    elif h1p + h2p < 360:
        h_bar_p = (h1p + h2p + 360) / 2.0
    else:
        h_bar_p = (h1p + h2p - 360) / 2.0

    t = (
        1
        - 0.17 * math.cos(math.radians(h_bar_p - 30))
        + 0.24 * math.cos(math.radians(2 * h_bar_p))
        + 0.32 * math.cos(math.radians(3 * h_bar_p + 6))
        - 0.20 * math.cos(math.radians(4 * h_bar_p - 63))
    )
    delta_theta = 30 * math.exp(-(((h_bar_p - 275) / 25) ** 2))
    rc = 2 * math.sqrt((c_bar_p ** 7) / ((c_bar_p ** 7) + (25 ** 7))) if c_bar_p else 0.0
    sl = 1 + (0.015 * ((l_bar_p - 50) ** 2)) / math.sqrt(20 + ((l_bar_p - 50) ** 2))
    sc = 1 + 0.045 * c_bar_p
    sh = 1 + 0.015 * c_bar_p * t
    rt = -math.sin(math.radians(2 * delta_theta)) * rc
    return math.sqrt(
        (dlp / sl) ** 2
        + (dcp / sc) ** 2
        + (dhp_term / sh) ** 2
        + rt * (dcp / sc) * (dhp_term / sh)
    )


class ColorCardStore:
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
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS color_libraries (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    source_file TEXT NOT NULL DEFAULT '',
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS color_cards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    library_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    illuminant TEXT NOT NULL DEFAULT '',
                    angle REAL,
                    l REAL NOT NULL,
                    a REAL NOT NULL,
                    b REAL NOT NULL,
                    hex TEXT NOT NULL DEFAULT '',
                    spectral_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(library_id, name),
                    FOREIGN KEY(library_id) REFERENCES color_libraries(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_color_cards_library_lab
                ON color_cards(library_id, l, a, b);
                """
            )

    def replace_library(self, library_id: str, name: str, source_file: str, rows: Iterable[Dict[str, Any]], sort_order: int = 0) -> int:
        rows_list = list(rows)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO color_libraries(id, name, source_file, sort_order)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    source_file=excluded.source_file,
                    sort_order=excluded.sort_order
                """,
                (library_id, name, source_file, int(sort_order)),
            )
            conn.execute("DELETE FROM color_cards WHERE library_id=?", (library_id,))
            for row in rows_list:
                l_val = float(row["l"])
                a_val = float(row["a"])
                b_val = float(row["b"])
                conn.execute(
                    """
                    INSERT INTO color_cards(
                        library_id, name, note, illuminant, angle, l, a, b, hex, spectral_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        library_id,
                        str(row.get("name", "")).strip(),
                        str(row.get("note", "") or "").strip(),
                        str(row.get("illuminant", "") or "").strip(),
                        row.get("angle"),
                        l_val,
                        a_val,
                        b_val,
                        str(row.get("hex") or lab_to_rgb_hex(l_val, a_val, b_val)).upper(),
                        json.dumps(row.get("spectral") or [], ensure_ascii=False),
                    ),
                )
            conn.commit()
        return len(rows_list)

    def list_libraries(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT l.id, l.name, l.source_file, COUNT(c.id) AS color_count
                FROM color_libraries l
                LEFT JOIN color_cards c ON c.library_id=l.id
                GROUP BY l.id
                ORDER BY l.sort_order ASC, l.name COLLATE NOCASE ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_library(self, library_id: str, name: str) -> Dict[str, Any]:
        clean_id = clean_library_id(library_id) or clean_library_id(name)
        clean_name = str(name or "").strip()
        if not clean_id:
            raise ValueError("library_id is empty")
        if not clean_name:
            raise ValueError("library name is empty")
        with self._connect() as conn:
            max_order = conn.execute("SELECT COALESCE(MAX(sort_order), 0) AS max_order FROM color_libraries").fetchone()
            conn.execute(
                """
                INSERT INTO color_libraries(id, name, source_file, sort_order)
                VALUES (?, ?, 'manual', ?)
                ON CONFLICT(id) DO UPDATE SET name=excluded.name
                """,
                (clean_id, clean_name, int(max_order["max_order"] or 0) + 1),
            )
            conn.commit()
        return {"id": clean_id, "name": clean_name}

    def upsert_card(
        self,
        *,
        library_id: str,
        library_name: str = "",
        name: str,
        note: str = "",
        illuminant: str = "D65",
        angle: float | None = 10,
        l: float,
        a: float,
        b: float,
        spectral: list[Any] | None = None,
    ) -> Dict[str, Any]:
        clean_lib_id = clean_library_id(library_id) or clean_library_id(library_name)
        clean_name = str(name or "").strip()
        if not clean_lib_id:
            raise ValueError("library_id is empty")
        if not clean_name:
            raise ValueError("color name is empty")
        lib_name = str(library_name or clean_lib_id).strip()
        self.upsert_library(clean_lib_id, lib_name)
        l_val = float(l)
        a_val = float(a)
        b_val = float(b)
        hex_value = lab_to_rgb_hex(l_val, a_val, b_val)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO color_cards(
                    library_id, name, note, illuminant, angle, l, a, b, hex, spectral_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(library_id, name) DO UPDATE SET
                    note=excluded.note,
                    illuminant=excluded.illuminant,
                    angle=excluded.angle,
                    l=excluded.l,
                    a=excluded.a,
                    b=excluded.b,
                    hex=excluded.hex,
                    spectral_json=excluded.spectral_json
                """,
                (
                    clean_lib_id,
                    clean_name,
                    str(note or "").strip(),
                    str(illuminant or "").strip(),
                    angle,
                    l_val,
                    a_val,
                    b_val,
                    hex_value,
                    json.dumps(spectral or [], ensure_ascii=False),
                ),
            )
            row = conn.execute(
                """
                SELECT c.id, c.library_id, l.name AS library_name, c.name, c.note,
                       c.illuminant, c.angle, c.l, c.a, c.b, c.hex
                FROM color_cards c
                JOIN color_libraries l ON l.id=c.library_id
                WHERE c.library_id=? AND c.name=?
                """,
                (clean_lib_id, clean_name),
            ).fetchone()
            conn.commit()
        return dict(row)

    def match(self, lab: tuple[float, float, float], library_id: str = "", limit: int = 12) -> List[Dict[str, Any]]:
        params: list[Any] = []
        where = ""
        if library_id:
            where = "WHERE c.library_id=?"
            params.append(library_id)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT c.id, c.library_id, l.name AS library_name, c.name, c.note,
                       c.illuminant, c.angle, c.l, c.a, c.b, c.hex
                FROM color_cards c
                JOIN color_libraries l ON l.id=c.library_id
                {where}
                """,
                params,
            ).fetchall()
        ranked = []
        for row in rows:
            item = dict(row)
            item["delta_e_00"] = delta_e_ciede2000(lab, (float(item["l"]), float(item["a"]), float(item["b"])))
            ranked.append(item)
        ranked.sort(key=lambda item: item["delta_e_00"])
        return ranked[: max(1, min(int(limit), 100))]
