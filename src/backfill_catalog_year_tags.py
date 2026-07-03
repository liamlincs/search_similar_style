import argparse
import json
import re
from pathlib import Path

from catalog_store import CatalogStore, derive_year_from_style_code, make_typed_tag

DEFAULT_CONFIG = Path("config/search_config.json")
YEAR_TAG_RE = re.compile(r"^(?:year:)?20\d{2}$")


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def derive_year_tag(style_code: str) -> str:
    return derive_year_from_style_code(style_code)


def main() -> None:
    parser = argparse.ArgumentParser(description="按现有产品库款号批量回填年份标签")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有年份标签，只保留新解析出的年份")
    parser.add_argument("--dry-run", action="store_true", help="只预览，不写入数据库")
    args = parser.parse_args()

    cfg = load_config(args.config)
    catalog_cfg = cfg.get("catalog", {})
    db_path = Path(catalog_cfg.get("db_path", "data/product_catalog.db"))
    store = CatalogStore(db_path)

    products = store.list_products(limit=100000, offset=0)
    updated = 0
    skipped = 0

    for product in products:
        style_code = str(product.get("style_code", "")).strip()
        if not style_code:
            skipped += 1
            continue
        year = derive_year_tag(style_code)
        year_tag = make_typed_tag("year", year)
        if not year_tag:
            print(f"SKIP {style_code}: cannot derive year")
            skipped += 1
            continue

        current_tags = [str(tag).strip() for tag in product.get("raw_tags", product.get("tags", [])) if str(tag).strip()]
        non_year_tags = [tag for tag in current_tags if not YEAR_TAG_RE.fullmatch(tag)]
        current_year_tags = [tag for tag in current_tags if YEAR_TAG_RE.fullmatch(tag)]

        if args.overwrite:
            next_tags = [*non_year_tags, year_tag]
        else:
            if year_tag in current_year_tags:
                skipped += 1
                continue
            next_tags = [*current_tags, year_tag]

        if args.dry_run:
            print(f"DRY {style_code}: {current_tags} -> {next_tags}")
            updated += 1
            continue

        store.replace_product_tags(style_code, next_tags)
        print(f"OK  {style_code}: set year tag {year_tag}")
        updated += 1

    print(f"done: updated={updated} skipped={skipped} total={len(products)} db={db_path}")


if __name__ == "__main__":
    main()
