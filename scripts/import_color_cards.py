import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
import sys
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from color_card_store import ColorCardStore
from color_card_importer import read_color_rows, slugify_library_id


DEFAULT_FILES = [
    "彩龙丝光棉（2024）.xlsx",
    "东莞国彩丝光棉.xlsx",
    "恩盛纺织（新色卡）丝光棉.xlsx",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Import color-card xlsx files into SQLite.")
    parser.add_argument("--db", default=str(ROOT / "data" / "color_cards.db"))
    parser.add_argument("--source-dir", default=str(ROOT / "tools" / "ColorMeter_miniprogram_bluetooth_example"))
    parser.add_argument("files", nargs="*", default=DEFAULT_FILES)
    args = parser.parse_args()

    store = ColorCardStore(Path(args.db))
    source_dir = Path(args.source_dir)
    total = 0
    for order, filename in enumerate(args.files):
        path = Path(filename)
        if not path.is_absolute():
            path = source_dir / filename
        library_name = path.stem
        library_id = slugify_library_id(library_name)
        rows = read_color_rows(path)
        count = store.replace_library(library_id, library_name, path.name, rows, sort_order=order)
        total += count
        print(f"{library_id}\t{library_name}\t{count}")
    print(f"total\t{total}")


if __name__ == "__main__":
    main()
