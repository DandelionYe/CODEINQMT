from __future__ import annotations

import csv
import io
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

# 允许从项目根目录运行
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_ths_members import safe_write_csv, safe_write_parquet  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = PROJECT_ROOT / "factors" / "concept_industry"
PARQUET_DIR = BASE_DIR / "parquet"

SEED_FILES = [
    PARQUET_DIR / "seed_concept.csv",
    PARQUET_DIR / "seed_industry.csv",
]


def read_text_auto_encoding(path: Path) -> str:
    """读取文本文件，自动检测编码。只从磁盘读取一次。"""
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030", "cp936"]

    data = path.read_bytes()

    for enc in encodings:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue

    raise RuntimeError(f"无法识别文件编码：{path}")


def maybe_unwrap_browser_copied_csv(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines:
        return text

    first_line = lines[0].lstrip("\ufeff").strip()

    try:
        first_parsed = next(csv.reader([first_line]))
    except Exception:
        return text

    # 处理这种情况：
    # 每一整行被浏览器/编辑器额外包成了一个字段。
    # 例如表头被读成一个字段：board_code,board_name,board_type,href
    if len(first_parsed) == 1 and "," in first_parsed[0] and "board_code" in first_parsed[0]:
        fixed_lines = []

        for line in lines:
            try:
                parsed = next(csv.reader([line]))
                if len(parsed) == 1:
                    fixed_lines.append(parsed[0])
                else:
                    fixed_lines.append(line)
            except Exception:
                fixed_lines.append(line)

        return "\n".join(fixed_lines)

    return text


def read_seed_csv(path: Path) -> pd.DataFrame:
    text = read_text_auto_encoding(path)
    text = maybe_unwrap_browser_copied_csv(text)

    df = pd.read_csv(io.StringIO(text), dtype=str)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]

    required = {"board_code", "board_name", "board_type"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} 缺少字段：{missing}，当前字段：{list(df.columns)}"
        )

    if "href" not in df.columns:
        df["href"] = ""

    return df


def main() -> None:
    dfs: list[pd.DataFrame] = []

    for file in SEED_FILES:
        if not file.exists():
            print(f"跳过，文件不存在：{file}")
            continue

        print(f"读取：{file}")

        df = read_seed_csv(file)

        df["board_code"] = df["board_code"].astype(str).str.strip()
        df["board_name"] = df["board_name"].astype(str).str.strip()
        df["board_type"] = df["board_type"].astype(str).str.strip()
        df["href"] = df["href"].fillna("").astype(str).str.strip()

        df = df[df["board_code"].str.fullmatch(r"\d+", na=False)]
        df = df[df["board_name"].ne("")]
        df = df[df["board_type"].isin(["concept", "industry"])]

        df["source"] = "10jqka_page_seed"
        df["fetch_date"] = datetime.now().strftime("%Y-%m-%d")
        df["fetch_time"] = datetime.now().isoformat(timespec="seconds")
        df["raw_query"] = ""

        print(f"有效板块数：{len(df)}")
        dfs.append(df)

    if not dfs:
        raise RuntimeError("没有找到 seed_concept.csv 或 seed_industry.csv")

    result = pd.concat(dfs, ignore_index=True)

    result = result[
        [
            "board_code",
            "board_name",
            "board_type",
            "source",
            "fetch_date",
            "fetch_time",
            "raw_query",
            "href",
        ]
    ]

    result = (
        result.dropna(subset=["board_code", "board_name"])
        .drop_duplicates(subset=["board_code", "board_type"])
        .sort_values(["board_type", "board_code"])
        .reset_index(drop=True)
    )

    out_parquet = PARQUET_DIR / "ths_boards.parquet"
    out_csv = PARQUET_DIR / "ths_boards.csv"

    safe_write_parquet(result, out_parquet)
    safe_write_csv(result, out_csv)

    print("\n完成。")
    print(f"板块总数：{len(result)}")
    print(result["board_type"].value_counts(dropna=False))
    print(f"已保存：{out_csv}")
    print(f"已保存：{out_parquet}")

    print("\n前 10 行预览：")
    print(result.head(10))


if __name__ == "__main__":
    main()