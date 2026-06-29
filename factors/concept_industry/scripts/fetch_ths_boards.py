from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import pywencai

# 允许从项目根目录运行
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_ths_members import safe_write_csv, safe_write_parquet  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = PROJECT_ROOT / "factors" / "concept_industry"
PARQUET_DIR = BASE_DIR / "parquet"
RAW_DIR = BASE_DIR / "raw" / datetime.now().strftime("%Y-%m-%d")

PARQUET_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)


QUERY_SPECS = [
    {
        "board_type": "industry",
        "query": "同花顺行业板块 指数代码",
    },
    {
        "board_type": "industry",
        "query": "主力流入金额排序,指数代码881开头",
    },
    {
        "board_type": "concept",
        "query": "同花顺概念板块 指数代码",
    },
    {
        "board_type": "concept",
        "query": "概念板块 指数代码 885开头",
    },
]


def call_wencai(query: str, cookie: str | None) -> pd.DataFrame:
    kwargs = {}
    if cookie:
        kwargs["cookie"] = cookie

    try:
        df = pywencai.get(query=query, **kwargs)
    except TypeError:
        df = pywencai.get(query, **kwargs)

    if df is None:
        return pd.DataFrame()

    if isinstance(df, pd.DataFrame):
        return df

    return pd.DataFrame(df)


def normalize_board_code(value) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()

    # 常见形式：881001、881001.TI、指数代码:881001
    match = re.search(r"(88\d{4})", text)
    if match:
        return match.group(1)

    return None


def pick_board_name(row: pd.Series, code: str) -> str | None:
    preferred_keywords = [
        "指数简称",
        "指数名称",
        "板块名称",
        "概念名称",
        "行业名称",
        "名称",
        "简称",
    ]

    for keyword in preferred_keywords:
        for col in row.index:
            if keyword in str(col):
                value = row[col]
                if pd.notna(value):
                    text = str(value).strip()
                    if text and code not in text and not re.fullmatch(r"88\d{4}", text):
                        return text

    # 兜底：找一个不像代码、长度合理的文本
    for col in row.index:
        value = row[col]
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text:
            continue
        if code in text:
            continue
        if re.fullmatch(r"[\d\.\-%]+", text):
            continue
        if 1 < len(text) <= 30:
            return text

    return None


def extract_boards(df: pd.DataFrame, board_type: str, query: str) -> pd.DataFrame:
    records = []

    if df.empty:
        return pd.DataFrame(records)

    for _, row in df.iterrows():
        code = None

        for col in df.columns:
            value = row[col]
            code = normalize_board_code(value)
            if code:
                break

        if not code:
            continue

        name = pick_board_name(row, code)
        if not name:
            name = code

        records.append(
            {
                "board_code": code,
                "board_name": name,
                "board_type": board_type,
                "source": "pywencai",
                "fetch_date": datetime.now().strftime("%Y-%m-%d"),
                "fetch_time": datetime.now().isoformat(timespec="seconds"),
                "raw_query": query,
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    cookie = os.environ.get("PYWENCAI_COOKIE")

    all_boards = []

    for i, spec in enumerate(QUERY_SPECS, start=1):
        board_type = spec["board_type"]
        query = spec["query"]

        print(f"\n[{i}/{len(QUERY_SPECS)}] 查询：{query}")

        try:
            raw_df = call_wencai(query, cookie)
            raw_path = RAW_DIR / f"wencai_{board_type}_{i}.csv"
            raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
            print(f"原始结果已保存：{raw_path}")

            boards_df = extract_boards(raw_df, board_type, query)
            print(f"提取板块数量：{len(boards_df)}")

            if not boards_df.empty:
                all_boards.append(boards_df)

        except Exception as e:
            print(f"查询失败：{query}")
            print(f"错误：{repr(e)}")

    if not all_boards:
        raise RuntimeError("没有提取到任何板块。请检查 cookie、Node.js、pywencai 是否可用，或调整 QUERY_SPECS。")

    result = pd.concat(all_boards, ignore_index=True)

    result = (
        result.drop_duplicates(subset=["board_code", "board_type"])
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
    print(f"Parquet：{out_parquet}")
    print(f"CSV：{out_csv}")


if __name__ == "__main__":
    main()