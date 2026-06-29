from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_FILE = PROJECT_ROOT / "factors" / "concept_industry" / "ths_concept_industry.duckdb"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", help="股票代码，例如 000858.SZ")
    parser.add_argument("--board", help="板块名称关键词，例如 白酒")
    args = parser.parse_args()

    con = duckdb.connect(str(DB_FILE), read_only=True)

    try:
        if args.stock:
            df = con.execute("""
                SELECT board_type, board_code, board_name
                FROM v_stock_boards
                WHERE stock_code = ?
                ORDER BY board_type, board_name
            """, [args.stock]).df()

            print(df)

        if args.board:
            df = con.execute("""
                SELECT board_type, board_code, board_name, stock_code, stock_name
                FROM v_board_stocks
                WHERE board_name LIKE ?
                ORDER BY board_type, board_name, stock_code
            """, [f"%{args.board}%"]).df()

            print(df)

        if not args.stock and not args.board:
            print("示例：")
            print(r"py .\scripts\query_examples.py --stock 000858.SZ")
            print(r"py .\scripts\query_examples.py --board 白酒")

    finally:
        con.close()


if __name__ == "__main__":
    main()
