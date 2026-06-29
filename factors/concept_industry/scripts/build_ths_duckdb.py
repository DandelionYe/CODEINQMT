from __future__ import annotations

from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = PROJECT_ROOT / "factors" / "concept_industry"
PARQUET_DIR = BASE_DIR / "parquet"
DB_FILE = BASE_DIR / "ths_concept_industry.duckdb"

BOARDS_FILE = PARQUET_DIR / "ths_boards.parquet"
MEMBERS_FILE = PARQUET_DIR / "ths_board_members.parquet"
LOG_FILE = PARQUET_DIR / "ths_fetch_log.parquet"


def main() -> None:
    if not BOARDS_FILE.exists():
        raise FileNotFoundError(BOARDS_FILE)

    if not MEMBERS_FILE.exists():
        raise FileNotFoundError(MEMBERS_FILE)

    con = duckdb.connect(str(DB_FILE))

    try:
        con.execute(
            "CREATE OR REPLACE TABLE ths_boards AS SELECT * FROM read_parquet(?)",
            [str(BOARDS_FILE)],
        )

        con.execute(
            "CREATE OR REPLACE TABLE ths_board_members AS SELECT * FROM read_parquet(?)",
            [str(MEMBERS_FILE)],
        )

        if LOG_FILE.exists():
            con.execute(
                "CREATE OR REPLACE TABLE ths_fetch_log AS SELECT * FROM read_parquet(?)",
                [str(LOG_FILE)],
            )

        con.execute("""
            CREATE OR REPLACE VIEW v_board_stocks AS
            SELECT
                board_type,
                board_code,
                board_name,
                stock_code,
                stock_name,
                fetch_date
            FROM ths_board_members
        """)

        con.execute("""
            CREATE OR REPLACE VIEW v_stock_boards AS
            SELECT
                stock_code,
                stock_name,
                board_type,
                board_code,
                board_name,
                fetch_date
            FROM ths_board_members
        """)

        print("DuckDB 已生成：", DB_FILE)

        print("\n板块数量：")
        print(con.execute("""
            SELECT board_type, COUNT(*) AS board_count
            FROM ths_boards
            GROUP BY board_type
            ORDER BY board_type
        """).df())

        print("\n成分股映射数量：")
        print(con.execute("""
            SELECT board_type, COUNT(*) AS member_rows
            FROM ths_board_members
            GROUP BY board_type
            ORDER BY board_type
        """).df())

        print("\n单只股票归属板块示例：")
        print(con.execute("""
            SELECT *
            FROM v_stock_boards
            WHERE stock_code = '000858.SZ'
            LIMIT 20
        """).df())

    finally:
        con.close()


if __name__ == "__main__":
    main()
