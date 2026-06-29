from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

# 允许从项目根目录运行：python factors/concept_industry/scripts/retry_failed_members.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_ths_members import (  # noqa: E402
    PARQUET_DIR,
    BOARDS_FILE,
    MEMBERS_FILE,
    LOG_FILE,
    MEMBER_COLUMNS,
    LOG_COLUMNS,
    HttpAuthError,
    make_session,
    fetch_html_single,
    fetch_board_all_members,
    safe_write_csv,
    safe_write_parquet,
    validate_cookie,
)


MEMBERS_CSV = PARQUET_DIR / "ths_board_members.csv"
LOG_CSV = PARQUET_DIR / "ths_fetch_log.csv"
HISTORY_LOG_PARQUET = PARQUET_DIR / "ths_fetch_log_history.parquet"
HISTORY_LOG_CSV = PARQUET_DIR / "ths_fetch_log_history.csv"


def read_existing_members() -> pd.DataFrame:
    if MEMBERS_FILE.exists():
        return pd.read_parquet(MEMBERS_FILE)

    if MEMBERS_CSV.exists():
        return pd.read_csv(MEMBERS_CSV, dtype=str)

    return pd.DataFrame(columns=MEMBER_COLUMNS)


def read_existing_log() -> pd.DataFrame:
    if LOG_FILE.exists():
        return pd.read_parquet(LOG_FILE)

    if LOG_CSV.exists():
        return pd.read_csv(LOG_CSV, dtype=str)

    raise FileNotFoundError(
        f"找不到日志文件：{LOG_FILE} 或 {LOG_CSV}。请先至少运行过一次 fetch_ths_members.py"
    )


def latest_log(log_df: pd.DataFrame) -> pd.DataFrame:
    if log_df.empty:
        return log_df

    df = log_df.copy()
    df["fetch_time_sort"] = pd.to_datetime(df["fetch_time"], errors="coerce")
    # NaT 会排在最后，导致 drop_duplicates(keep='last') 选中 NaT 行。
    # 用最小时间填充 NaT，确保有效时间戳优先。
    df["fetch_time_sort"] = df["fetch_time_sort"].fillna(pd.Timestamp.min)

    df = (
        df.sort_values(["board_type", "board_code", "fetch_time_sort"])
        .drop_duplicates(subset=["board_code", "board_type"], keep="last")
        .drop(columns=["fetch_time_sort"])
        .reset_index(drop=True)
    )

    return df


def select_retry_boards(
    boards_df: pd.DataFrame,
    log_df: pd.DataFrame,
    statuses: set[str],
) -> pd.DataFrame:
    latest = latest_log(log_df)

    retry_keys = latest[latest["status"].isin(statuses)][
        ["board_code", "board_type", "status", "message", "rows"]
    ].copy()

    retry_keys["board_code"] = retry_keys["board_code"].astype(str)
    retry_keys["board_type"] = retry_keys["board_type"].astype(str)

    boards = boards_df.copy()
    boards["board_code"] = boards["board_code"].astype(str)
    boards["board_type"] = boards["board_type"].astype(str)

    retry_boards = boards.merge(
        retry_keys[["board_code", "board_type"]],
        on=["board_code", "board_type"],
        how="inner",
    )

    retry_boards = retry_boards.drop_duplicates(
        subset=["board_code", "board_type"]
    ).reset_index(drop=True)

    return retry_boards


def merge_members(old_members: pd.DataFrame, new_members: pd.DataFrame) -> pd.DataFrame:
    if old_members.empty and new_members.empty:
        return pd.DataFrame(columns=MEMBER_COLUMNS)

    if old_members.empty:
        merged = new_members.copy()
    elif new_members.empty:
        merged = old_members.copy()
    else:
        merged = pd.concat([old_members, new_members], ignore_index=True)

    for col in MEMBER_COLUMNS:
        if col not in merged.columns:
            merged[col] = ""

    merged = merged[MEMBER_COLUMNS]

    merged = (
        merged.drop_duplicates(
            subset=["board_code", "board_type", "stock_code"],
            keep="last",
        )
        .sort_values(["board_type", "board_code", "stock_code"])
        .reset_index(drop=True)
    )

    return merged


def merge_logs(old_log: pd.DataFrame, retry_log: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if old_log.empty and retry_log.empty:
        history = pd.DataFrame(columns=LOG_COLUMNS)
    elif old_log.empty:
        history = retry_log.copy()
    elif retry_log.empty:
        history = old_log.copy()
    else:
        history = pd.concat([old_log, retry_log], ignore_index=True)

    for col in LOG_COLUMNS:
        if col not in history.columns:
            history[col] = ""

    history = history[LOG_COLUMNS]
    current = latest_log(history)

    return current, history


def _do_incremental_save(
    old_members: pd.DataFrame,
    new_members: list[dict],
    old_log: pd.DataFrame,
    retry_logs: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """将当前进度写盘（增量保存）。返回 (merged_members, current_log, history_log)。"""
    new_members_df = pd.DataFrame(new_members, columns=MEMBER_COLUMNS)
    retry_log_df = pd.DataFrame(retry_logs, columns=LOG_COLUMNS)

    merged_members = merge_members(old_members, new_members_df)
    current_log, history_log = merge_logs(old_log, retry_log_df)

    safe_write_parquet(merged_members, MEMBERS_FILE)
    safe_write_csv(merged_members, MEMBERS_CSV)
    safe_write_parquet(current_log, LOG_FILE)
    safe_write_csv(current_log, LOG_CSV)
    safe_write_parquet(history_log, HISTORY_LOG_PARQUET)
    safe_write_csv(history_log, HISTORY_LOG_CSV)

    return merged_members, current_log, history_log


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(description="补跑 THS 失败板块成分股采集")
    parser.add_argument("--statuses", default="failed,empty", help="要补跑的状态，默认 failed,empty")
    parser.add_argument("--limit", type=int, default=0, help="只补跑前 N 个，用于分批")
    parser.add_argument("--sleep", type=float, default=10.0, help="每页/每板块之间暂停秒数")
    parser.add_argument("--max-pages", type=int, default=30, help="每个板块最多抓几页")
    parser.add_argument("--cooldown-every", type=int, default=20, help="每跑多少个板块长暂停一次")
    parser.add_argument("--cooldown", type=float, default=90.0, help="长暂停秒数")
    parser.add_argument("--save-every", type=int, default=10, help="每处理多少个板块增量保存一次进度")
    parser.add_argument("--dry-run", action="store_true", help="只显示待补跑板块，不实际请求")
    parser.add_argument("--debug", action="store_true", help="保存更多 HTML 调试文件")
    args = parser.parse_args()

    statuses = {s.strip() for s in args.statuses.split(",") if s.strip()}

    boards_df = pd.read_parquet(BOARDS_FILE)
    old_log = read_existing_log()

    retry_boards = select_retry_boards(boards_df, old_log, statuses=statuses)

    if args.limit and args.limit > 0:
        retry_boards = retry_boards.head(args.limit).copy()

    if retry_boards.empty:
        print("没有需要补跑的板块。")
        return

    # --dry-run 模式：只展示待补跑信息，不请求网络
    if args.dry_run:
        print(f"待补跑板块数量：{len(retry_boards)}")
        print(f"补跑状态：{sorted(statuses)}")
        print(f"sleep={args.sleep}, save_every={args.save_every}")
        print()
        type_counts = retry_boards["board_type"].value_counts()
        print("按 board_type 分组：")
        for btype, count in type_counts.items():
            print(f"  {btype}: {count}")
        est_seconds = len(retry_boards) * args.sleep
        est_minutes = est_seconds / 60
        print(f"\n预估耗时（不含冷却/退避）：{est_minutes:.0f} 分钟")
        print("\n样例板块（前 10 个）：")
        print(retry_boards.head(10)[["board_code", "board_name", "board_type"]].to_string(index=False))
        return

    session, manual_cookie, cookie_info = make_session()

    if not validate_cookie(session, manual_cookie):
        logger.error("Cookie 无效或已过期。请先在 Firefox 中登录 q.10jqka.com.cn，或设置 THS_COOKIE 环境变量。")
        sys.exit(1)

    old_members = read_existing_members()
    # 读取历史日志（包含所有中间条目），而非仅最新日志
    if HISTORY_LOG_PARQUET.exists():
        old_history_log = pd.read_parquet(HISTORY_LOG_PARQUET)
    elif HISTORY_LOG_CSV.exists():
        old_history_log = pd.read_csv(HISTORY_LOG_CSV, dtype=str)
    else:
        old_history_log = old_log

    print(f"待补跑板块数量：{len(retry_boards)}")
    print(f"补跑状态：{sorted(statuses)}")
    print(f"Cookie 状态：{cookie_info}")
    print(f"手动 THS_COOKIE 长度：{len(manual_cookie)}")
    print(f"sleep={args.sleep}, cooldown_every={args.cooldown_every}, cooldown={args.cooldown}, save_every={args.save_every}")

    new_members: list[dict] = []
    retry_logs: list[dict] = []
    last_exception: Exception | None = None

    consecutive_failures = 0
    circuit_trip_count = 0
    CIRCUIT_BREAKER_THRESHOLD = 5
    CIRCUIT_BREAKER_DELAY = 300  # 5 分钟
    CIRCUIT_BREAKER_MAX_TRIPS = 3  # 累计触发 3 次断路器后中止

    for idx, (_, row) in enumerate(
        tqdm(retry_boards.iterrows(), total=len(retry_boards), desc="补跑失败板块"),
        start=1,
    ):
        board = row.to_dict()

        board_code = str(board["board_code"]).strip()
        board_name = str(board["board_name"]).strip()
        board_type = str(board["board_type"]).strip()

        try:
            member_rows = fetch_board_all_members(
                session=session,
                board=board,
                max_pages=args.max_pages,
                sleep=args.sleep,
                debug=args.debug,
                manual_cookie=manual_cookie,
            )

            new_members.extend(member_rows)

            status = "success" if member_rows else "empty"
            message = ""
            last_exception = None

        except Exception as e:
            member_rows = []
            status = "failed"
            message = f"{type(e).__name__}: {str(e)[:300]}"
            last_exception = e

        retry_logs.append(
            {
                "fetch_time": datetime.now().isoformat(timespec="seconds"),
                "target": "members",
                "board_code": board_code,
                "board_name": board_name,
                "board_type": board_type,
                "status": status,
                "message": message,
                "rows": len(member_rows),
            }
        )

        # 断路器：检查是否为 HTTP 认证错误（401/403）
        is_auth_failure = isinstance(last_exception, HttpAuthError)

        if is_auth_failure:
            consecutive_failures += 1
            if consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
                circuit_trip_count += 1
                consecutive_failures = 0
                if circuit_trip_count >= CIRCUIT_BREAKER_MAX_TRIPS:
                    logger.error("断路器已触发 %d 次，疑似 Cookie 已过期。中止运行。", circuit_trip_count)
                    break
                logger.warning("连续 %d 次 401/403，暂停 %d 秒（第 %d 次断路）...",
                               CIRCUIT_BREAKER_THRESHOLD, CIRCUIT_BREAKER_DELAY, circuit_trip_count)
                time.sleep(CIRCUIT_BREAKER_DELAY)
        else:
            consecutive_failures = 0

        time.sleep(args.sleep)

        # 增量保存：每 save_every 个板块写盘一次，防止中断丢失进度
        if args.save_every > 0 and idx % args.save_every == 0:
            logger.info("增量保存进度（已处理 %d/%d 个板块）...", idx, len(retry_boards))
            _do_incremental_save(old_members, new_members, old_history_log, retry_logs)

        if args.cooldown_every > 0 and idx % args.cooldown_every == 0:
            logger.info("已补跑 %d 个，长暂停 %.0f 秒，降低 401 概率...", idx, args.cooldown)
            time.sleep(args.cooldown)

    # 最终保存（复用增量保存函数，直接获取合并后的 DataFrame）
    merged_members, current_log, _ = _do_incremental_save(
        old_members, new_members, old_history_log, retry_logs
    )

    new_members_df = pd.DataFrame(new_members, columns=MEMBER_COLUMNS)
    retry_log_df = pd.DataFrame(retry_logs, columns=LOG_COLUMNS)

    print("\n补跑完成。")
    print(f"本次新增成分股行数：{len(new_members_df)}")
    print(f"合并后成分股总行数：{len(merged_members)}")

    print("\n本次补跑状态：")
    print(retry_log_df["status"].value_counts(dropna=False))

    print("\n当前最新总状态：")
    print(current_log["status"].value_counts(dropna=False))

    print("\n当前 rows 统计：")
    print(current_log["rows"].describe())

    failed = current_log[current_log["status"] == "failed"]
    if not failed.empty:
        print("\n仍失败样例：")
        print(failed.head(20)[["board_code", "board_name", "board_type", "message"]])

    print(f"\n成员表：{MEMBERS_FILE}")
    print(f"当前日志：{LOG_FILE}")
    print(f"历史日志：{HISTORY_LOG_PARQUET}")


if __name__ == "__main__":
    main()