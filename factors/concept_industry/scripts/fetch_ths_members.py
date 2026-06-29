from __future__ import annotations

import argparse
import io
import logging
import os
import random
import re
import time
from pathlib import Path
from datetime import datetime

import browser_cookie3
import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
]


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = PROJECT_ROOT / "factors" / "concept_industry"
PARQUET_DIR = BASE_DIR / "parquet"
RAW_DIR = BASE_DIR / "raw" / datetime.now().strftime("%Y-%m-%d")
HTML_DIR = RAW_DIR / "members_html"

PARQUET_DIR.mkdir(parents=True, exist_ok=True)
HTML_DIR.mkdir(parents=True, exist_ok=True)

BOARDS_FILE = PARQUET_DIR / "ths_boards.parquet"
MEMBERS_FILE = PARQUET_DIR / "ths_board_members.parquet"
LOG_FILE = PARQUET_DIR / "ths_fetch_log.parquet"

MEMBER_COLUMNS = [
    "board_code",
    "board_name",
    "board_type",
    "stock_code",
    "stock_name",
    "fetch_date",
    "fetch_time",
    "source",
]

LOG_COLUMNS = [
    "fetch_time",
    "target",
    "board_code",
    "board_name",
    "board_type",
    "status",
    "message",
    "rows",
]


def normalize_stock_code(value) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    match = re.search(r"(?<!\d)(\d{6})(?!\d)", text)

    if not match:
        return None

    code = match.group(1)

    if code.startswith(("60", "68", "90")):
        return f"{code}.SH"

    if code.startswith(("00", "30", "20")):
        return f"{code}.SZ"

    if code.startswith(("43", "83", "87", "88", "92")):
        return f"{code}.BJ"

    return code


def clean_cookie(raw: str | None) -> str:
    if not raw:
        return ""

    cookie = raw.strip()

    if cookie.lower().startswith("cookie:"):
        cookie = cookie.split(":", 1)[1].strip()

    cookie = cookie.replace("；", ";")
    cookie = cookie.replace("\r", " ").replace("\n", " ")
    cookie = re.sub(r"\s+", " ", cookie).strip()

    # 请求头需要 latin-1 可编码；正常 Cookie 应该全是 ASCII。
    cookie = cookie.encode("latin-1", errors="ignore").decode("latin-1")

    return cookie


def build_ajax_url(
    board_type: str,
    board_code: str,
    page: int,
    field: str,
    scheme: str,
) -> str:
    if board_type == "concept":
        path = "gn"
    elif board_type == "industry":
        path = "thshy"
    else:
        raise ValueError(f"未知 board_type: {board_type}")

    return (
        f"{scheme}://q.10jqka.com.cn/{path}/detail/"
        f"field/{field}/order/desc/page/{page}/ajax/1/code/{board_code}"
    )


def get_preferred_schemes(board: dict) -> list[str]:
    href = str(board.get("href", "") or "").strip().lower()

    schemes: list[str] = []

    if href.startswith("http://"):
        schemes.append("http")
    elif href.startswith("https://"):
        schemes.append("https")

    for scheme in ["https", "http"]:
        if scheme not in schemes:
            schemes.append(scheme)

    return schemes


def make_headers(board: dict, manual_cookie: str = "") -> dict:
    href = str(board.get("href", "") or "").strip()

    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": href or "https://q.10jqka.com.cn/",
        "X-Requested-With": "XMLHttpRequest",
        "Connection": "keep-alive",
    }

    if manual_cookie:
        headers["Cookie"] = manual_cookie

    return headers


def make_session() -> tuple[requests.Session, str, str]:
    """
    返回：
    - requests.Session
    - 手动 Cookie 字符串，来自环境变量 THS_COOKIE
    - Cookie 加载说明
    """
    session = requests.Session()

    loaded_messages: list[str] = []

    # 优先尝试读取 Firefox 中 10jqka 相关 Cookie。
    # 如果 Firefox 正在运行导致数据库锁住，可以关闭 Firefox 后再试。
    for domain in ["q.10jqka.com.cn", ".10jqka.com.cn", "10jqka.com.cn"]:
        try:
            cj = browser_cookie3.firefox(domain_name=domain)
            cookies = list(cj)

            if cookies:
                for cookie in cookies:
                    session.cookies.set(
                        name=cookie.name,
                        value=cookie.value,
                        domain=cookie.domain,
                        path=cookie.path,
                    )

                loaded_messages.append(f"Firefox Cookie: domain={domain}, count={len(cookies)}")
                break

        except Exception as e:
            loaded_messages.append(
                f"Firefox Cookie 读取失败: domain={domain}, {type(e).__name__}: {str(e)[:120]}"
            )

    manual_cookie = clean_cookie(os.environ.get("THS_COOKIE"))

    if manual_cookie:
        loaded_messages.append(f"THS_COOKIE 环境变量长度={len(manual_cookie)}")

    if not loaded_messages:
        loaded_messages.append("没有读取到任何 Cookie")

    return session, manual_cookie, " | ".join(loaded_messages)


def decode_response_content(response: requests.Response) -> str:
    # 同花顺行情中心页面常见 GBK/GB18030 编码；不要依赖 apparent_encoding。
    return response.content.decode("gb18030", errors="replace")


def save_debug_html(
    html: str,
    board_type: str,
    board_code: str,
    field: str,
    page: int,
    status_code: int,
    scheme: str,
) -> None:
    filename = (
        f"{board_type}_{board_code}_field{field}_page{page}_"
        f"{scheme}_status{status_code}.html"
    )
    path = HTML_DIR / filename
    path.write_text(html, encoding="utf-8", errors="ignore")


def fetch_html_single(
    session: requests.Session,
    board: dict,
    page: int,
    field: str,
    manual_cookie: str,
) -> tuple[str, int]:
    """单次请求，返回 (html, status_code)。"""
    board_code = str(board["board_code"]).strip()
    board_type = str(board["board_type"]).strip()

    headers = make_headers(board, manual_cookie=manual_cookie)

    href = str(board.get("href", "") or "").strip()

    # 先访问详情页，建立站点上下文。
    if href:
        try:
            session.get(href, headers=headers, timeout=20)
            time.sleep(0.3)
        except Exception:
            pass

    last_html = ""
    last_status = 0

    for scheme in get_preferred_schemes(board):
        url = build_ajax_url(
            board_type=board_type,
            board_code=board_code,
            page=page,
            field=field,
            scheme=scheme,
        )

        response = session.get(url, headers=headers, timeout=20)
        html = decode_response_content(response)

        save_debug_html(
            html=html,
            board_type=board_type,
            board_code=board_code,
            field=field,
            page=page,
            status_code=response.status_code,
            scheme=scheme,
        )

        if response.status_code < 400:
            return html, response.status_code

        last_html = html
        last_status = response.status_code

        # 如果 https 401，继续试 http；如果 http 也失败，最后统一返回。
        time.sleep(0.2)

    if last_status == 0:
        raise RuntimeError(f"未能获取 HTML，最后返回：{last_html[:300]}")

    return last_html, last_status


def fetch_with_backoff(
    session: requests.Session,
    board: dict,
    page: int,
    field: str,
    manual_cookie: str,
    max_retries: int = 3,
    base_delay: float = 30.0,
    max_delay: float = 300.0,
) -> tuple[str, int]:
    """带指数退避的请求，返回 (html, status_code)。

    对 401/403/429 和 5xx 错误进行重试。网络异常（Timeout、ConnectionError）也会重试。
    """
    delay = base_delay
    last_status = 0
    last_html = ""
    for attempt in range(max_retries + 1):
        try:
            html, status_code = fetch_html_single(session, board, page, field, manual_cookie)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            # 网络异常：视为可重试的临时错误
            last_status = 0
            last_html = ""
            if attempt < max_retries:
                jitter = random.uniform(0.5, 2.0)
                wait = min(delay * (2 ** attempt) + jitter, max_delay)
                logger.warning("[BACKOFF] %s on %s page %d, waiting %.0fs (attempt %d/%d)",
                               type(e).__name__, board.get('board_code'), page,
                               wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue
            break

        last_status = status_code
        last_html = html

        # 成功或不可重试的错误（非 401/403/429/5xx）
        if status_code not in (401, 403, 429) and status_code < 500:
            return html, status_code

        if attempt < max_retries:
            jitter = random.uniform(0.5, 2.0)
            wait = min(delay * (2 ** attempt) + jitter, max_delay)
            logger.warning("[BACKOFF] HTTP %s on %s page %d, waiting %.0fs (attempt %d/%d)",
                           status_code, board.get('board_code'), page,
                           wait, attempt + 1, max_retries)
            time.sleep(wait)

    # 重试后仍然失败，返回最后一次的响应内容（而非空字符串）
    return last_html, last_status


class HttpAuthError(RuntimeError):
    """HTTP 认证错误（401/403），携带结构化状态码。"""

    def __init__(self, status_code: int, board_code: str, page: int, body: str = ""):
        self.status_code = status_code
        self.board_code = board_code
        self.page = page
        super().__init__(f"HTTP {status_code} on {board_code} page {page}")
        self.body = body


def fetch_html(
    session: requests.Session,
    board: dict,
    page: int,
    field: str,
    manual_cookie: str,
) -> str:
    """带指数退避的请求，返回 html（兼容旧接口）。"""
    html, status_code = fetch_with_backoff(session, board, page, field, manual_cookie)
    if status_code >= 400:
        board_code = str(board.get("board_code", ""))
        if status_code in (401, 403):
            raise HttpAuthError(status_code, board_code, page, body=html[:200])
        raise RuntimeError(
            f"HTTP {status_code} on {board_code} page {page} "
            f"[body[:200]={html[:200]!r}]"
        )
    return html


def find_code_and_name_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    columns = [str(c).strip() for c in df.columns]
    df.columns = columns

    code_col = None
    name_col = None

    for col in columns:
        sample = df[col].astype(str).head(30)
        hit_count = sample.str.contains(r"(?<!\d)\d{6}(?!\d)", regex=True, na=False).sum()
        if hit_count > 0:
            code_col = col
            break

    for col in columns:
        col_text = str(col)
        if any(k in col_text for k in ["名称", "简称", "股票名称", "股票简称"]):
            name_col = col
            break

    if code_col is not None and name_col is None:
        try:
            idx = columns.index(code_col)
            if idx + 1 < len(columns):
                name_col = columns[idx + 1]
        except ValueError:
            pass

    return code_col, name_col


def pick_stock_name(row: pd.Series, name_col: str | None) -> str:
    if name_col is not None:
        value = row.get(name_col)
        if pd.notna(value):
            candidate = str(value).strip()
            if candidate and not normalize_stock_code(candidate):
                return candidate

    values = [str(v).strip() for v in row.tolist() if pd.notna(v)]

    for value in values:
        if not value:
            continue
        if normalize_stock_code(value):
            continue
        if re.fullmatch(r"[\d\.\-\+%]+", value):
            continue
        if 1 < len(value) <= 20:
            return value

    return ""


def parse_members_from_html(html: str, board: dict) -> list[dict]:
    rows: list[dict] = []

    try:
        tables = pd.read_html(io.StringIO(html))
    except ValueError:
        return rows

    board_code = str(board["board_code"]).strip()
    board_name = str(board["board_name"]).strip()
    board_type = str(board["board_type"]).strip()

    for df in tables:
        if df.empty:
            continue

        code_col, name_col = find_code_and_name_columns(df)

        if code_col is None:
            continue

        for _, row in df.iterrows():
            stock_code = normalize_stock_code(row.get(code_col))

            if not stock_code:
                continue

            stock_name = pick_stock_name(row, name_col)

            rows.append(
                {
                    "board_code": board_code,
                    "board_name": board_name,
                    "board_type": board_type,
                    "stock_code": stock_code,
                    "stock_name": stock_name,
                    "fetch_date": datetime.now().strftime("%Y-%m-%d"),
                    "fetch_time": datetime.now().isoformat(timespec="seconds"),
                    "source": "10jqka_q_ajax",
                }
            )

    return rows


def fetch_board_all_members(
    session: requests.Session,
    board: dict,
    max_pages: int,
    sleep: float,
    debug: bool,
    manual_cookie: str,
) -> list[dict]:
    board_code = str(board["board_code"]).strip()
    board_type = str(board["board_type"]).strip()

    if board_type == "concept":
        fields = ["264648", "199112"]
    else:
        fields = ["199112", "264648"]

    last_error: Exception | None = None

    for field in fields:
        all_rows: list[dict] = []
        seen_page_signatures: set[tuple[str, ...]] = set()

        for page in range(1, max_pages + 1):
            try:
                html = fetch_html(
                    session=session,
                    board=board,
                    page=page,
                    field=field,
                    manual_cookie=manual_cookie,
                )

                if debug:
                    html_path = HTML_DIR / f"{board_type}_{board_code}_field{field}_page{page}_parsed.html"
                    html_path.write_text(html, encoding="utf-8", errors="ignore")

                page_rows = parse_members_from_html(html, board)

                if not page_rows:
                    break

                signature = tuple(sorted({row["stock_code"] for row in page_rows}))

                # 防止某些接口 page 参数无效，导致每页重复返回第一页。
                if signature in seen_page_signatures:
                    break

                seen_page_signatures.add(signature)
                all_rows.extend(page_rows)

                time.sleep(sleep + random.uniform(0.5, 2.0))

            except Exception as e:
                last_error = e
                break

        if all_rows:
            return all_rows

    if last_error:
        raise last_error

    return []


def safe_write_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
        df.to_csv(alt_path, index=False, encoding="utf-8-sig")
        print(f"警告：{path} 被占用，已另存为：{alt_path}")
        return alt_path


def safe_write_parquet(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_parquet(path, index=False)
        return path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
        df.to_parquet(alt_path, index=False)
        print(f"警告：{path} 被占用，已另存为：{alt_path}")
        return alt_path


def validate_cookie(session, manual_cookie: str) -> bool:
    """验证 Cookie 是否有效。"""
    test_board = {
        "board_code": "300008",
        "board_type": "concept",
        "href": "https://q.10jqka.com.cn/gn/detail/code/300008/",
    }
    try:
        html, status = fetch_html_single(session, test_board, 1, "199112", manual_cookie)
        if status >= 400:
            logger.error("Cookie 验证失败: HTTP %s, 响应长度 %d", status, len(html) if html else 0)
            return False
        if not html or len(html) < 1000:
            logger.error("Cookie 验证失败: 响应过短 (%d 字符)", len(html) if html else 0)
            return False
        return True
    except Exception as e:
        logger.error("Cookie 验证异常: %s", e)
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只抓前 N 个板块，用于测试")
    parser.add_argument("--sleep", type=float, default=0.6, help="每页/每板块之间暂停秒数")
    parser.add_argument("--max-pages", type=int, default=30, help="每个板块最多抓几页")
    parser.add_argument("--debug", action="store_true", help="保存更多 HTML 调试文件")
    args = parser.parse_args()

    if not BOARDS_FILE.exists():
        raise FileNotFoundError(f"找不到板块文件：{BOARDS_FILE}")

    boards = pd.read_parquet(BOARDS_FILE)

    if boards.empty:
        raise RuntimeError("ths_boards.parquet 为空，无法抓成分股。")

    if args.limit and args.limit > 0:
        boards = boards.head(args.limit).copy()

    session, manual_cookie, cookie_info = make_session()

    if not validate_cookie(session, manual_cookie):
        logger.error("Cookie 无效或已过期。请先在 Firefox 中登录 q.10jqka.com.cn，或设置 THS_COOKIE 环境变量。")
        return

    print(f"准备抓取板块数量：{len(boards)}")
    print(f"Cookie 状态：{cookie_info}")
    print(f"手动 THS_COOKIE 长度：{len(manual_cookie)}")

    all_members: list[dict] = []
    logs: list[dict] = []

    for _, row in tqdm(boards.iterrows(), total=len(boards), desc="抓取成分股"):
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

            all_members.extend(member_rows)

            status = "success" if member_rows else "empty"

            logs.append(
                {
                    "fetch_time": datetime.now().isoformat(timespec="seconds"),
                    "target": "members",
                    "board_code": board_code,
                    "board_name": board_name,
                    "board_type": board_type,
                    "status": status,
                    "message": "",
                    "rows": len(member_rows),
                }
            )

        except Exception as e:
            logs.append(
                {
                    "fetch_time": datetime.now().isoformat(timespec="seconds"),
                    "target": "members",
                    "board_code": board_code,
                    "board_name": board_name,
                    "board_type": board_type,
                    "status": "failed",
                    "message": f"{type(e).__name__}: {str(e)[:300]}",
                    "rows": 0,
                }
            )

        time.sleep(args.sleep)

    members_df = pd.DataFrame(all_members, columns=MEMBER_COLUMNS)
    logs_df = pd.DataFrame(logs, columns=LOG_COLUMNS)

    if not members_df.empty:
        members_df = (
            members_df.drop_duplicates(
                subset=["board_code", "board_type", "stock_code"]
            )
            .sort_values(["board_type", "board_code", "stock_code"])
            .reset_index(drop=True)
        )

    members_parquet_path = safe_write_parquet(members_df, MEMBERS_FILE)
    members_csv_path = safe_write_csv(
        members_df,
        PARQUET_DIR / "ths_board_members.csv",
    )

    log_parquet_path = safe_write_parquet(logs_df, LOG_FILE)
    log_csv_path = safe_write_csv(
        logs_df,
        PARQUET_DIR / "ths_fetch_log.csv",
    )

    print("\n完成。")
    print(f"成分股映射行数：{len(members_df)}")

    if not logs_df.empty:
        print("\n状态统计：")
        print(logs_df["status"].value_counts(dropna=False))

        print("\n抓取行数统计：")
        print(logs_df["rows"].describe())

        failed = logs_df[logs_df["status"] == "failed"]
        if not failed.empty:
            print("\n失败样例：")
            print(failed.head(20)[["board_code", "board_name", "board_type", "message"]])

        empty = logs_df[logs_df["status"] == "empty"]
        if not empty.empty:
            print("\n空结果样例：")
            print(empty.head(20)[["board_code", "board_name", "board_type"]])

    print(f"\n成分股 Parquet：{members_parquet_path}")
    print(f"成分股 CSV：{members_csv_path}")
    print(f"日志 Parquet：{log_parquet_path}")
    print(f"日志 CSV：{log_csv_path}")


if __name__ == "__main__":
    main()