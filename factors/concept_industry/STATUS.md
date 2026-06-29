# THS Concept/Industry Board Data -- Status Summary

## Current Data State

| Metric | Value |
|--------|-------|
| Total boards | 452 (362 concept + 90 industry) |
| Successfully scraped | 27 (26 concept + 1 industry) |
| Failed to scrape | 425 (336 concept + 89 industry) |
| Failure rate | 94.0% |
| Stock-board member rows | 610 (from 27 successful boards) |
| Fetch date | 2026-05-19 |

## Failure Breakdown

| Error Type | Count | Percentage |
|------------|-------|------------|
| 401 Unauthorized | 421 | 99.1% of failures |
| 403 Forbidden | 4 | 0.9% of failures |
| **Total failures** | **425** | **100%** |

### What the Failures Mean

All 425 failures are **anti-scraping HTTP errors** from q.10jqka.com.cn (Tonghuashun / THS):

- **401 Unauthorized**: The THS server rejected the request because it detected automated scraping. This is a rate-limiting / bot-detection mechanism, NOT a data quality issue.
- **403 Forbidden**: Same anti-scraping protection, slightly different rejection method.

These are **infrastructure limitations**, not data problems. The 27 boards that were successfully scraped have valid, complete data.

## Valid Data from Successful Boards

The 27 successful boards produced 610 stock-board membership rows. This data is:

- Stored in `parquet/ths_board_members.parquet` and `parquet/ths_board_members.csv`
- Indexed in a DuckDB database at `ths_concept_industry.duckdb`
- Ready for use in factor analysis (with the caveat of limited board coverage)

The successful boards are biased toward concept boards (26 concept vs 1 industry), so industry coverage is minimal.

## How to Retry Failed Boards

Use the retry script to re-attempt the 425 failed boards:

```bash
# Dry-run: preview boards to retry, no network requests
python factors/concept_industry/scripts/retry_failed_members.py --dry-run

# Preview: only retry first 5 boards
python factors/concept_industry/scripts/retry_failed_members.py --limit 5

# Execute full retry (with incremental progress saves every 10 boards)
python factors/concept_industry/scripts/retry_failed_members.py
```

**Script features (improved):**

- **Cookie pre-validation**: Tests Cookie before starting; exits immediately if invalid
- **Exponential backoff**: 30s→60s→120s→240s→300s on 401/403 errors (3 retries)
- **Circuit breaker**: 5 consecutive 401/403 triggers a 5-minute cooldown
- **Periodic cooldown**: 90s pause every 20 boards (configurable)
- **Incremental saves**: Progress saved every 10 boards (`--save-every`); survives crashes
- **Dry-run mode**: `--dry-run` shows what would be retried without network requests
- **UA rotation**: 5 browser User-Agent strings, randomly selected per request

**Tips for successful retries:**

1. Wait at least 24 hours between retry attempts to avoid IP bans
2. Use a VPN or rotate proxies if available
3. Retry during off-peak hours (early morning or late night)
4. The script automatically respects the existing fetch log and only retries failures
5. Use `--dry-run` first to verify the retry scope

## Regenerating DuckDB

If the DuckDB file needs to be rebuilt from parquet data:

```bash
# Requires duckdb Python package (available in factors/concept_industry/.venv)
cd factors/concept_industry
.venv/Scripts/python.exe scripts/build_ths_duckdb.py
```

Or install duckdb in the main environment first:

```bash
pip install duckdb
python factors/concept_industry/scripts/build_ths_duckdb.py
```

## Files in This Directory

| File | Description |
|------|-------------|
| `parquet/ths_boards.parquet` | All 452 boards (metadata) |
| `parquet/ths_board_members.parquet` | 610 member rows from 27 successful boards |
| `parquet/ths_fetch_log.csv` | Fetch status log for all 452 boards |
| `ths_concept_industry.duckdb` | DuckDB with boards, members, and convenience views |
| `scripts/retry_failed_members.py` | Script to retry failed board scraping |
| `scripts/build_ths_duckdb.py` | Script to regenerate DuckDB from parquet |
