# Factor Registry Draft: THS Concept/Industry Board Membership

## Data Identity

- **Factor name**: `ths_board_membership`
- **Data source**: Tonghuashun (THS) via q.10jqka.com.cn
- **Update frequency**: Manual / on-demand
- **Current coverage**: See [STATUS.md](STATUS.md) for latest numbers

## Schema

| Field | Type | Description | PIT? |
|-------|------|-------------|------|
| `stock_code` | str | Stock code (000001.SZ format) | Yes -- code changes are PIT |
| `board_type` | str | "concept" or "industry" | Yes -- board classification can change |
| `board_code` | str | Board code (881xxx/885xxx) | Yes -- code is stable |
| `board_name` | str | Board display name | No -- name can change |
| `fetch_date` | str | Date when data was fetched (YYYY-MM-DD) | Yes -- serves as PIT anchor |
| `fetch_time` | str | ISO timestamp of fetch | Yes -- more precise PIT anchor |
| `stock_name` | str | Stock display name | No -- name can change |
| `source` | str | Data source identifier | No -- metadata |

## PIT Considerations

1. **Board membership is NOT static**: Stocks can be added/removed from boards over time.
2. **`fetch_date` is the PIT anchor**: Use `fetch_date <= query_date` to get point-in-time membership.
3. **Historical snapshots**: Each fetch creates a new snapshot; merge with `keep='last'` on `(board_code, board_type, stock_code)`.
4. **Board classification can change**: A stock may move from "concept" to "industry" or vice versa.
5. **Board names are display-only**: Use `board_code` for joins, not `board_name`.

## Usage Examples

### Query a stock's boards (PIT-safe)

```sql
-- In DuckDB
SELECT board_type, board_code, board_name, fetch_date
FROM ths_board_members
WHERE stock_code = '000858.SZ'
  AND fetch_date <= '2026-05-19'
ORDER BY fetch_date DESC
```

### Get all stocks in a board

```sql
SELECT stock_code, stock_name, fetch_date
FROM ths_board_members
WHERE board_code = '300008'
  AND fetch_date = (SELECT MAX(fetch_date) FROM ths_board_members WHERE board_code = '300008')
```

### Use the convenience views

```sql
-- v_board_stocks: board-centric view
SELECT * FROM v_board_stocks WHERE board_code = '300008';

-- v_stock_boards: stock-centric view
SELECT * FROM v_stock_boards WHERE stock_code = '000858.SZ';
```

## Usage in D's_Flow

- **NOT yet integrated** into the main research pipeline.
- **Potential applications**:
  - Sector-neutral portfolio construction (neutralize concept/industry exposure)
  - Concept momentum strategies (ride trending board themes)
  - Cross-board membership signals (stocks in multiple boards may have unique properties)
- **Requires before integration**:
  - Unified factor interface (`factors/base.py` or similar)
  - PIT query layer with date-aware lookups
  - Data freshness tracking (how stale is the board membership data?)
  - Higher coverage (current 6% is insufficient for systematic strategies)

## Data Quality Notes

See [STATUS.md](STATUS.md) for detailed data state and failure breakdown.

Key points:
- Failed boards are due to anti-scraping (401/403), not data quality issues
- Successfully scraped data is validated and complete
- Coverage is biased toward concept boards
- Retry with `scripts/retry_failed_members.py` to improve coverage
