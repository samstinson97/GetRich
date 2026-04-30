#!/usr/bin/env python3
"""
reconcile_fills.py -- Post-close trade reconciliation

Runs ~30 min after market close. For each paper trading account:
1. Reads state JSON to find trades with buy/sell order IDs
2. Fetches actual fill prices from Alpaca API
3. Computes slippage (actual fill vs signal-time estimate vs official close)
4. Writes enriched data to production/tracking/reconciled_trades.csv

This CSV is the single source of truth for backtest-vs-live comparison.
One row per completed trade, all fields needed for analysis.
"""
import csv
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

PROD = Path(__file__).parent
ROOT = PROD.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

import alpaca_trade_api as tradeapi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(PROD / "logs" / f"reconcile_{datetime.now():%Y%m%d}.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("reconcile")

TRACKING_DIR = PROD / "tracking"
TRACKING_DIR.mkdir(parents=True, exist_ok=True)
RECONCILED_CSV = TRACKING_DIR / "reconciled_trades.csv"

CSV_COLUMNS = [
    # Identity
    "account", "ticker", "entry_date", "exit_date", "days_held",
    # Prices: signal-time estimates
    "signal_entry_price", "signal_exit_price",
    # Prices: actual Alpaca fills
    "actual_entry_price", "actual_exit_price",
    # Prices: official close on fill day
    "entry_day_close", "exit_day_close",
    # Slippage analysis
    "entry_slippage_vs_close_bps", "exit_slippage_vs_close_bps",
    "total_slippage_bps",
    # P&L
    "pnl_pct_signal", "pnl_pct_actual",
    # ML scores
    "entry_ml_score", "exit_ml_score",
    # Market context at entry
    "entry_vix", "entry_spy_5d_ret", "entry_spy_dist200",
    # Market context at exit
    "exit_vix", "exit_spy_5d_ret", "exit_spy_dist200",
    # Top features at entry (JSON string)
    "entry_top_features",
    # Top features at exit (JSON string)
    "exit_top_features",
    # Exit reason
    "exit_reason",
    # Order IDs (for audit trail)
    "buy_order_id", "sell_order_id",
    # Reconciliation metadata
    "reconciled_at",
]

# Account configs: name -> (env_file, state_file)
ACCOUNTS = {
    "v3_fresh": {
        "env": PROD / "accounts" / "v3_fresh.env",
        "state": PROD / "ml_v3" / "state_fresh.json",
    },
    "v4_gtc": {
        "env": PROD / "accounts" / "v4_gtc.env",
        "state": PROD / "ml_v4" / "state_gtc.json",
    },
    "v4_moc": {
        "env": PROD / "accounts" / "v4_moc.env",
        "state": PROD / "ml_v4" / "state_moc.json",
    },
}


def _load_env(env_path):
    """Load env file and return dict."""
    env = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def _get_api(env_dict):
    """Create Alpaca API client from env dict."""
    key = env_dict.get("ALPACA_API_KEY", "")
    secret = env_dict.get("ALPACA_SECRET_KEY", "")
    base = env_dict.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    return tradeapi.REST(key, secret, base, api_version="v2")


def _get_order_fill(api, order_id):
    """Fetch fill details for an order. Returns dict or None."""
    if not order_id or order_id == "DRY-RUN":
        return None
    try:
        order = api.get_order(order_id)
        if order.status == "filled":
            return {
                "filled_avg_price": float(order.filled_avg_price),
                "filled_qty": float(order.filled_qty),
                "filled_at": str(order.filled_at),
            }
    except Exception as e:
        log.warning(f"  Failed to fetch order {order_id}: {e}")
    return None


def _get_close_price(ticker, date_str):
    """Fetch the official close price for a ticker on a given date.
    Uses yfinance for consolidated data."""
    try:
        import yfinance as yf
        dt = datetime.fromisoformat(date_str.replace("+00:00", "").replace("Z", ""))
        start = dt.strftime("%Y-%m-%d")
        end = (dt + timedelta(days=3)).strftime("%Y-%m-%d")
        data = yf.download(ticker, start=start, end=end,
                           auto_adjust=True, progress=False)
        if data.empty:
            return None
        # Find the row matching the target date
        target = dt.date()
        for idx in data.index:
            if idx.date() == target:
                close = data.loc[idx, "Close"]
                if hasattr(close, "item"):
                    return close.item()
                return float(close)
        # If exact date not found, return first available
        close = data["Close"].iloc[0]
        if hasattr(close, "item"):
            return close.item()
        return float(close)
    except Exception as e:
        log.warning(f"  Failed to fetch close for {ticker} on {date_str}: {e}")
        return None


def _load_existing_keys():
    """Load set of (account, ticker, entry_date) already reconciled."""
    keys = set()
    if not RECONCILED_CSV.exists():
        return keys
    with open(RECONCILED_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add((row["account"], row["ticker"], row["entry_date"]))
    return keys


def _ensure_csv():
    """Create CSV with headers if it doesn't exist."""
    if not RECONCILED_CSV.exists():
        with open(RECONCILED_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)


def _append_row(row_dict):
    """Append a single row to the reconciled CSV."""
    with open(RECONCILED_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row_dict)


def _extract_ctx(enrichment, prefix):
    """Extract market context fields from enrichment dict."""
    ctx = enrichment.get("market_ctx", {})
    return {
        f"{prefix}_vix": ctx.get("vix", ""),
        f"{prefix}_spy_5d_ret": ctx.get("spy_5d_ret", ""),
        f"{prefix}_spy_dist200": ctx.get("spy_dist200", ""),
    }


def reconcile_account(account_name, config):
    """Reconcile all unreconciled trades for one account."""
    log.info(f"--- {account_name} ---")

    env = _load_env(config["env"])
    api = _get_api(env)

    state_file = config["state"]
    if not state_file.exists():
        log.info(f"  No state file: {state_file}")
        return 0

    with open(state_file) as f:
        state = json.load(f)

    trades = state.get("trades", [])
    if not trades:
        log.info("  No trades to reconcile")
        return 0

    existing_keys = _load_existing_keys()
    reconciled = 0

    for trade in trades:
        key = (account_name, trade["ticker"], trade.get("entry_date", ""))
        if key in existing_keys:
            continue

        ticker = trade["ticker"]
        log.info(f"  Reconciling {ticker} ({trade.get('entry_date', '?')[:10]} -> {trade.get('exit_date', '?')[:10]})")

        # Fetch actual fill prices from Alpaca
        buy_fill = _get_order_fill(api, trade.get("buy_order_id"))
        sell_fill = _get_order_fill(api, trade.get("sell_order_id"))

        actual_entry = buy_fill["filled_avg_price"] if buy_fill else None
        actual_exit = sell_fill["filled_avg_price"] if sell_fill else None

        # Fetch official close prices on fill days
        entry_close = None
        exit_close = None
        if buy_fill and buy_fill.get("filled_at"):
            entry_close = _get_close_price(ticker, buy_fill["filled_at"])
        elif trade.get("entry_date"):
            entry_close = _get_close_price(ticker, trade["entry_date"])

        if sell_fill and sell_fill.get("filled_at"):
            exit_close = _get_close_price(ticker, sell_fill["filled_at"])
        elif trade.get("exit_date"):
            exit_close = _get_close_price(ticker, trade["exit_date"])

        # Compute slippage in basis points
        # Entry slippage: (actual_fill - close) / close * 10000
        # Positive = paid more than close (bad for buys)
        entry_slip = None
        if actual_entry and entry_close and entry_close > 0:
            entry_slip = round((actual_entry - entry_close) / entry_close * 10000, 2)

        exit_slip = None
        if actual_exit and exit_close and exit_close > 0:
            # For sells, negative = sold below close (bad)
            exit_slip = round((actual_exit - exit_close) / exit_close * 10000, 2)

        total_slip = None
        if entry_slip is not None and exit_slip is not None:
            # Total round-trip cost: paid more on entry + received less on exit
            total_slip = round(entry_slip - exit_slip, 2)

        # Actual P&L
        pnl_actual = None
        if actual_entry and actual_exit and actual_entry > 0:
            pnl_actual = round((actual_exit / actual_entry - 1) * 100, 4)

        # Extract enrichment data
        entry_enrich = trade.get("entry_enrichment", {})
        exit_enrich = trade.get("exit_enrichment", {})

        row = {
            "account": account_name,
            "ticker": ticker,
            "entry_date": trade.get("entry_date", ""),
            "exit_date": trade.get("exit_date", ""),
            "days_held": trade.get("days_held", ""),
            # Signal-time estimates
            "signal_entry_price": trade.get("entry_price", ""),
            "signal_exit_price": trade.get("exit_price", ""),
            # Actual fills
            "actual_entry_price": actual_entry if actual_entry else "",
            "actual_exit_price": actual_exit if actual_exit else "",
            # Official close on fill day
            "entry_day_close": round(entry_close, 4) if entry_close else "",
            "exit_day_close": round(exit_close, 4) if exit_close else "",
            # Slippage
            "entry_slippage_vs_close_bps": entry_slip if entry_slip is not None else "",
            "exit_slippage_vs_close_bps": exit_slip if exit_slip is not None else "",
            "total_slippage_bps": total_slip if total_slip is not None else "",
            # P&L
            "pnl_pct_signal": trade.get("pnl_pct", ""),
            "pnl_pct_actual": pnl_actual if pnl_actual is not None else "",
            # ML scores
            "entry_ml_score": entry_enrich.get("ml_score", ""),
            "exit_ml_score": exit_enrich.get("ml_score", ""),
            # Market context
            **_extract_ctx(entry_enrich, "entry"),
            **_extract_ctx(exit_enrich, "exit"),
            # Top features (JSON strings for CSV compatibility)
            "entry_top_features": json.dumps(entry_enrich.get("top_features", {})),
            "exit_top_features": json.dumps(exit_enrich.get("top_features", {})),
            # Exit reason
            "exit_reason": trade.get("exit_reason", ""),
            # Order IDs
            "buy_order_id": trade.get("buy_order_id", ""),
            "sell_order_id": trade.get("sell_order_id", ""),
            # Metadata
            "reconciled_at": datetime.now().isoformat(),
        }

        _append_row(row)
        reconciled += 1

        # Log slippage summary
        slip_parts = []
        if entry_slip is not None:
            slip_parts.append(f"entry={entry_slip:+.1f}bps")
        if exit_slip is not None:
            slip_parts.append(f"exit={exit_slip:+.1f}bps")
        if total_slip is not None:
            slip_parts.append(f"total={total_slip:+.1f}bps")
        if pnl_actual is not None:
            slip_parts.append(f"actual_pnl={pnl_actual:+.2f}%")
        log.info(f"    {', '.join(slip_parts) if slip_parts else 'no fill data yet'}")

    return reconciled


def reconcile_open_positions():
    """For open positions, update entry price with actual fill if available.
    This runs non-destructively -- only updates if actual fill differs from estimate."""
    log.info("--- Reconciling open position entry fills ---")
    for account_name, config in ACCOUNTS.items():
        env = _load_env(config["env"])
        api = _get_api(env)
        state_file = config["state"]
        if not state_file.exists():
            continue

        with open(state_file) as f:
            state = json.load(f)

        pos = state.get("position")
        if not pos or not pos.get("buy_order_id"):
            continue

        # Check if we already reconciled this entry
        if pos.get("actual_entry_price"):
            continue

        buy_fill = _get_order_fill(api, pos["buy_order_id"])
        if buy_fill:
            actual = buy_fill["filled_avg_price"]
            est = pos["entry_price"]
            slip_bps = round((actual - est) / est * 10000, 1) if est > 0 else 0
            pos["actual_entry_price"] = actual
            pos["entry_fill_at"] = buy_fill["filled_at"]
            log.info(f"  {account_name} {pos['ticker']}: "
                     f"est=${est:.4f} -> actual=${actual:.4f} "
                     f"(slip={slip_bps:+.1f}bps)")
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2, default=str)


def main():
    log.info("=" * 60)
    log.info(f"Trade Reconciliation -- {datetime.now():%Y-%m-%d %H:%M}")
    log.info("=" * 60)

    _ensure_csv()

    # Step 1: Reconcile completed trades
    total = 0
    for account_name, config in ACCOUNTS.items():
        try:
            n = reconcile_account(account_name, config)
            total += n
        except Exception as e:
            log.error(f"  {account_name} failed: {e}")

    # Step 2: Reconcile open position entry fills
    try:
        reconcile_open_positions()
    except Exception as e:
        log.error(f"Open position reconciliation failed: {e}")

    log.info(f"Reconciled {total} new trades. CSV: {RECONCILED_CSV}")

    # Print summary
    if RECONCILED_CSV.exists():
        with open(RECONCILED_CSV, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        if reader:
            with_actual = [r for r in reader if r.get("actual_entry_price")]
            slips = [float(r["total_slippage_bps"]) for r in reader
                     if r.get("total_slippage_bps")]
            log.info(f"Total reconciled trades: {len(reader)}")
            log.info(f"  With actual fills: {len(with_actual)}")
            if slips:
                log.info(f"  Avg round-trip slippage: {sum(slips)/len(slips):+.1f}bps")
                log.info(f"  Max round-trip slippage: {max(slips, key=abs):+.1f}bps")


if __name__ == "__main__":
    main()
