#!/usr/bin/env python3
"""
paper_trade_tracker.py -- Track paper trading performance, compute go/no-go gates.

Reads Alpaca paper positions, parses log files, maintains tracking_log.csv
and trades_log.csv, and evaluates go/no-go gates for live trading readiness.
"""

import csv
import glob
import json
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import alpaca_trade_api as tradeapi

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

PRODUCTION_DIR = Path(__file__).parent
TRACKING_DIR = PRODUCTION_DIR / "tracking"
LOGS_DIR = PRODUCTION_DIR / "logs"
TRACKING_LOG = TRACKING_DIR / "tracking_log.csv"
TRADES_LOG = TRACKING_DIR / "trades_log.csv"

TRACKING_COLUMNS = [
    "date",
    "paper_portfolio_value",
    "paper_daily_return",
    "cumulative_tracking_error",
    "positions_held",
    "realized_slippage_avg",
]
TRADES_COLUMNS = [
    "ticker",
    "entry_date",
    "exit_date",
    "entry_price",
    "exit_price",
    "return_pct",
    "composite_score_at_entry",
    "exit_reason",
]

# Go/no-go gate thresholds (from config.yaml)
GATE_MAX_TRACKING_ERROR = 0.05   # 5%
GATE_MAX_SLIPPAGE = 0.002        # 0.20%
GATE_MIN_TRADES_30D = 20


class PaperTradeTracker:
    """Tracks paper trading performance against expectations."""

    def __init__(self):
        api_key = os.environ["ALPACA_API_KEY"]
        secret_key = os.environ["ALPACA_SECRET_KEY"]
        base_url = os.environ.get(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        )
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version="v2")
        TRACKING_DIR.mkdir(parents=True, exist_ok=True)
        self._ensure_csv(TRACKING_LOG, TRACKING_COLUMNS)
        self._ensure_csv(TRADES_LOG, TRADES_COLUMNS)

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_csv(path: Path, columns: List[str]) -> None:
        """Create CSV with headers if it doesn't exist."""
        if not path.exists():
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(columns)

    @staticmethod
    def _read_csv(path: Path) -> List[dict]:
        """Read CSV into list of dicts."""
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _append_csv(path: Path, row: dict, columns: List[str]) -> None:
        """Append a single row to CSV."""
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writerow(row)

    # ------------------------------------------------------------------
    # Alpaca portfolio data
    # ------------------------------------------------------------------

    def get_portfolio_value(self) -> float:
        """Get current paper portfolio equity."""
        account = self.api.get_account()
        return float(account.equity)

    def get_positions(self) -> Dict[str, dict]:
        """Get current paper positions."""
        positions = {}
        for p in self.api.list_positions():
            positions[p.symbol] = {
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "avg_entry": float(p.avg_entry_price),
                "unrealized_pl": float(p.unrealized_pl),
                "current_price": float(p.current_price),
            }
        return positions

    def get_closed_orders_today(self) -> List[dict]:
        """Get today's filled sell orders (potential trade exits)."""
        today = datetime.now().strftime("%Y-%m-%d")
        orders = self.api.list_orders(
            status="filled",
            after=f"{today}T00:00:00Z",
            direction="desc",
            limit=200,
        )
        return [
            {
                "symbol": o.symbol,
                "side": o.side,
                "qty": float(o.filled_qty),
                "avg_price": float(o.filled_avg_price),
                "filled_at": str(o.filled_at),
            }
            for o in orders
        ]

    # ------------------------------------------------------------------
    # Log file parsing
    # ------------------------------------------------------------------

    def parse_log_file(self, log_path: str) -> Optional[dict]:
        """Parse a daily JSON log file from production/logs/."""
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
            logger.warning("Could not parse log %s: %s", log_path, e)
            return None

    def get_latest_log(self) -> Optional[dict]:
        """Find and parse the most recent log file."""
        pattern = str(LOGS_DIR / "*.json")
        files = sorted(glob.glob(pattern))
        if not files:
            return None
        return self.parse_log_file(files[-1])

    def get_log_for_date(self, date_str: str) -> Optional[dict]:
        """Get log for a specific date (YYYY-MM-DD)."""
        path = LOGS_DIR / f"{date_str}.json"
        if path.exists():
            return self.parse_log_file(str(path))
        return None

    def extract_trades_from_logs(self) -> List[dict]:
        """Extract completed round-trip trades from log files.
        v8_live_trader.py logs have 'entries' and 'exits' keys."""
        trades = []
        pattern = str(LOGS_DIR / "*.json")
        for log_path in sorted(glob.glob(pattern)):
            log_data = self.parse_log_file(log_path)
            if not log_data:
                continue
            # v8_live_trader writes exits with entry_date, reason, etc.
            for ex in log_data.get("exits", []):
                trades.append(ex)
        return trades

    def extract_slippage_from_logs(self) -> List[float]:
        """Extract realized slippage values from log files."""
        slippages = []
        pattern = str(LOGS_DIR / "*.json")
        for log_path in sorted(glob.glob(pattern)):
            log_data = self.parse_log_file(log_path)
            if not log_data:
                continue
            # Check for slippage data in various log formats
            if "slippage" in log_data:
                val = log_data["slippage"]
                if isinstance(val, (int, float)):
                    slippages.append(float(val))
                elif isinstance(val, list):
                    slippages.extend(float(s) for s in val if isinstance(s, (int, float)))
            # Check fills for slippage
            for fill in log_data.get("fills", []):
                if "slippage" in fill:
                    slippages.append(float(fill["slippage"]))
                elif "expected_price" in fill and "avg_price" in fill:
                    expected = float(fill["expected_price"])
                    actual = float(fill["avg_price"])
                    if expected > 0:
                        slippages.append(abs(actual - expected) / expected)
        return slippages

    # ------------------------------------------------------------------
    # Tracking computations
    # ------------------------------------------------------------------

    def compute_daily_return(self, current_value: float) -> float:
        """Compute daily return vs previous day's portfolio value."""
        rows = self._read_csv(TRACKING_LOG)
        if not rows:
            return 0.0
        prev_value = float(rows[-1]["paper_portfolio_value"])
        if prev_value <= 0:
            return 0.0
        return (current_value - prev_value) / prev_value

    def compute_tracking_error(self) -> float:
        """Compute cumulative tracking error from daily returns.

        Tracking error = std(paper_returns - benchmark_returns).
        Without a live benchmark feed, we use std(daily_returns) as a proxy
        for deviation from expected strategy behavior.
        """
        rows = self._read_csv(TRACKING_LOG)
        if len(rows) < 2:
            return 0.0
        returns = []
        for r in rows:
            try:
                returns.append(float(r["paper_daily_return"]))
            except (ValueError, KeyError):
                continue
        if len(returns) < 2:
            return 0.0
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        return variance ** 0.5

    def compute_avg_slippage(self) -> float:
        """Compute average realized slippage across all logged trades."""
        slippages = self.extract_slippage_from_logs()
        if not slippages:
            return 0.0
        return sum(slippages) / len(slippages)

    def count_trades_last_30d(self) -> int:
        """Count completed round-trip trades in the last 30 calendar days."""
        cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        rows = self._read_csv(TRADES_LOG)
        count = 0
        for r in rows:
            exit_date = r.get("exit_date", "")
            if exit_date >= cutoff:
                count += 1
        return count

    # ------------------------------------------------------------------
    # Trade logging
    # ------------------------------------------------------------------

    def log_round_trip(
        self,
        ticker: str,
        entry_date: str,
        exit_date: str,
        entry_price: float,
        exit_price: float,
        composite_score_at_entry: float,
        exit_reason: str,
    ) -> None:
        """Log a completed round-trip trade."""
        return_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0.0
        row = {
            "ticker": ticker,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": f"{entry_price:.4f}",
            "exit_price": f"{exit_price:.4f}",
            "return_pct": f"{return_pct:.2f}",
            "composite_score_at_entry": f"{composite_score_at_entry:.1f}",
            "exit_reason": exit_reason,
        }
        self._append_csv(TRADES_LOG, row, TRADES_COLUMNS)
        logger.info(
            "Round-trip logged: %s %s->%s %.2f%%",
            ticker, entry_date, exit_date, return_pct,
        )

    def log_round_trips_from_log(self, log_data: dict) -> int:
        """Extract and log round-trips from a daily log file. Returns count.
        v8_live_trader writes 'exits' list with entry_date, reason, entry_score."""
        logged = 0
        existing = self._read_csv(TRADES_LOG)
        existing_keys = {
            (r["ticker"], r["entry_date"], r.get("exit_date", ""))
            for r in existing
        }
        log_date = log_data.get("date", "")
        for ex in log_data.get("exits", []):
            ticker = ex.get("ticker", "")
            entry_date = ex.get("entry_date", "")
            exit_date = log_date  # exit happens on the log date
            key = (ticker, entry_date, exit_date)
            if key in existing_keys:
                continue
            self.log_round_trip(
                ticker=ticker,
                entry_date=entry_date,
                exit_date=exit_date,
                entry_price=0.0,  # fill price comes from Alpaca, not log
                exit_price=0.0,
                composite_score_at_entry=float(ex.get("entry_score", 0)),
                exit_reason=ex.get("reason", "unknown"),
            )
            logged += 1
        return logged

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self) -> dict:
        """Run the full daily tracking update. Returns summary dict."""
        today = datetime.now().strftime("%Y-%m-%d")
        logger.info("=== Paper Trade Tracker Update: %s ===", today)

        # Check if already logged today
        existing = self._read_csv(TRACKING_LOG)
        if existing and existing[-1].get("date") == today:
            logger.info("Already tracked today, skipping duplicate entry")
            return self._build_summary(existing[-1])

        # Get portfolio state from Alpaca
        portfolio_value = self.get_portfolio_value()
        positions = self.get_positions()
        daily_return = self.compute_daily_return(portfolio_value)

        # Process log file for today's trades
        log_data = self.get_log_for_date(today) or self.get_latest_log() or {}
        if log_data:
            new_trades = self.log_round_trips_from_log(log_data)
            logger.info("Logged %d new round-trips from logs", new_trades)

        # Compute metrics
        avg_slippage = self.compute_avg_slippage()
        tracking_error = self.compute_tracking_error()

        # Write tracking row
        row = {
            "date": today,
            "paper_portfolio_value": f"{portfolio_value:.2f}",
            "paper_daily_return": f"{daily_return:.6f}",
            "cumulative_tracking_error": f"{tracking_error:.6f}",
            "positions_held": len(positions),
            "realized_slippage_avg": f"{avg_slippage:.6f}",
        }
        self._append_csv(TRACKING_LOG, row, TRACKING_COLUMNS)
        logger.info("Tracking row written: value=$%.2f, return=%.4f%%", portfolio_value, daily_return * 100)

        return self._build_summary(row)

    def _build_summary(self, row: dict) -> dict:
        """Build a summary dict from a tracking row."""
        tracking_error = float(row.get("cumulative_tracking_error", 0))
        avg_slippage = float(row.get("realized_slippage_avg", 0))
        trades_30d = self.count_trades_last_30d()

        gates = self.evaluate_gates(tracking_error, avg_slippage, trades_30d)

        return {
            "date": row["date"],
            "portfolio_value": float(row["paper_portfolio_value"]),
            "daily_return": float(row["paper_daily_return"]),
            "tracking_error": tracking_error,
            "positions_held": int(row["positions_held"]),
            "avg_slippage": avg_slippage,
            "trades_30d": trades_30d,
            "gates": gates,
        }

    # ------------------------------------------------------------------
    # Go/No-Go gates
    # ------------------------------------------------------------------

    def evaluate_gates(
        self,
        tracking_error: float,
        avg_slippage: float,
        trades_30d: int,
    ) -> dict:
        """Evaluate go/no-go gates for live trading readiness.

        Returns dict with gate status and overall decision.
        """
        te_ok = tracking_error < GATE_MAX_TRACKING_ERROR
        slip_ok = avg_slippage < GATE_MAX_SLIPPAGE
        trades_ok = trades_30d >= GATE_MIN_TRADES_30D

        gates = {
            "tracking_error": {
                "value": tracking_error,
                "threshold": GATE_MAX_TRACKING_ERROR,
                "pass": te_ok,
                "label": f"{tracking_error:.2%} < {GATE_MAX_TRACKING_ERROR:.0%}",
            },
            "avg_slippage": {
                "value": avg_slippage,
                "threshold": GATE_MAX_SLIPPAGE,
                "pass": slip_ok,
                "label": f"{avg_slippage:.3%} < {GATE_MAX_SLIPPAGE:.2%}",
            },
            "min_trades_30d": {
                "value": trades_30d,
                "threshold": GATE_MIN_TRADES_30D,
                "pass": trades_ok,
                "label": f"{trades_30d} >= {GATE_MIN_TRADES_30D}",
            },
        }

        all_pass = te_ok and slip_ok and trades_ok
        gates["overall"] = "GO" if all_pass else "NO-GO"

        logger.info(
            "Gates: TE=%s SLIP=%s TRADES=%s => %s",
            "PASS" if te_ok else "FAIL",
            "PASS" if slip_ok else "FAIL",
            "PASS" if trades_ok else "FAIL",
            gates["overall"],
        )
        return gates

    def get_gate_summary(self) -> dict:
        """Get current gate status without updating tracking log."""
        tracking_error = self.compute_tracking_error()
        avg_slippage = self.compute_avg_slippage()
        trades_30d = self.count_trades_last_30d()
        return self.evaluate_gates(tracking_error, avg_slippage, trades_30d)


def main():
    """CLI entry point — run daily tracking update."""
    tracker = PaperTradeTracker()
    summary = tracker.update()

    print(f"\n{'='*50}")
    print(f"Paper Trade Tracker — {summary['date']}")
    print(f"{'='*50}")
    print(f"Portfolio Value:  ${summary['portfolio_value']:,.2f}")
    print(f"Daily Return:     {summary['daily_return']:.4%}")
    print(f"Tracking Error:   {summary['tracking_error']:.4%}")
    print(f"Positions Held:   {summary['positions_held']}")
    print(f"Avg Slippage:     {summary['avg_slippage']:.4%}")
    print(f"Trades (30d):     {summary['trades_30d']}")
    print(f"\nGo/No-Go: {summary['gates']['overall']}")
    for name, gate in summary["gates"].items():
        if name == "overall":
            continue
        status = "PASS" if gate["pass"] else "FAIL"
        print(f"  {name}: {gate['label']} [{status}]")


if __name__ == "__main__":
    main()
