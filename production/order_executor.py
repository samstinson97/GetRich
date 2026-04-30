#!/usr/bin/env python3
"""
order_executor.py -- Alpaca API wrapper for MOO order execution and reconciliation.
"""
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional

import alpaca_trade_api as tradeapi

logger = logging.getLogger(__name__)


class OrderExecutor:
    """Alpaca API wrapper for submitting MOO/market orders and reconciling fills."""

    def __init__(self):
        api_key = os.environ["ALPACA_API_KEY"]
        secret_key = os.environ["ALPACA_SECRET_KEY"]
        base_url = os.environ.get(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        )
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version="v2")
        logger.info("OrderExecutor initialized (base_url=%s)", base_url)

    # ------------------------------------------------------------------
    # Portfolio queries
    # ------------------------------------------------------------------
    def get_positions(self) -> Dict[str, dict]:
        """Return {ticker: {qty, market_value, avg_entry, unrealized_pl}}."""
        positions = {}
        for p in self.api.list_positions():
            positions[p.symbol] = {
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "avg_entry": float(p.avg_entry_price),
                "unrealized_pl": float(p.unrealized_pl),
            }
        return positions

    def get_portfolio_value(self) -> float:
        """Return current portfolio equity."""
        account = self.api.get_account()
        return float(account.equity)

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------
    def submit_buy_order(
        self, ticker: str, notional: float, reason: str
    ) -> Optional[str]:
        """Submit a MOO (market-on-open) buy order for a dollar amount.

        Returns the Alpaca order ID or None on failure.
        """
        try:
            order = self.api.submit_order(
                symbol=ticker,
                notional=round(notional, 2),
                side="buy",
                type="market",
                time_in_force="opg",  # MOO
            )
            logger.info(
                "BUY submitted: %s $%.2f | reason=%s | order_id=%s",
                ticker,
                notional,
                reason,
                order.id,
            )
            return order.id
        except Exception as e:
            logger.error("BUY failed: %s $%.2f | %s", ticker, notional, e)
            return None

    def submit_sell_order(
        self, ticker: str, qty: float, reason: str
    ) -> Optional[str]:
        """Submit a market sell order for a given quantity.

        Returns the Alpaca order ID or None on failure.
        """
        try:
            order = self.api.submit_order(
                symbol=ticker,
                qty=qty,
                side="sell",
                type="market",
                time_in_force="day",
            )
            logger.info(
                "SELL submitted: %s qty=%.4f | reason=%s | order_id=%s",
                ticker,
                qty,
                reason,
                order.id,
            )
            return order.id
        except Exception as e:
            logger.error("SELL failed: %s qty=%.4f | %s", ticker, qty, e)
            return None

    # ------------------------------------------------------------------
    # Fill checking & reconciliation
    # ------------------------------------------------------------------
    def check_fills(self) -> List[dict]:
        """Check today's filled orders. Returns list of fill dicts."""
        today = datetime.now().strftime("%Y-%m-%d")
        orders = self.api.list_orders(
            status="filled",
            after=f"{today}T00:00:00Z",
            direction="desc",
            limit=100,
        )
        fills = []
        for o in orders:
            fills.append(
                {
                    "order_id": o.id,
                    "symbol": o.symbol,
                    "side": o.side,
                    "qty": float(o.filled_qty),
                    "avg_price": float(o.filled_avg_price),
                    "filled_at": str(o.filled_at),
                }
            )
        logger.info("check_fills: %d fills today", len(fills))
        return fills

    def reconcile(
        self, intended: Dict[str, float]
    ) -> Dict[str, dict]:
        """Compare intended positions {ticker: target_qty} vs actual.

        Returns {ticker: {intended, actual, diff}} for mismatches.
        """
        actual_positions = self.get_positions()
        mismatches = {}

        all_tickers = set(intended.keys()) | set(actual_positions.keys())
        for ticker in all_tickers:
            target = intended.get(ticker, 0.0)
            actual = actual_positions.get(ticker, {}).get("qty", 0.0)
            if abs(target - actual) > 0.01:
                mismatches[ticker] = {
                    "intended": target,
                    "actual": actual,
                    "diff": target - actual,
                }

        if mismatches:
            logger.warning("Reconciliation mismatches: %s", mismatches)
        else:
            logger.info("Reconciliation OK — all positions match")

        return mismatches
