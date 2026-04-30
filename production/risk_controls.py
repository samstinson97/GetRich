"""Risk controls for the GetRich trading system."""

import json
import os
import datetime
from pathlib import Path

STATE_DIR = Path(__file__).parent / "state"
COOLDOWN_FILE = STATE_DIR / "cooldown.json"

# 2026 NYSE holidays (market closed)
NYSE_HOLIDAYS_2026 = {
    datetime.date(2026, 1, 1),   # New Year's Day
    datetime.date(2026, 1, 19),  # MLK Jr. Day
    datetime.date(2026, 2, 16),  # Presidents' Day
    datetime.date(2026, 4, 3),   # Good Friday
    datetime.date(2026, 5, 25),  # Memorial Day
    datetime.date(2026, 7, 3),   # Independence Day (observed)
    datetime.date(2026, 9, 7),   # Labor Day
    datetime.date(2026, 11, 26), # Thanksgiving
    datetime.date(2026, 12, 25), # Christmas
}


class RiskManager:
    """Pre-trade risk gate that must approve every order."""

    def __init__(self):
        STATE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_position_size(self, order_notional: float, portfolio_value: float) -> tuple[bool, str]:
        """Reject if a single position exceeds 40% of portfolio."""
        if portfolio_value <= 0:
            return False, "Portfolio value must be positive"
        ratio = order_notional / portfolio_value
        if ratio > 0.40:
            return False, f"Position size {ratio:.1%} exceeds 40% limit"
        return True, "Position size OK"

    def check_daily_loss(self, current_value: float, start_value: float) -> tuple[bool, str]:
        """Trigger a 2-day cooldown if intraday loss exceeds 8%."""
        if start_value <= 0:
            return False, "Start value must be positive"
        loss = (start_value - current_value) / start_value
        # Check existing cooldown first
        cooldown_end = self._read_cooldown()
        today = datetime.date.today()
        if cooldown_end and today < cooldown_end:
            return False, f"In cooldown until {cooldown_end.isoformat()}"
        if loss > 0.08:
            self._set_cooldown(days=2)
            return False, f"Daily loss {loss:.1%} exceeds 8% — 2-day cooldown activated"
        return True, "Daily loss within limits"

    def check_leverage(self, total_exposure: float, portfolio_value: float,
                       max_leverage: float = 2.0) -> tuple[bool, str]:
        """Reject if total exposure / portfolio value exceeds account margin limit."""
        if portfolio_value <= 0:
            return False, "Portfolio value must be positive"
        lev = total_exposure / portfolio_value
        if lev > max_leverage:
            return False, f"Leverage {lev:.2f}x exceeds {max_leverage:.1f}x account limit"
        return True, "Leverage OK"

    def check_data_freshness(self, file_timestamp: datetime.datetime) -> tuple[bool, str]:
        """Abort if data file is more than 2 hours old."""
        age = datetime.datetime.now() - file_timestamp
        if age.total_seconds() > 2 * 3600:
            return False, f"Data is {age.total_seconds()/3600:.1f}h old — exceeds 2h limit"
        return True, "Data is fresh"

    def check_market_open(self, date: datetime.date | None = None) -> tuple[bool, str]:
        """Check whether the NYSE is open on the given date (2026 calendar)."""
        if date is None:
            date = datetime.date.today()
        if date.weekday() >= 5:
            return False, f"{date.isoformat()} is a weekend"
        if date in NYSE_HOLIDAYS_2026:
            return False, f"{date.isoformat()} is an NYSE holiday"
        return True, "Market is open"

    def pre_trade_check(
        self,
        order_notional: float,
        portfolio_value: float,
        current_value: float,
        start_value: float,
        total_exposure: float,
        file_timestamp: datetime.datetime,
        date: datetime.date | None = None,
    ) -> tuple[bool, str]:
        """Run all risk checks. Returns (approved, reason)."""
        checks = [
            self.check_market_open(date),
            self.check_data_freshness(file_timestamp),
            self.check_daily_loss(current_value, start_value),
            self.check_leverage(total_exposure, portfolio_value),
            self.check_position_size(order_notional, portfolio_value),
        ]
        for ok, reason in checks:
            if not ok:
                return False, reason
        return True, "All risk checks passed"

    # ------------------------------------------------------------------
    # Cooldown state helpers
    # ------------------------------------------------------------------

    def _read_cooldown(self) -> datetime.date | None:
        if not COOLDOWN_FILE.exists():
            return None
        with open(COOLDOWN_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        end = data.get("cooldown_end")
        if end:
            return datetime.date.fromisoformat(end)
        return None

    def _set_cooldown(self, days: int = 2) -> None:
        end = datetime.date.today() + datetime.timedelta(days=days)
        with open(COOLDOWN_FILE, "w", encoding="utf-8") as f:
            json.dump({"cooldown_end": end.isoformat()}, f)

    def _clear_cooldown(self) -> None:
        """Remove cooldown state (for testing)."""
        if COOLDOWN_FILE.exists():
            COOLDOWN_FILE.unlink()


if __name__ == "__main__":
    rm = RiskManager()
    passed = 0
    total = 0

    def check(label, expected, result):
        global passed, total
        total += 1
        ok, reason = result
        status = "PASS" if ok == expected else "FAIL"
        if ok == expected:
            passed += 1
        print(f"  [{status}] {label}: ok={ok} reason={reason}")

    print("=== Risk Controls Self-Test ===\n")

    # Clear any leftover cooldown
    rm._clear_cooldown()

    print("Position size:")
    check("39% of $10k (should pass)", True, rm.check_position_size(3900, 10000))
    check("41% of $10k (should fail)", False, rm.check_position_size(4100, 10000))

    print("\nDaily loss:")
    check("7% loss (should pass)", True, rm.check_daily_loss(9300, 10000))
    check("9% loss (should fail)", False, rm.check_daily_loss(9100, 10000))
    # Clear cooldown set by the 9% test
    rm._clear_cooldown()

    print("\nLeverage (2.0x account limit):")
    check("1.9x (should pass)", True, rm.check_leverage(19000, 10000))
    check("2.1x (should fail)", False, rm.check_leverage(21000, 10000))

    print("\nData freshness:")
    fresh = datetime.datetime.now() - datetime.timedelta(hours=1)
    stale = datetime.datetime.now() - datetime.timedelta(hours=3)
    check("1hr old (should pass)", True, rm.check_data_freshness(fresh))
    check("3hr old (should fail)", False, rm.check_data_freshness(stale))

    print("\nMarket open:")
    monday = datetime.date(2026, 3, 16)  # Monday
    saturday = datetime.date(2026, 3, 21)  # Saturday
    check("Monday (should pass)", True, rm.check_market_open(monday))
    check("Saturday (should fail)", False, rm.check_market_open(saturday))

    print(f"\n=== {passed}/{total} checks passed ===")
