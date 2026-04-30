#!/usr/bin/env python3
"""
Place pending GTC stop orders at market open.
Runs at 9:35 AM ET (6:35 AM PT) daily via Task Scheduler.

Reads state_gtc.json for any pending_stop entries and submits
them to Alpaca. By this time the buy order has settled overnight.
"""
import os, sys, json, logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

PROD = Path(__file__).parent
ROOT = PROD.parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

# Load GTC account keys
env_file = PROD.parent / "accounts" / "v4_gtc.env"
with open(env_file) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(PROD.parent / "logs" / f"stops_{datetime.now():%Y%m%d}.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("place_stops")

STATE_FILE = PROD / "state_gtc.json"


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def main():
    log.info("=" * 50)
    log.info(f"Pending stop check -- {datetime.now():%Y-%m-%d %H:%M}")
    log.info("=" * 50)

    state = load_state()
    held = state.get("position")

    if not held:
        log.info("No position -- nothing to do")
        return

    pending = held.get("pending_stop")
    existing_stop = held.get("stop_order_id")

    if not pending:
        if existing_stop:
            log.info(f"Stop already placed (order {existing_stop})")
        else:
            log.info("No pending stop and no existing stop -- position unprotected")
        return

    tk = held["ticker"]
    stop_price = pending["price"]
    qty = pending["qty"]

    log.info(f"Placing GTC stop for {tk}: {qty:.2f} shares @ ${stop_price:.2f}")

    import alpaca_trade_api as tradeapi
    import math

    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")
    base = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    api = tradeapi.REST(key, secret, base, api_version="v2")

    # Verify position exists in Alpaca
    try:
        positions = {p.symbol: p for p in api.list_positions()}
        if tk not in positions:
            log.warning(f"{tk} not in Alpaca positions -- clearing pending stop")
            held.pop("pending_stop", None)
            save_state(state)
            return
        actual_qty = float(positions[tk].qty)
        log.info(f"Alpaca confirms {tk}: {actual_qty:.4f} shares")
    except Exception as e:
        log.error(f"Failed to check positions: {e}")
        return

    # Place stop order (whole shares only for GTC)
    whole_qty = math.floor(actual_qty)
    if whole_qty <= 0:
        log.warning(f"Only {actual_qty:.4f} fractional shares -- cannot place GTC stop")
        held.pop("pending_stop", None)
        save_state(state)
        return

    try:
        order = api.submit_order(
            symbol=tk,
            qty=whole_qty,
            side="sell",
            type="stop",
            stop_price=round(stop_price, 2),
            time_in_force="gtc",
        )
        log.info(f"GTC STOP PLACED: {tk} qty={whole_qty} @ ${stop_price:.2f} "
                 f"-> order_id={order.id}")
        held["stop_order_id"] = order.id
        held.pop("pending_stop", None)
        save_state(state)
    except Exception as e:
        log.error(f"STOP FAILED: {tk} @ ${stop_price:.2f} -> {e}")
        log.info("Will retry on next run")


if __name__ == "__main__":
    main()
