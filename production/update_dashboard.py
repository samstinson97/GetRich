#!/usr/bin/env python3
"""
Update dashboard_multi.json from all 3 account state files + Alpaca live data.
Then upload to S3.
"""
import os, sys, json, subprocess
from datetime import datetime
from pathlib import Path

PROD = Path(__file__).parent
ROOT = PROD.parent

def load_env(path):
    env = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env

def get_account_data(key, secret):
    import alpaca_trade_api as tradeapi
    api = tradeapi.REST(key, secret, "https://paper-api.alpaca.markets", api_version="v2")
    acct = api.get_account()
    positions = api.list_positions()
    orders = api.list_orders(status="all", limit=10)

    pos = None
    if positions:
        # Filter out dust positions (fractional remainders from GTC stops)
        real = [p for p in positions if float(p.qty) >= 0.01]
        if real:
            p = real[0]
            pos = {
                "ticker": p.symbol,
                "entry_price": float(p.avg_entry_price),
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "unrealized_pct": float(p.unrealized_plpc) * 100 if p.unrealized_plpc else 0,
                "current_price": float(p.current_price) if hasattr(p, "current_price") else float(p.market_value) / max(float(p.qty), 0.001),
            }

    return {
        "equity": float(acct.equity),
        "cash": float(acct.cash),
        "position_live": pos,
    }

# Load existing dashboard data
dash_path = PROD / "dashboard_multi.json"
with open(dash_path) as f:
    dash = json.load(f)

# Account configs
accounts_cfg = {
    "v4_gtc": {
        "env": PROD / "accounts" / "v4_gtc.env",
        "state": PROD / "ml_v4" / "state_gtc.json",
    },
    "v4_moc": {
        "env": PROD / "accounts" / "v4_moc.env",
        "state": PROD / "ml_v4" / "state_moc.json",
    },
    "v3_fresh": {
        "env": PROD / "accounts" / "v3_fresh.env",
        "state": PROD / "ml_v3" / "state_fresh.json",
    },
}

for acct_key, cfg in accounts_cfg.items():
    try:
        # Load Alpaca live data
        env = load_env(cfg["env"])
        live = get_account_data(env["ALPACA_API_KEY"], env["ALPACA_SECRET_KEY"])

        # Load state file
        state = {}
        if cfg["state"].exists():
            with open(cfg["state"]) as f:
                state = json.load(f)

        # Update dashboard account
        acct = dash["accounts"][acct_key]
        acct["equity"] = live["equity"]

        # Position from Alpaca (live) merged with state (entry details)
        state_pos = state.get("position")
        if live["position_live"]:
            lp = live["position_live"]
            acct["position"] = {
                "ticker": lp["ticker"],
                "entry_price": state_pos.get("entry_price", lp["entry_price"]) if state_pos else lp["entry_price"],
                "current_price": lp["current_price"],
                "unrealized_pct": lp["unrealized_pct"],
                "qty": lp["qty"],
            }
        else:
            acct["position"] = None

        # Trades from state
        acct["trades"] = state.get("trades", [])
        acct["n_trades"] = len(acct["trades"])
        if acct["n_trades"] > 0:
            wins = sum(1 for t in acct["trades"] if t.get("pnl_pct", 0) > 0)
            acct["win_rate"] = round(wins / acct["n_trades"] * 100, 1)
        else:
            acct["win_rate"] = 0

        # Equity curve from state
        state_curve = state.get("equity_curve", [])
        if state_curve:
            acct["equity_curve"] = state_curve
        # Add today's equity if not already there
        today = datetime.now().strftime("%Y-%m-%d")
        if acct["equity_curve"] and acct["equity_curve"][-1].get("date") != today:
            acct["equity_curve"].append({"date": today, "actual": live["equity"]})
        elif acct["equity_curve"] and acct["equity_curve"][-1].get("date") == today:
            acct["equity_curve"][-1]["actual"] = live["equity"]
        else:
            acct["equity_curve"] = [{"date": today, "actual": live["equity"]}]

        # Today's signal from state
        acct["today_signal"] = state.get("today_signal")

        print(f"{acct_key}: equity=${live['equity']:.2f} pos={live['position_live']['ticker'] if live['position_live'] else 'none'}")

    except Exception as e:
        print(f"{acct_key}: ERROR - {e}")

# Update metadata
dash["last_updated"] = datetime.now().isoformat()
has_trades = any(dash["accounts"][k].get("n_trades", 0) > 0 for k in dash["accounts"])
has_positions = any(dash["accounts"][k].get("position") for k in dash["accounts"])
dash["status"] = "LIVE" if has_positions else ("TRADED" if has_trades else "AWAITING_FIRST_TRADE")

# Save locally
with open(dash_path, "w") as f:
    json.dump(dash, f, indent=2, default=str)
print(f"\nDashboard updated: {dash_path}")

# Also write dashboard_data.json in the OLD format for the original dashboard.html
# Picks the best-performing account for the "account" summary section,
# and merges all 3 equity curves with per-account keys for multi-line chart.
old_dash_path = PROD / "dashboard_data.json"
try:
    # Find best account by equity
    best_key = max(dash["accounts"], key=lambda k: dash["accounts"][k].get("equity", 0))
    best = dash["accounts"][best_key]
    best_pos = best.get("position")

    # Merge equity curves: build unified date list, each account gets its own key
    all_dates = set()
    for acct_data in dash["accounts"].values():
        for pt in acct_data.get("equity_curve", []):
            all_dates.add(pt["date"])
    all_dates = sorted(all_dates)

    # Build per-date lookup for each account
    acct_maps = {}
    for k, acct_data in dash["accounts"].items():
        acct_maps[k] = {pt["date"]: pt["actual"] for pt in acct_data.get("equity_curve", [])}

    # Fetch SPY history and normalize to $1000 at the first paper-trading date.
    spy_map = {}
    try:
        import yfinance as yf
        if all_dates:
            start = all_dates[0]
            end_dt = datetime.strptime(all_dates[-1], "%Y-%m-%d") + \
                     __import__("datetime").timedelta(days=2)
            spy = yf.download("SPY", start=start, end=end_dt.strftime("%Y-%m-%d"),
                              progress=False, auto_adjust=True)
            if not spy.empty:
                close = spy["Close"]
                if hasattr(close, "iloc") and hasattr(close.iloc[0], "iloc"):
                    close = close.iloc[:, 0]
                base_px = float(close.iloc[0])
                for ts, px in close.items():
                    d = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
                    spy_map[d] = round(1000.0 * float(px) / base_px, 2)
                print(f"SPY benchmark: {len(spy_map)} points, "
                      f"start ${base_px:.2f}, last ${float(close.iloc[-1]):.2f}, "
                      f"normalized ${list(spy_map.values())[-1]:.2f}")
    except Exception as e:
        print(f"SPY benchmark fetch failed: {e}")

    merged_curve = []
    last_spy = None
    for d in all_dates:
        entry = {"date": d}
        entry["actual"] = acct_maps.get("v4_moc", {}).get(d)       # v4 MOC as "actual"
        entry["v4_gtc"] = acct_maps.get("v4_gtc", {}).get(d)
        entry["v4_moc"] = acct_maps.get("v4_moc", {}).get(d)
        entry["v3_fresh"] = acct_maps.get("v3_fresh", {}).get(d)
        # SPY: carry-forward over weekends/holidays so chart line is continuous
        if d in spy_map:
            last_spy = spy_map[d]
        entry["sp500"] = last_spy
        merged_curve.append(entry)

    # Collect all trades across accounts
    all_trades = []
    for k, acct_data in dash["accounts"].items():
        for t in acct_data.get("trades", []):
            t_copy = dict(t)
            t_copy["account"] = k
            all_trades.append(t_copy)
    all_trades.sort(key=lambda t: t.get("entry_date", ""))

    total_trades = sum(a.get("n_trades", 0) for a in dash["accounts"].values())

    old_dash = {
        "last_updated": dash["last_updated"],
        "status": dash["status"],
        "account": {
            "equity": round(best.get("equity", 0), 2),
            "cash": 0,
            "settling": 0,
            "starting": 1000,
        },
        "position": {
            "ticker": best_pos["ticker"] if best_pos else None,
            "entry_price": best_pos.get("entry_price") if best_pos else None,
            "days_held": 0,
            "max_hold": 6,
            "unrealized_pct": best_pos.get("unrealized_pct", 0) if best_pos else 0,
        } if best_pos else None,
        "today_signal": best.get("today_signal", {"status": "No signal"}),
        "equity_curve": merged_curve,
        "trades": all_trades,
        "monthly_returns": {},
        "yearly_summary": [],
        "alerts": [],
        "signals": {"generated_30d": 0, "trading_days_30d": 0},
        "tracking": {"cumulative_error": 0, "pick_agreement": 0},
        "risk": {
            "current_dd": min((a.get("equity", 1000) / 1000 - 1) for a in dash["accounts"].values()),
            "max_dd": 0,
            "consec_losers": 0,
            "win_rate": round(sum(a.get("win_rate", 0) for a in dash["accounts"].values()) / 3, 1),
        },
        "model": {"last_train_date": "2025-12-31", "days_since_retrain": 106, "next_retrain": "2027-01-01"},
        "accounts": dash["accounts"],  # pass through for 3-account cards
    }

    with open(old_dash_path, "w") as f:
        json.dump(old_dash, f, indent=2, default=str)
    print(f"Legacy dashboard_data.json updated (3 curves merged)")
except Exception as e:
    print(f"Legacy dashboard_data.json update failed: {e}")

# Upload to S3 (both JSON and HTML)
S3_BUCKET = "s3://getrich-ml-v3-601512582170"
uploads = [
    (str(dash_path), f"{S3_BUCKET}/dashboard_multi.json", "application/json"),
    (str(PROD / "dashboard_multi.html"), f"{S3_BUCKET}/dashboard_multi.html", "text/html"),
    (str(old_dash_path), f"{S3_BUCKET}/dashboard_data.json", "application/json"),
    (str(PROD / "dashboard.html"), f"{S3_BUCKET}/dashboard.html", "text/html"),
]
for local, remote, ctype in uploads:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "awscli", "s3", "cp",
             local, remote, "--content-type", ctype],
            capture_output=True, text=True, timeout=30)
        fname = local.split("\\")[-1].split("/")[-1]
        if result.returncode == 0:
            print(f"S3 upload {fname}: OK")
        else:
            print(f"S3 upload {fname} failed: {result.stderr[:100]}")
    except Exception as e:
        print(f"S3 upload error: {e}")
