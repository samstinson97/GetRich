#!/usr/bin/env python3
"""
ML v4 #1188 Seed 6 -- Live Signal Generator
Runs daily at 3:50 PM ET. Scores full universe, picks #1 stock, submits MOC order.

Key differences from v3:
- 77 features (22 raw + 55 expanded) vs v3's 45
- Expanded features: 10 interactions, 10 polynomials, 22 cross-sectional ranks,
  10 cross-sectional z-scores (ddof=0!), 3 ratios
- Different exit params: 7% stop, 6d max hold, 0.66 score exit, etc.
- Supports both GTC and MOC stop modes via config
"""
import os, sys, json, logging, pickle, time, yaml
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Setup
PROD = Path(__file__).parent  # production/ml_v4/
PROD_ROOT = PROD.parent       # production/
ROOT = PROD_ROOT.parent       # ai-research-team/
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

# Stop mode must be set BEFORE logging (used in log filename)
_stop_mode = os.environ.get("V4_STOP_MODE", "moc")

import numpy as np
import pandas as pd
import lightgbm as lgb
from shap_monitor import run_shap_monitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(PROD_ROOT / "logs" / f"signal_v4_{_stop_mode}_{datetime.now():%Y%m%d}.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("ml_v4_signal")

# Load config
with open(PROD / "config.yaml") as f:
    CFG = yaml.safe_load(f)
STATE_FILE = PROD / f"state_{_stop_mode}.json"
if not STATE_FILE.exists():
    import json as _json
    with open(STATE_FILE, "w") as _f:
        _json.dump({"position": None, "trades": [], "equity_curve": [], "today_signal": None}, _f, indent=2)

# OHLCV cache -- SHARED between v4 accounts (same data = same signals)
# Race condition prevented by staggered launch: GTC writes first, MOC reads after
CACHE_FILE = PROD / "ohlcv_cache.pkl"

# Shared scoring state -- ensures GTC and MOC see identical features/scores.
# GTC (V4_SCORING_MODE=write) saves feat_dict + market_ctx after computation.
# MOC (V4_SCORING_MODE=read) loads it, skipping independent data fetch entirely.
# This prevents Alpaca snapshot timing differences from producing divergent scores.
SHARED_SCORING_FILE = PROD / "shared_scoring_state.pkl"
_scoring_mode = os.environ.get("V4_SCORING_MODE", "")  # "write", "read", or ""

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"position": None, "trades": [], "equity_curve": []}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _save_shared_scoring(feat_dict, market_ctx):
    """Save scoring state so the MOC account can load identical data."""
    try:
        with open(SHARED_SCORING_FILE, "wb") as f:
            pickle.dump({
                "date": datetime.now().date(),
                "feat_dict": feat_dict,
                "market_ctx": market_ctx,
            }, f)
        log.info("Saved shared scoring state for MOC account")
    except Exception as e:
        log.warning(f"Failed to save shared scoring state: {e}")


def _load_shared_scoring():
    """Load scoring state saved by the GTC account. Returns dict or None."""
    if not SHARED_SCORING_FILE.exists():
        return None
    try:
        with open(SHARED_SCORING_FILE, "rb") as f:
            shared = pickle.load(f)
        if shared.get("date") != datetime.now().date():
            log.warning(f"Shared scoring state is stale ({shared.get('date')}), ignoring")
            return None
        log.info(f"Loaded shared scoring state ({len(shared['feat_dict'])} stocks)")
        return shared
    except Exception as e:
        log.warning(f"Failed to load shared scoring state: {e}")
        return None


# =====================================================================
# ALPACA API
# =====================================================================
def get_alpaca_api():
    import alpaca_trade_api as tradeapi
    # Support both env var names (wrapper uses ALPACA_API_KEY, .env may use either)
    key = os.environ.get("ALPACA_API_KEY", os.environ.get("APCA_API_KEY_ID", ""))
    secret = os.environ.get("ALPACA_SECRET_KEY", os.environ.get("APCA_API_SECRET_KEY", ""))
    base = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    return tradeapi.REST(key, secret, base, api_version="v2")


def get_account_info(api):
    acct = api.get_account()
    return {
        "equity": float(acct.equity),
        "cash": float(acct.cash),
        "buying_power": float(acct.buying_power),
    }


def get_positions(api):
    positions = {}
    for p in api.list_positions():
        positions[p.symbol] = {
            "qty": float(p.qty),
            "avg_entry": float(p.avg_entry_price),
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl),
        }
    return positions


def submit_moc_buy(api, ticker, notional, dry_run=False):
    """Submit buy order at 3:50 PM. Uses 'day' TIF for fractional/notional orders
    (Alpaca requires DAY for notional orders, not CLS)."""
    if dry_run:
        log.info(f"[DRY-RUN] WOULD BUY {ticker} notional=${notional:.2f}")
        return "DRY-RUN"
    try:
        order = api.submit_order(
            symbol=ticker,
            notional=round(notional, 2),
            side="buy",
            type="market",
            time_in_force="day",  # Alpaca requires DAY for notional/fractional orders
        )
        log.info(f"ORDER SUBMITTED: BUY {ticker} ${notional:.2f} MOC -> order_id={order.id}")
        return order.id
    except Exception as e:
        log.error(f"ORDER FAILED: BUY {ticker} ${notional:.2f} -> {e}")
        return None


def submit_moc_sell(api, ticker, qty, reason, dry_run=False):
    """Submit sell order. Uses 'day' TIF for compatibility with fractional shares."""
    import math
    if dry_run:
        log.info(f"[DRY-RUN] WOULD SELL {ticker} qty={qty:.4f} reason={reason}")
        return "DRY-RUN"
    try:
        # Floor to 4 decimals to never exceed available shares (round can overshoot)
        sell_qty = math.floor(qty * 10000) / 10000
        if sell_qty <= 0:
            log.warning(f"SELL SKIPPED: {ticker} qty={qty:.6f} floors to 0")
            return None
        order = api.submit_order(
            symbol=ticker,
            qty=sell_qty,
            side="sell",
            type="market",
            time_in_force="day",  # DAY for fractional compatibility
        )
        log.info(f"ORDER SUBMITTED: SELL {ticker} qty={qty:.4f} MOC reason={reason} -> order_id={order.id}")
        return order.id
    except Exception as e:
        log.error(f"ORDER FAILED: SELL {ticker} qty={qty:.4f} -> {e}")
        return None


def submit_stop_order(api, ticker, qty, stop_price, dry_run=False):
    """Submit GTC stop order. Uses whole shares (floor) because Alpaca
    rejects GTC orders for fractional quantities."""
    import math
    whole_qty = math.floor(qty)
    if whole_qty <= 0:
        log.warning(f"STOP SKIPPED: {ticker} qty={qty:.4f} rounds to 0 whole shares")
        return None
    fractional_remainder = qty - whole_qty
    if fractional_remainder > 0.001:
        log.info(f"  Stop uses {whole_qty} whole shares ({fractional_remainder:.4f} fractional unprotected)")
    if dry_run:
        log.info(f"[DRY-RUN] WOULD PLACE STOP {ticker} qty={whole_qty} @ ${stop_price:.2f}")
        return "DRY-RUN"
    try:
        order = api.submit_order(
            symbol=ticker,
            qty=whole_qty,
            side="sell",
            type="stop",
            stop_price=round(stop_price, 2),
            time_in_force="gtc",
        )
        log.info(f"STOP PLACED: {ticker} qty={whole_qty} @ ${stop_price:.2f} -> order_id={order.id}")
        return order.id
    except Exception as e:
        log.error(f"STOP FAILED: {ticker} @ ${stop_price:.2f} -> {e}")
        return None


def cancel_order(api, order_id, dry_run=False):
    """Cancel an existing order."""
    if dry_run or order_id in (None, "DRY-RUN"):
        return
    try:
        api.cancel_order(order_id)
        log.info(f"ORDER CANCELLED: {order_id}")
    except Exception as e:
        log.warning(f"CANCEL FAILED: {order_id} -> {e}")


# =====================================================================
# DATA + FEATURE COMPUTATION
# =====================================================================
def _yf_batch_download(tickers, batch_size=50, **yf_kwargs):
    """Download OHLCV from yfinance in batches.
    Returns dict[ticker] -> list[bar_dict].
    Accepts any yf.download kwargs (period, start, end, etc.)."""
    import yfinance as yf
    result = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            raw = yf.download(batch, auto_adjust=True, threads=True,
                              progress=False, group_by="ticker", **yf_kwargs)
            if raw.empty:
                continue
            if len(batch) == 1:
                tk = batch[0]
                if "Close" in raw.columns and len(raw) > 0:
                    bars = []
                    for idx, row in raw.iterrows():
                        dt = idx.date() if hasattr(idx, "date") else idx
                        bars.append({
                            "date": dt, "open": float(row["Open"]),
                            "high": float(row["High"]), "low": float(row["Low"]),
                            "close": float(row["Close"]),
                            "volume": int(row["Volume"]),
                        })
                    result[tk] = bars
            else:
                for tk in batch:
                    try:
                        df = raw[tk].dropna(subset=["Close"])
                        if len(df) == 0:
                            continue
                        bars = []
                        for idx, row in df.iterrows():
                            dt = idx.date() if hasattr(idx, "date") else idx
                            bars.append({
                                "date": dt, "open": float(row["Open"]),
                                "high": float(row["High"]),
                                "low": float(row["Low"]),
                                "close": float(row["Close"]),
                                "volume": int(row["Volume"]),
                            })
                        result[tk] = bars
                    except Exception:
                        pass
        except Exception as e:
            log.warning(f"  yfinance batch {i // batch_size} failed: {e}")
    return result


def fetch_ohlcv(api, tickers, days=380):
    """Fetch OHLCV with local cache. Only new/missing data is downloaded.

    Cache strategy:
    - ohlcv_cache.pkl stores all bars through last run
    - Warm run: load cache, fetch 5d from yfinance to update (~30-60s)
    - Cold run: full 380d download for all tickers (~11min), then cache
    - Today's bar: yfinance consolidated H/L + Alpaca current price

    This avoids the IEX-only H/L problem: Alpaca free tier snapshots have IEX
    daily bar H/L which misses extreme prints on other exchanges (~2-3% of volume).
    yfinance gives consolidated H/L matching what the backtest used.
    """
    today = datetime.now().date()
    end = datetime.now()
    start_full = end - timedelta(days=days * 2)

    # --- Load cache ---
    cache = {}
    cache_last_date = None
    if CACHE_FILE.exists():
        try:
            t0 = time.time()
            with open(CACHE_FILE, "rb") as f:
                cache_data = pickle.load(f)
            cache = cache_data.get("bars", {})
            cache_last_date = cache_data.get("last_date")
            log.info(f"  Cache loaded: {len(cache)} tickers, "
                     f"last_date={cache_last_date} ({time.time()-t0:.1f}s)")
        except Exception as e:
            log.warning(f"  Cache load failed ({e}), full download")
            cache = {}

    # --- Partition tickers ---
    cached_tickers = [tk for tk in tickers if tk in cache and len(cache[tk]) > 0]
    missing_tickers = [tk for tk in tickers if tk not in cache or not cache.get(tk)]

    all_data = {}
    yf_today_hl = {}

    # --- Cached tickers: load history + 5d refresh ---
    if cached_tickers:
        for tk in cached_tickers:
            all_data[tk] = list(cache[tk])

        stale = cache_last_date is None or (today - cache_last_date).days > 10
        if stale:
            log.info(f"  Cache stale ({cache_last_date}), fetching from that date...")
            start_str = (cache_last_date.strftime("%Y-%m-%d")
                         if cache_last_date else start_full.strftime("%Y-%m-%d"))
            end_str = (end + timedelta(days=1)).strftime("%Y-%m-%d")
            fresh = _yf_batch_download(cached_tickers, batch_size=50,
                                       start=start_str, end=end_str)
        else:
            log.info(f"  Fetching 5d update for {len(cached_tickers)} cached tickers...")
            t0 = time.time()
            fresh = _yf_batch_download(cached_tickers, batch_size=200, period="5d")
            log.info(f"  5d update: {len(fresh)} tickers in {time.time()-t0:.1f}s")

        for tk, fresh_bars in fresh.items():
            if tk not in all_data:
                continue
            fresh_dates = {b["date"] for b in fresh_bars}
            all_data[tk] = [b for b in all_data[tk] if b["date"] not in fresh_dates]
            for bar in fresh_bars:
                all_data[tk].append(bar)
                if bar["date"] == today:
                    yf_today_hl[tk] = {"high": bar["high"], "low": bar["low"]}

    # --- Missing tickers: full history ---
    if missing_tickers:
        log.info(f"  {len(missing_tickers)} new tickers, fetching full {days}d...")
        t0 = time.time()
        full = _yf_batch_download(
            missing_tickers, batch_size=50,
            start=start_full.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"))
        log.info(f"  Full history: {len(full)} tickers in {time.time()-t0:.1f}s")
        for tk, bars in full.items():
            all_data[tk] = bars
            for bar in bars:
                if bar["date"] == today:
                    yf_today_hl[tk] = {"high": bar["high"], "low": bar["low"]}

    # --- Sort all bars by date ---
    for tk in all_data:
        all_data[tk].sort(key=lambda b: b["date"])

    log.info(f"  yfinance: {len(all_data)} tickers, {len(yf_today_hl)} with today's bar")

    # Step 2: Alpaca snapshots for real-time current price
    # We ONLY use latest_trade.p from Alpaca (most accurate current price).
    # H/L come from yfinance (consolidated) -- NOT from Alpaca's IEX-only bar.
    log.info("  Fetching Alpaca snapshots (current price only)...")
    snapshot_count = 0
    snapshot_failed = 0

    batch_size_snap = 100
    tickers_with_data = [tk for tk in tickers if tk in all_data]
    for i in range(0, len(tickers_with_data), batch_size_snap):
        batch = tickers_with_data[i:i+batch_size_snap]
        clean = {tk.replace("-", "."): tk for tk in batch}
        try:
            snapshots = api.get_snapshots(list(clean.keys()))
            for alpaca_tk, snap in snapshots.items():
                orig_tk = clean.get(alpaca_tk, alpaca_tk)
                if orig_tk not in all_data:
                    continue
                try:
                    lt = snap.latest_trade
                    db = snap.daily_bar
                    if lt is None and db is None:
                        continue
                    current_price = float(lt.p) if lt else float(db.c)

                    # Build today's bar: yfinance H/L (consolidated) + Alpaca price
                    existing = all_data[orig_tk]
                    yf_hl = yf_today_hl.get(orig_tk)

                    if yf_hl:
                        # Best case: yfinance gave us consolidated H/L
                        snap_bar = {
                            "date": today,
                            "open": existing[-1]["open"] if existing and existing[-1]["date"] == today else float(db.o) if db else current_price,
                            "high": max(yf_hl["high"], current_price),
                            "low": min(yf_hl["low"], current_price),
                            "close": current_price,
                            "volume": int(db.v) if db else (existing[-1]["volume"] if existing and existing[-1]["date"] == today else 0),
                        }
                    else:
                        # Fallback: no yfinance today bar, use Alpaca IEX bar
                        snap_bar = {
                            "date": today,
                            "open": float(db.o) if db else current_price,
                            "high": max(float(db.h), current_price) if db else current_price,
                            "low": min(float(db.l), current_price) if db else current_price,
                            "close": current_price,
                            "volume": int(db.v) if db else 0,
                        }

                    # Append or replace today's bar
                    if existing and existing[-1]["date"] == today:
                        existing[-1] = snap_bar
                    else:
                        existing.append(snap_bar)
                    snapshot_count += 1
                except Exception:
                    snapshot_failed += 1
        except Exception as e:
            log.warning(f"  Alpaca snapshot batch {i} failed: {e}")
            snapshot_failed += len(batch)

    log.info(f"  Alpaca snapshots: {snapshot_count} updated, {snapshot_failed} failed")
    log.info(f"  Today H/L source: {len(yf_today_hl)} consolidated (yfinance), "
             f"{snapshot_count - len(yf_today_hl)} IEX fallback (Alpaca)")

    # --- Save cache (all bars including today's, for next run) ---
    try:
        t0 = time.time()
        with open(CACHE_FILE, "wb") as f:
            pickle.dump({"bars": dict(all_data), "last_date": today}, f)
        log.info(f"  Cache saved: {len(all_data)} tickers ({time.time()-t0:.1f}s)")
    except Exception as e:
        log.warning(f"  Cache save failed: {e}")

    return all_data


def fetch_market_context(days=30):
    """Fetch VIX and SPY data needed for market-wide features.
    SPY needs 200+ days for SMA200; VIX needs 30+ for RSI(20) warmup."""
    import yfinance as yf

    # SPY: need 200 trading days (~14 months) for SMA200
    log.info("  Fetching SPY (2yr) and VIX (3mo) from yfinance...")
    spy_data = yf.download("SPY", period="2y", auto_adjust=True,
                           progress=False, threads=True)
    spy_close = spy_data["Close"].squeeze().dropna().values.astype(float).ravel()

    # VIX: need ~40 days for RSI(20) EWM warmup
    vix_data = yf.download("^VIX", period="3mo", auto_adjust=True,
                           progress=False, threads=True)
    vix_close = vix_data["Close"].squeeze().dropna().values.astype(float).ravel()

    ctx = {}

    # F25_VIX: raw VIX level
    ctx["vix_level"] = float(vix_close[-1]) if len(vix_close) > 0 else 0

    # F61_VIX_RSI20: RSI(20) of VIX using EWM
    ctx["vix_rsi20"] = _rsi_ewm(vix_close, 20) if len(vix_close) >= 21 else 50.0

    # F28_SPY5dRet: SPY 5-day return
    ctx["spy_5d_ret"] = float(spy_close[-1] / spy_close[-6] - 1) if len(spy_close) >= 6 else 0

    # F32_SPYdist200: SPY distance from 200-day SMA
    if len(spy_close) >= 200:
        sma200 = np.mean(spy_close[-200:])
        ctx["spy_dist200"] = float((spy_close[-1] - sma200) / sma200) if sma200 > 0 else 0
    else:
        ctx["spy_dist200"] = 0

    # SPY RSI(5) for F33_SectorRelRSI
    ctx["spy_rsi5"] = _rsi_ewm(spy_close, 5) if len(spy_close) >= 6 else 50.0

    # SPY 5d return (needed for F37_IdioReturn5d)
    ctx["spy_5d_ret_val"] = ctx["spy_5d_ret"]

    # Trading calendar: SPY's date index = NYSE trading days
    ctx["trading_dates"] = sorted(spy_data.index.normalize().date.tolist())

    log.info(f"  Market context: VIX={ctx['vix_level']:.1f} VIX_RSI20={ctx['vix_rsi20']:.1f} "
             f"SPY_5d={ctx['spy_5d_ret']*100:.2f}% SPY_dist200={ctx['spy_dist200']*100:.2f}%")
    return ctx


def load_mcap_lookup():
    """Load static market cap data for F39_LogMcap."""
    mcap_path = ROOT / "backtest_results_v4" / "mcap_cache.csv"
    try:
        mcap_df = pd.read_csv(mcap_path, index_col=0)
        mcap_dict = mcap_df["marketCap"].to_dict()
        log.info(f"  Loaded market caps for {len(mcap_dict)} tickers")
        return mcap_dict
    except Exception as e:
        log.warning(f"  Failed to load mcap_cache.csv: {e}")
        return {}


def compute_features_live(ohlcv_dict, feature_names, market_ctx=None, mcap_dict=None):
    """Compute 45 features for each stock from recent OHLCV.
    Matches v10_2_engine.py formulas exactly."""
    if market_ctx is None:
        market_ctx = {}
    if mcap_dict is None:
        mcap_dict = {}

    results = {}

    # Pre-compute breadth: fraction of stocks with close > SMA20
    above_sma20 = 0
    total_sma20 = 0
    for ticker, bars in ohlcv_dict.items():
        if len(bars) < 20:
            continue
        df = pd.DataFrame(bars).sort_values("date")
        c = df["close"].values
        if len(c) >= 20 and not np.isnan(c[-1]) and c[-1] > 0:
            sma20 = np.mean(c[-20:])
            if c[-1] > sma20:
                above_sma20 += 1
            total_sma20 += 1
    breadth = above_sma20 / total_sma20 if total_sma20 > 0 else 0.5
    log.info(f"  Breadth: {above_sma20}/{total_sma20} = {breadth:.3f}")

    # Market-wide values (same for all stocks)
    vix_level = market_ctx.get("vix_level", 0)
    vix_rsi20 = market_ctx.get("vix_rsi20", 50.0)
    spy_5d_ret = market_ctx.get("spy_5d_ret", 0)
    spy_dist200 = market_ctx.get("spy_dist200", 0)
    spy_rsi5 = market_ctx.get("spy_rsi5", 50.0)

    for ticker, bars in ohlcv_dict.items():
        if len(bars) < 10:
            continue

        df = pd.DataFrame(bars).sort_values("date")
        c = df["close"].values
        h = df["high"].values
        l = df["low"].values
        o = df["open"].values
        v = df["volume"].values

        if len(c) < 5 or np.isnan(c[-1]) or c[-1] <= 0:
            continue

        features = {}
        try:
            # --- Raw factors (matching v10_2_engine.py formulas) ---

            # F05_RSI14: Wilder's EWM RSI (NOT simple average)
            features["F05_RSI14"] = _rsi_ewm(c, 14)

            # F09_DistSMA20: (close - sma20) / sma20
            features["F09_DistSMA20"] = (c[-1] / np.mean(c[-20:]) - 1) if len(c) >= 20 else 0

            # F13_DD52wk: drawdown from 52-week high of HIGH (not close)
            hi252 = np.max(h[-252:]) if len(h) >= 252 else np.max(h)
            features["F13_DD52wk"] = (c[-1] - hi252) / hi252 if hi252 > 0 else 0

            # F14_IBS: (close - low) / (high - low)
            hl = h[-1] - l[-1]
            features["F14_IBS"] = (c[-1] - l[-1]) / hl if hl > 0 else 0.5

            # F17_RealVol: 20-day rolling std of ARITHMETIC returns, annualized
            # Backtest uses c[t]/c[t-1]-1 (not log returns), with ddof=1 (pandas default)
            if len(c) >= 21:
                arith_rets = np.diff(c[-21:]) / (c[-21:-1] + 1e-15)
                features["F17_RealVol"] = np.std(arith_rets, ddof=1) * np.sqrt(252)
            else:
                features["F17_RealVol"] = 0

            # F20_VolSpike: today's volume / 20-day avg volume
            vol20 = np.mean(v[-20:]) if len(v) >= 20 else 0
            features["F20_VolSpike"] = v[-1] / vol20 if vol20 > 0 else 1

            # F23_VolAtLows: ratio of down-day volume to total volume (5-day)
            # Engine: dv = sum(vol on down days, 5d) / sum(vol, 5d)
            if len(c) >= 6:
                rets = np.diff(c[-6:]) / (c[-6:-1] + 1e-15)
                down_mask = rets < 0
                dv = np.sum(v[-5:][down_mask]) if len(v) >= 5 else 0
                tv = np.sum(v[-5:]) if len(v) >= 5 else 1
                features["F23_VolAtLows"] = dv / tv if tv > 0 else 0
            else:
                features["F23_VolAtLows"] = 0

            # F25_VIX: raw VIX close level
            features["F25_VIX"] = vix_level

            # F28_SPY5dRet: SPY 5-day return
            features["F28_SPY5dRet"] = spy_5d_ret

            # F30_Breadth: fraction of stocks above SMA20
            features["F30_Breadth"] = breadth

            # F32_SPYdist200: SPY distance from 200 SMA
            features["F32_SPYdist200"] = spy_dist200

            # F33_SectorRelRSI: stock RSI(5) - SPY RSI(5)
            stock_rsi5 = _rsi_ewm(c, 5) if len(c) >= 6 else 50.0
            features["F33_SectorRelRSI"] = stock_rsi5 - spy_rsi5

            # F37_IdioReturn5d: stock 5d return MINUS SPY 5d return
            raw_5d = (c[-1] / c[-6] - 1) if len(c) >= 6 else 0
            features["F37_IdioReturn5d"] = raw_5d - spy_5d_ret

            # F39_LogMcap: log10(market_cap) from static lookup
            mcap = mcap_dict.get(ticker, np.nan)
            features["F39_LogMcap"] = np.log10(max(mcap, 1)) if not np.isnan(mcap) else 0

            # F43_5dRoC: 5-day rate of change
            features["F43_5dRoC"] = (c[-1] / c[-6] - 1) if len(c) >= 6 else 0

            # F45_3dRoC: 3-day rate of change
            features["F45_3dRoC"] = (c[-1] / c[-4] - 1) if len(c) >= 4 else 0

            # F52_OvernightGapZ: overnight gap Z-SCORED over 20-day window
            if len(c) >= 22 and len(o) >= 22:
                gaps = o[-20:] / c[-21:-1] - 1
                gap_today = o[-1] / c[-2] - 1
                g_mean = np.nanmean(gaps)
                g_std = np.nanstd(gaps, ddof=1)  # ddof=1 matches pandas .rolling().std()
                features["F52_OvernightGapZ"] = (gap_today - g_mean) / (g_std + 1e-9)
            elif len(c) >= 2:
                features["F52_OvernightGapZ"] = o[-1] / c[-2] - 1  # fallback raw
            else:
                features["F52_OvernightGapZ"] = 0

            # F57_PriorDayRet: close-to-close return
            features["F57_PriorDayRet"] = (c[-1] / c[-2] - 1) if len(c) >= 2 else 0

            # F54_IBSxRange: (1 - IBS) * range_expansion where range_exp = (H-L) / ATR14
            if len(c) >= 15:
                prev_c = np.roll(c, 1); prev_c[0] = c[0]
                tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
                atr14_arr = pd.Series(tr).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
                atr14_val = atr14_arr[-1]
                range_exp = (h[-1] - l[-1]) / atr14_val if atr14_val > 0 else 1.0
                features["F54_IBSxRange"] = (1 - features["F14_IBS"]) * range_exp
            else:
                features["F54_IBSxRange"] = 0

            # F60_IntradayMom: (close - open) / open (percentage, NOT range-normalized)
            features["F60_IntradayMom"] = (c[-1] - o[-1]) / o[-1] if o[-1] > 0 else 0

            # F61_VIX_RSI20: RSI(20) of VIX
            features["F61_VIX_RSI20"] = vix_rsi20

            # F16_ATRpctile: placeholder, computed cross-sectionally after loop
            features["F16_ATRpctile"] = 0
            # Store raw ATR14 for cross-sectional ranking
            if len(c) >= 15:
                features["_atr14"] = atr14_val
            else:
                features["_atr14"] = np.nan

            # --- V4 Interactions (10 pairs, simple products) ---
            features["INT_F14_IBS_F17_RealVol"] = features["F14_IBS"] * features["F17_RealVol"]
            features["INT_F14_IBS_F54_IBSxRange"] = features["F14_IBS"] * features["F54_IBSxRange"]
            features["INT_F57_PriorDayRet_F60_IntradayMom"] = features["F57_PriorDayRet"] * features["F60_IntradayMom"]
            features["INT_F14_IBS_F60_IntradayMom"] = features["F14_IBS"] * features["F60_IntradayMom"]
            features["INT_F17_RealVol_F57_PriorDayRet"] = features["F17_RealVol"] * features["F57_PriorDayRet"]
            features["INT_F14_IBS_F45_3dRoC"] = features["F14_IBS"] * features["F45_3dRoC"]
            features["INT_F20_VolSpike_F14_IBS"] = features["F20_VolSpike"] * features["F14_IBS"]
            features["INT_F23_VolAtLows_F17_RealVol"] = features["F23_VolAtLows"] * features["F17_RealVol"]
            features["INT_F37_IdioReturn5d_F33_SectorRelRSI"] = features["F37_IdioReturn5d"] * features["F33_SectorRelRSI"]
            features["INT_F09_DistSMA20_F14_IBS"] = features["F09_DistSMA20"] * features["F14_IBS"]

            # --- V4 Polynomials (10 squared features) ---
            for sq_f in ["F14_IBS", "F57_PriorDayRet", "F60_IntradayMom", "F54_IBSxRange",
                         "F17_RealVol", "F45_3dRoC", "F37_IdioReturn5d", "F09_DistSMA20",
                         "F20_VolSpike", "F23_VolAtLows"]:
                features[f"SQ_{sq_f}"] = features[sq_f] ** 2

            # --- V4 Ratios (3 features) ---
            rv = features["F17_RealVol"]
            features["RATIO_F14_IBS_over_F17_RealVol"] = features["F14_IBS"] / (rv + 1e-9)
            features["RATIO_F57_PriorDayRet_over_F17_RealVol"] = features["F57_PriorDayRet"] / (rv + 1e-9)
            features["RATIO_F45_3dRoC_over_F17_RealVol"] = features["F45_3dRoC"] / (rv + 1e-9)

            # --- Cross-sectional RANK and ZSCORE placeholders (computed after loop) ---
            # All 22 RANK_ and 10 ZSCORE_ features are set to 0 here,
            # then overwritten in the post-loop cross-sectional pass

            features["_close"] = c[-1]
            features["_ibs"] = features["F14_IBS"]
            features["_volume"] = v[-1]
            features["_avg_dollar_vol"] = np.mean(c[-20:] * v[-20:]) if len(c) >= 20 else 0

            results[ticker] = features

        except Exception as e:
            log.warning(f"Feature computation failed for {ticker}: {e}")
            continue

    # ---- V4 Cross-sectional features (computed AFTER per-stock loop) ----
    if results:
        from scipy.stats import rankdata

        # All 22 raw factor names for cross-sectional RANK
        raw_factors = [
            "F14_IBS", "F57_PriorDayRet", "F45_3dRoC", "F37_IdioReturn5d",
            "F33_SectorRelRSI", "F09_DistSMA20", "F20_VolSpike", "F23_VolAtLows",
            "F17_RealVol", "F39_LogMcap", "F16_ATRpctile", "F52_OvernightGapZ",
            "F25_VIX", "F28_SPY5dRet", "F30_Breadth", "F32_SPYdist200",
            "F43_5dRoC", "F05_RSI14", "F13_DD52wk", "F54_IBSxRange",
            "F60_IntradayMom", "F61_VIX_RSI20",
        ]

        # F16_ATRpctile: rank of raw ATR14 (special case, uses _atr14)
        atr_vals = {tk: results[tk]["_atr14"] for tk in results
                    if not np.isnan(results[tk].get("_atr14", np.nan))}
        if len(atr_vals) > 1:
            tks = list(atr_vals.keys())
            raw = np.array([atr_vals[tk] for tk in tks])
            ranks = rankdata(raw) / len(raw)
            for i, tk in enumerate(tks):
                results[tk]["F16_ATRpctile"] = ranks[i]

        # 22 cross-sectional RANK features
        for fn in raw_factors:
            rank_name = f"RANK_{fn}"
            vals = {}
            for tk in results:
                v_val = results[tk].get(fn, np.nan)
                if not np.isnan(v_val):
                    vals[tk] = v_val
            if len(vals) > 1:
                tks = list(vals.keys())
                raw = np.array([vals[tk] for tk in tks])
                ranks = rankdata(raw) / len(raw)  # matches backtest
                for i, tk in enumerate(tks):
                    results[tk][rank_name] = ranks[i]
            else:
                for tk in results:
                    results[tk][rank_name] = 0.5

        # 10 cross-sectional ZSCORE features (ddof=0 to match training!)
        zscore_factors = [
            "F14_IBS", "F57_PriorDayRet", "F60_IntradayMom", "F54_IBSxRange",
            "F17_RealVol", "F45_3dRoC", "F37_IdioReturn5d", "F20_VolSpike",
            "F23_VolAtLows", "F33_SectorRelRSI",
        ]
        for fn in zscore_factors:
            zs_name = f"ZSCORE_{fn}"
            vals = {}
            for tk in results:
                v_val = results[tk].get(fn, np.nan)
                if not np.isnan(v_val):
                    vals[tk] = v_val
            if len(vals) > 2:
                tks = list(vals.keys())
                raw = np.array([vals[tk] for tk in tks])
                mu = np.mean(raw)
                std = np.std(raw)  # ddof=0! matches backtest training data
                if std > 1e-9:
                    for i, tk in enumerate(tks):
                        results[tk][zs_name] = (raw[i] - mu) / std
                else:
                    for tk in tks:
                        results[tk][zs_name] = 0.0
            else:
                for tk in results:
                    results[tk][zs_name] = 0.0

    return results


def _rsi_ewm(prices, period=14):
    """Compute RSI using Wilder's EWM smoothing (matches v10_2_engine.rsi_arr)."""
    if len(prices) < period + 1:
        return 50.0
    s = pd.Series(prices)
    delta = s.diff()
    up = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    dn = (-delta).clip(lower=0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / (dn + 1e-9))
    return float(rsi.values[-1])


# =====================================================================
# ML SCORING
# =====================================================================
def load_model(model_dir, year=2026):
    """Load LightGBM model for the given year, with fallback."""
    for yr in [year, year - 1, year - 2]:
        path = Path(model_dir) / f"lgb_model_{yr}.txt"
        if path.exists():
            model = lgb.Booster(model_file=str(path))
            log.info(f"Loaded model: {path} ({model.num_feature()} features)")
            return model
    raise FileNotFoundError(f"No model found in {model_dir}")


def score_stocks(model, feature_dict, feature_names, return_matrix=False):
    """Score all stocks, return sorted list of (ticker, score, features).
    If return_matrix=True, also returns (X, tickers_ordered) for SHAP."""
    tickers = list(feature_dict.keys())
    if not tickers:
        return ([], None, []) if return_matrix else []

    # Build feature matrix
    X = np.zeros((len(tickers), len(feature_names)))
    for i, tk in enumerate(tickers):
        feats = feature_dict[tk]
        for j, fn in enumerate(feature_names):
            X[i, j] = feats.get(fn, 0)

    X = np.nan_to_num(X, nan=0.0)
    preds = model.predict(X)

    # Min-max normalize
    pmin, pmax = preds.min(), preds.max()
    if pmax > pmin:
        scores = (preds - pmin) / (pmax - pmin)
    else:
        scores = np.full(len(preds), 0.5)

    results = [(tickers[i], scores[i], feature_dict[tickers[i]]) for i in range(len(tickers))]
    # Tiebreak: same ML score -> lower IBS wins (more oversold = better MR candidate)
    results.sort(key=lambda x: (-x[1], x[2].get("_ibs", 0.5)))

    if return_matrix:
        return results, X, tickers
    return results


# =====================================================================
# TRADE ENRICHMENT -- capture context for backtest-vs-live comparison
# =====================================================================
def _build_enrichment(ticker, feat_dict, market_ctx, score, feature_names):
    """Capture top features, market context, and metadata for a trade record.
    This data lets us compare backtest assumptions vs live execution."""
    enrichment = {}

    # Top 10 features by absolute value (for debugging score changes)
    if ticker in feat_dict:
        feats = feat_dict[ticker]
        feat_vals = {fn: feats.get(fn, 0) for fn in feature_names}
        top10 = sorted(feat_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        enrichment["top_features"] = {k: round(v, 6) for k, v in top10}

    # Market context snapshot
    if market_ctx:
        enrichment["market_ctx"] = {
            "vix": round(market_ctx.get("vix_level", 0), 2),
            "vix_rsi20": round(market_ctx.get("vix_rsi20", 0), 1),
            "spy_5d_ret": round(market_ctx.get("spy_5d_ret", 0), 5),
            "spy_dist200": round(market_ctx.get("spy_dist200", 0), 5),
        }

    # Signal-time close price (what we used to estimate entry)
    if ticker in feat_dict:
        enrichment["signal_price"] = round(feat_dict[ticker].get("_close", 0), 4)

    # ML score at this moment
    enrichment["ml_score"] = round(score, 6)

    return enrichment


# =====================================================================
# MAIN SIGNAL LOGIC
# =====================================================================
def run_signal(dry_run=False):
    log.info("=" * 60)
    log.info(f"ML v4 Signal Generator -- {datetime.now():%Y-%m-%d %H:%M}")
    log.info(f"Mode: {'DRY-RUN' if dry_run else 'LIVE'}")
    log.info("=" * 60)

    # Load config
    max_price = CFG["filters"]["max_price"]
    ibs_max = CFG["filters"]["ibs_max"]
    score_threshold = CFG["filters"]["score_threshold"]
    min_dollar_vol = CFG["filters"]["min_dollar_volume"]
    stop_pct = CFG["exits"]["stop_pct"]
    max_hold = CFG["exits"]["max_hold_days"]
    score_exit_floor = CFG["exits"]["score_exit_floor"]
    trail_pct = CFG["exits"]["trailing_pct"]
    trail_act = CFG["exits"]["trailing_activation"]

    # Connect to Alpaca
    api = get_alpaca_api()
    acct = get_account_info(api)
    positions = get_positions(api)
    log.info(f"Account: equity=${acct['equity']:.2f} cash=${acct['cash']:.2f} buying_power=${acct['buying_power']:.2f}")
    log.info(f"Open positions: {list(positions.keys()) if positions else 'none'}")

    # Load state
    state = load_state()

    # ----- GTC STOP RECONCILIATION -----
    # If a GTC stop order filled between runs (e.g. gap-down at open), detect it
    # and record the trade before proceeding. This prevents stale position state.
    held = state.get("position")
    if held and _stop_mode == "gtc" and held.get("stop_order_id"):
        try:
            stop_order = api.get_order(held["stop_order_id"])
            if stop_order.status == "filled":
                fill_price = float(stop_order.filled_avg_price)
                fill_qty = float(stop_order.filled_qty)
                ep = held["entry_price"]
                log.info(f"GTC STOP FILLED between runs: {held['ticker']} "
                         f"{fill_qty:.0f} shares @ ${fill_price:.2f} "
                         f"(PnL: {(fill_price/ep - 1)*100:+.2f}%)")
                # Record the trade with actual GTC fill price
                trade_record = {
                    "ticker": held["ticker"],
                    "entry_date": held["entry_date"],
                    "exit_date": str(stop_order.filled_at),
                    "entry_price": ep,
                    "exit_price": round(fill_price, 2),
                    "days_held": 0,  # will be approximate
                    "exit_reason": f"gtc_stop_filled(${fill_price:.2f})",
                    "pnl_pct": round((fill_price / ep - 1) * 100, 2),
                    "notional": held.get("notional", 0),
                    "buy_order_id": held.get("buy_order_id", ""),
                    "sell_order_id": held["stop_order_id"],
                    "entry_enrichment": held.get("entry_enrichment", {}),
                    "exit_enrichment": {
                        "ml_score": 0,
                        "signal_price": round(fill_price, 4),
                        "market_ctx": {},
                        "top_features": {},
                        "note": "GTC stop filled between runs -- no live scoring",
                    },
                }
                state.setdefault("trades", []).append(trade_record)
                # Check for fractional remainder
                tk = held["ticker"]
                if tk in positions:
                    remainder = positions[tk]["qty"]
                    log.info(f"  Fractional remainder: {remainder:.4f} shares -- "
                             f"will sell at close")
                    # Keep position in state but mark as remainder-only
                    held["gtc_stop_filled"] = True
                    held["remainder_qty"] = remainder
                else:
                    state["position"] = None
                save_state(state)
        except Exception as e:
            log.warning(f"GTC stop reconciliation failed (non-fatal): {e}")

    # Load universe
    universe_file = PROD / "universe_live.csv"
    if universe_file.exists():
        universe = pd.read_csv(universe_file)["ticker"].tolist()
        log.info(f"Universe filter: {len(universe)} tickers from universe_live.csv")
    else:
        # Fallback: use factor cache tickers
        import pickle
        with open(ROOT / "factor_tournament" / "v10_2_factors.pkl", "rb") as f:
            cache = pickle.load(f)
        universe = list(cache["tickers"])
        log.warning(f"No universe_live.csv found, using factor cache ({len(universe)} tickers)")

    # ----- FETCH DATA + FEATURES (needed for both exits and entries) -----
    # V4 uses 77 features: 22 raw + 10 INT + 10 SQ + 22 RANK + 10 ZSCORE + 3 RATIO
    # Load exact feature order from saved model metadata
    v4_meta_path = Path(CFG["model"]["model_dir"]) / ".." / "shared_data" / "metadata.json"
    if v4_meta_path.exists():
        with open(v4_meta_path) as f:
            feature_names = json.load(f)["feature_names"]
    else:
        # Fallback: construct from known v4 feature set
        from v10_2_engine import FACTOR_NAMES
        feature_names = list(FACTOR_NAMES)
        # Interactions
        feature_names += [
            "INT_F14_IBS_F17_RealVol", "INT_F14_IBS_F54_IBSxRange",
            "INT_F57_PriorDayRet_F60_IntradayMom", "INT_F14_IBS_F60_IntradayMom",
            "INT_F17_RealVol_F57_PriorDayRet", "INT_F14_IBS_F45_3dRoC",
            "INT_F20_VolSpike_F14_IBS", "INT_F23_VolAtLows_F17_RealVol",
            "INT_F37_IdioReturn5d_F33_SectorRelRSI", "INT_F09_DistSMA20_F14_IBS",
        ]
        # Polynomials
        for f in ["F14_IBS", "F57_PriorDayRet", "F60_IntradayMom", "F54_IBSxRange",
                   "F17_RealVol", "F45_3dRoC", "F37_IdioReturn5d", "F09_DistSMA20",
                   "F20_VolSpike", "F23_VolAtLows"]:
            feature_names.append(f"SQ_{f}")
        # Ranks
        for f in FACTOR_NAMES:
            feature_names.append(f"RANK_{f}")
        # Z-scores
        for f in ["F14_IBS", "F57_PriorDayRet", "F60_IntradayMom", "F54_IBSxRange",
                   "F17_RealVol", "F45_3dRoC", "F37_IdioReturn5d", "F20_VolSpike",
                   "F23_VolAtLows", "F33_SectorRelRSI"]:
            feature_names.append(f"ZSCORE_{f}")
        # Ratios
        feature_names += ["RATIO_F14_IBS_over_F17_RealVol",
                          "RATIO_F57_PriorDayRet_over_F17_RealVol",
                          "RATIO_F45_3dRoC_over_F17_RealVol"]
    log.info(f"V4 feature set: {len(feature_names)} features")

    # ----- DATA FETCH + FEATURE COMPUTATION -----
    # In "read" mode, load shared scoring state from the GTC run to ensure
    # identical features/scores. Falls back to independent computation on failure.
    _shared_loaded = False
    if _scoring_mode == "read":
        shared = _load_shared_scoring()
        if shared:
            feat_dict = shared["feat_dict"]
            market_ctx = shared["market_ctx"]
            _shared_loaded = True

    if not _shared_loaded:
        log.info(f"Fetching OHLCV for {len(universe)} tickers...")
        # days=380 (~252 trading days) for proper DD52wk, ATR14, RealVol warmup
        ohlcv = fetch_ohlcv(api, universe, days=380)
        log.info(f"OHLCV received for {len(ohlcv)} tickers")

        log.info("Fetching market context (VIX, SPY)...")
        market_ctx = fetch_market_context()
        mcap_dict = load_mcap_lookup()

        feat_dict = compute_features_live(ohlcv, feature_names, market_ctx, mcap_dict)
        log.info(f"Features computed for {len(feat_dict)} stocks")

        # In "write" mode, save scoring state for the MOC account
        if _scoring_mode == "write":
            _save_shared_scoring(feat_dict, market_ctx)

    model = load_model(CFG["model"]["model_dir"], CFG["model"]["current_model_year"])

    # ----- POSITION MANAGEMENT (check exits before new entries) -----
    held = state.get("position")

    # Handle fractional remainder from GTC stop that already filled
    if held and held.get("gtc_stop_filled"):
        tk = held["ticker"]
        if tk in positions:
            remainder = positions[tk]["qty"]
            log.info(f"Selling fractional remainder: {tk} {remainder:.4f} shares")
            oid = submit_moc_sell(api, tk, remainder, "gtc_remainder", dry_run=dry_run)
            if oid:
                log.info(f"  Remainder sell submitted: {oid}")
        else:
            log.info(f"Fractional remainder {tk} already gone from Alpaca")
        state["position"] = None
        save_state(state)
        held = None  # proceed to entry logic

    if held:
        tk = held["ticker"]
        entry_date = datetime.fromisoformat(held["entry_date"]).date()
        # Count actual trading days using SPY calendar (handles holidays)
        trading_dates = market_ctx.get("trading_dates", [])
        if trading_dates:
            days_held = sum(1 for d in trading_dates if entry_date < d <= datetime.now().date())
        else:
            days_held = int(np.busday_count(entry_date, datetime.now().date()))
        ep = held["entry_price"]
        highest_close = held.get("highest_close", ep)
        log.info(f"Held position: {tk} day {days_held} of {max_hold}, "
                 f"entry ${ep:.2f}, highest_close ${highest_close:.2f}")

        # Place pending GTC stop if it exists (deferred from buy day due to settlement)
        pending_stop = held.get("pending_stop")
        if pending_stop and _stop_mode == "gtc" and not held.get("stop_order_id"):
            stop_oid = submit_stop_order(api, tk, pending_stop["qty"],
                                          pending_stop["price"], dry_run=dry_run)
            if stop_oid:
                held["stop_order_id"] = stop_oid
                held.pop("pending_stop", None)
                log.info(f"GTC stop NOW PLACED at ${pending_stop['price']:.2f}")
            else:
                log.warning(f"GTC stop still failed -- will retry next run")

        # Check if position exists in Alpaca
        if tk not in positions:
            asset_exists = True
            try:
                api.get_asset(tk)
            except Exception:
                asset_exists = False
            unfamiliar = [s for s in positions if s != tk]
            if not asset_exists and unfamiliar:
                log.error(f"Position {tk} not in Alpaca AND asset 404. Likely ticker "
                          f"rename (corp action). Unfamiliar position(s): {unfamiliar}. "
                          f"PATCH STATE MANUALLY -- skipping run to avoid double-buy.")
                save_state(state)
                return
            if asset_exists:
                log.error(f"Position {tk} in state but NOT in Alpaca (asset still tradeable). "
                          f"Possible failed buy or external sale -- skipping run, NOT clearing.")
                save_state(state)
                return
            log.warning(f"Position {tk} in state, NOT in Alpaca, asset 404, no other "
                        f"positions -- assuming corp-action cash settlement, clearing.")
            state["position"] = None
        else:
            qty = positions[tk]["qty"]
            current_price = positions[tk].get("market_value", 0) / max(qty, 1e-9)

            # Also try to get current price from our snapshot data
            if tk in feat_dict:
                current_price = feat_dict[tk]["_close"]

            # Update highest close for trailing stop
            new_hc = max(highest_close, current_price)
            held["highest_close"] = new_hc

            exit_reason = None

            # Pre-compute current ML score for exit logic + enrichment
            held_score = None
            if tk in feat_dict:
                scored_all = score_stocks(model, feat_dict, feature_names)
                for t, sc, _ in scored_all:
                    if t == tk:
                        held_score = sc
                        break
                if held_score is not None:
                    log.info(f"  Re-scored {tk}: {held_score:.4f} "
                             f"(floor={score_exit_floor})")

            # EXIT PRIORITY (matches v10_2_engine.py order):
            # 1. Stop-loss: close <= entry * (1 - stop_pct)
            stop_level = ep * (1 - stop_pct)
            if current_price <= stop_level:
                exit_reason = (f"stop(${current_price:.2f} <= "
                               f"${stop_level:.2f} = entry ${ep:.2f} x "
                               f"{1 - stop_pct:.3f})")

            # 2. Trailing stop: highest_close activated AND close fell below trail
            elif (trail_pct > 0 and trail_act > 0
                  and new_hc >= ep * (1 + trail_act)
                  and current_price <= new_hc * (1 - trail_pct)):
                trail_level = new_hc * (1 - trail_pct)
                exit_reason = (f"trailing(${current_price:.2f} <= "
                               f"${trail_level:.2f} = hc ${new_hc:.2f} x "
                               f"{1 - trail_pct:.3f})")

            # 3. Score-exit: exit if re-score below floor (after day 0)
            elif days_held >= 1 and score_exit_floor > 0 and held_score is not None:
                if held_score < score_exit_floor:
                    exit_reason = (f"score_exit({held_score:.4f} < "
                                   f"{score_exit_floor})")

            # 4. Hold expiry (fallback)
            if exit_reason is None and days_held >= max_hold:
                exit_reason = f"hold_expiry(day{days_held}>={max_hold})"

            # Execute exit
            if exit_reason:
                log.info(f"EXIT SIGNAL: {tk} -> {exit_reason}")
                # Cancel GTC stop order before selling (if exists)
                if _stop_mode == "gtc" and held.get("stop_order_id"):
                    cancel_order(api, held["stop_order_id"], dry_run=dry_run)
                oid = submit_moc_sell(api, tk, qty, exit_reason, dry_run=dry_run)
                if oid:
                    exit_enrichment = _build_enrichment(
                        tk, feat_dict, market_ctx,
                        held_score if held_score is not None else 0,
                        feature_names)
                    trade_record = {
                        "ticker": tk,
                        "entry_date": held["entry_date"],
                        "exit_date": datetime.now().isoformat(),
                        "entry_price": ep,
                        "exit_price": round(current_price, 2),
                        "days_held": days_held,
                        "exit_reason": exit_reason,
                        "pnl_pct": round((current_price / ep - 1) * 100, 2),
                        "notional": held.get("notional", 0),
                        "buy_order_id": held.get("buy_order_id", ""),
                        "sell_order_id": oid,
                        "entry_enrichment": held.get("entry_enrichment", {}),
                        "exit_enrichment": exit_enrichment,
                    }
                    state.setdefault("trades", []).append(trade_record)
                    state["position"] = None
                    log.info(f"EXIT {tk}: {exit_reason} | "
                             f"PnL: {trade_record['pnl_pct']:+.2f}%")
            else:
                log.info(f"  Holding {tk}: price=${current_price:.2f}, "
                         f"stop=${stop_level:.2f}, hc=${new_hc:.2f}, "
                         f"day {days_held}/{max_hold}")

    # ----- ENTRY LOGIC -----
    if state.get("position") is None and acct["buying_power"] > 10:
        log.info("Looking for entry signal...")

        # Apply pre-filters
        filtered = {}
        filter_stats = {"total": len(feat_dict), "price": 0, "ibs": 0, "volume": 0, "passed": 0}
        for tk, feats in feat_dict.items():
            price = feats["_close"]
            avg_dv = feats["_avg_dollar_vol"]

            if price > max_price:
                filter_stats["price"] += 1
                continue
            if avg_dv < min_dollar_vol:
                filter_stats["volume"] += 1
                continue
            filtered[tk] = feats
            filter_stats["passed"] += 1

        log.info(f"Universe filter: {filter_stats['total']} -> {filter_stats['passed']} "
                 f"(price: -{filter_stats['price']}, volume: -{filter_stats['volume']})")

        # ML scoring: score FULL universe for proper min-max normalization
        # (matches backtest), then filter to tradeable candidates
        scored_full, X_scored, scored_tickers = score_stocks(
            model, feat_dict, feature_names, return_matrix=True)
        # Keep only filtered (tradeable) stocks for entry candidates
        filtered_set = set(filtered.keys())
        scored = [s for s in scored_full if s[0] in filtered_set]

        # --- SHAP Alpha Decay Monitor ---
        # Runs after scoring, adds ~10-20s to pipeline
        try:
            equity_curve = state.get("equity_curve", [])

            if X_scored is not None and len(X_scored) > 0:
                shap_alerts, sharpe_alerts = run_shap_monitor(
                    model, X_scored, feature_names, equity_curve=equity_curve)
                for severity, message in shap_alerts + sharpe_alerts:
                    log.warning(f"SHAP ALERT [{severity.upper()}]: {message}")
            else:
                log.info("SHAP monitor skipped: no scored stocks")
        except Exception as e:
            log.error(f"SHAP monitor failed (non-fatal): {e}")

        # Apply entry filters
        candidates = []
        for tk, score, feats in scored:
            if score < score_threshold:
                break  # sorted descending, no more will pass
            if feats["_ibs"] > ibs_max:
                continue
            if tk in positions:
                continue
            candidates.append((tk, score, feats))

        log.info(f"Candidates after filters: {len(candidates)}")
        if candidates:
            for i, (tk, sc, _) in enumerate(candidates[:5]):
                log.info(f"  #{i+1}: {tk} score={sc:.4f}")
        else:
            log.info("No qualifying signals today")
            state["today_signal"] = {"status": "No signal"}
            # Log top 5 scored stocks with reasons for rejection
            for i, (tk, sc, feats) in enumerate(scored[:5]):
                reasons = []
                if sc < score_threshold:
                    reasons.append(f"score {sc:.3f} < {score_threshold}")
                if feats["_ibs"] > ibs_max:
                    reasons.append(f"IBS {feats['_ibs']:.3f} > {ibs_max}")
                log.info(f"  Top {i+1}: {tk} score={sc:.4f} -- rejected: {', '.join(reasons) if reasons else 'held'}")

        # Submit order for #1 pick
        if candidates:
            ticker, score, feats = candidates[0]
            equity = acct["equity"]
            notional = equity * 0.98  # 1x leverage, small buffer
            log.info(f"SIGNAL: {ticker} score={score:.4f} IBS={feats['_ibs']:.4f} price=${feats['_close']:.2f}")
            log.info(f"Cash available: ${acct['buying_power']:.2f} -> notional=${notional:.2f}")

            # Save today's signal for dashboard
            state["today_signal"] = {
                "status": "Submitted",
                "ticker": ticker,
                "score": round(score, 4),
                "ibs": round(feats["_ibs"], 4),
                "price": round(feats["_close"], 2),
                "notional": round(notional, 2),
                "shares_est": int(notional / feats["_close"]),
            }

            oid = submit_moc_buy(api, ticker, notional, dry_run=dry_run)
            if oid:
                entry_price_est = feats["_close"]
                stop_level = entry_price_est * (1 - stop_pct)
                est_qty = notional / entry_price_est

                entry_enrichment = _build_enrichment(
                    ticker, feat_dict, market_ctx, score, feature_names)
                state["position"] = {
                    "ticker": ticker,
                    "entry_date": datetime.now().isoformat(),
                    "entry_price": entry_price_est,  # estimate, updated after fill
                    "highest_close": entry_price_est,
                    "ml_score": round(score, 4),
                    "ibs": round(feats["_ibs"], 4),
                    "buy_order_id": oid,
                    "notional": round(notional, 2),
                    "entry_enrichment": entry_enrichment,
                }
                log.info(f"Entry recorded: {ticker} @ ~${entry_price_est:.2f}")

                # GTC vs MOC stop handling
                if _stop_mode == "gtc":
                    # Alpaca rejects sell stops while buy is unsettled.
                    # Store stop params in state -- place the GTC stop on NEXT run
                    # when the position is settled.
                    state["position"]["pending_stop"] = {
                        "price": round(stop_level, 2),
                        "qty": est_qty,
                    }
                    log.info(f"GTC stop PENDING at ${stop_level:.2f} "
                             f"({stop_pct*100:.1f}% below entry) -- "
                             f"will place on next run after settlement")
                else:
                    log.info(f"Software stop at ${stop_level:.2f} "
                             f"({stop_pct*100:.1f}% below entry) -- "
                             f"checked at each daily run (MOC exit)")

    # Record equity for rolling Sharpe calculation
    state.setdefault("equity_curve", [])
    today_eq = round(acct["equity"], 2)
    prev_eq = state["equity_curve"][-1].get("actual") if state["equity_curve"] else None
    # Sanity check: corp-action transients (e.g. ASGN->EFOR) can briefly drop
    # equity to a wrong value. Skip writing if equity halved with no closing trade.
    if prev_eq and today_eq < prev_eq * 0.5 and state.get("position") is not None:
        log.error(f"Equity sanity: ${prev_eq:.2f} -> ${today_eq:.2f} with position held. "
                  f"Suspect Alpaca transient -- NOT recording to equity_curve.")
    else:
        state["equity_curve"].append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "actual": today_eq,
        })
    # Keep last 120 days max (2x the 60-day Sharpe window)
    if len(state["equity_curve"]) > 120:
        state["equity_curve"] = state["equity_curve"][-120:]

    save_state(state)
    log.info("Signal generator complete.")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    run_signal(dry_run=dry_run)
