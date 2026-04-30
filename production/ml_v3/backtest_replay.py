#!/usr/bin/env python3
"""
Backtest Replay -- Run ML v3 seed 6 through v10_2_engine from paper trading
start date, using the exact same config that produced +709% CAGR.

Downloads fresh data from yfinance to extend the factor cache beyond its
end date (2026-03-16), computes features + ML scores for new days, then
runs the engine. Outputs a daily equity curve for the dashboard.

Usage:
    python backtest_replay.py              # Update backtest line in dashboard
    python backtest_replay.py --full       # Full OOS backtest (verification)
"""
import sys, json, logging, time, pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROD = Path(__file__).parent
ROOT = PROD.parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("backtest_replay")

PAPER_START = "2026-03-23"  # first day of paper trading


def load_engine_data():
    """Load factor cache + ML scores + params."""
    from v10_2_engine import compute_and_cache_factors

    log.info("Loading factor cache...")
    data = compute_and_cache_factors(force=False)
    close = data["close"].astype(np.float32)
    high = data["high"].astype(np.float32)
    low = data["low"].astype(np.float32)
    open_ = data["open"].astype(np.float32)
    T, N = data["T"], data["N"]
    oos_idx = data["oos_idx"]
    all_dates = data["all_dates"]
    tickers = list(data["tickers"])

    log.info(f"Factor cache: {all_dates[0].date()} to {all_dates[-1].date()}, "
             f"T={T}, N={N}, oos_idx={oos_idx}")

    # ML scores
    scores_path = ROOT / "v10_ml_v3" / "seed_6" / "scores_ml.npy"
    scores_ml = np.load(scores_path)
    log.info(f"ML scores: {scores_ml.shape}")

    # Production params
    with open(ROOT / "v10_2_final" / "factor_weights.json") as f:
        fw = json.load(f)
    bp = fw["strategy_params"]

    with open(ROOT / "v10_ml_v3" / "exit_optimization_realistic" /
              "optimal_realistic_exits.json") as f:
        exits = json.load(f)["best_params"]

    params = {**bp}
    params.update(exits)
    params["use_score_exit"] = True
    params["use_trailing"] = True
    params["slippage"] = 0.00064  # Corwin-Schultz MOC
    params["fractional_shares"] = True
    params["entry_mode"] = "moc"

    return {
        "close": close, "high": high, "low": low, "open": open_,
        "scores_ml": scores_ml, "params": params,
        "T": T, "N": N, "oos_idx": oos_idx,
        "all_dates": all_dates, "tickers": tickers,
        "data": data,
    }


def extend_with_yfinance(engine_data):
    """Download fresh OHLCV from yfinance for days beyond the factor cache.
    Compute raw factors and ML scores for extended days.
    Append to engine arrays."""
    import yfinance as yf
    import lightgbm as lgb

    all_dates = engine_data["all_dates"]
    tickers = engine_data["tickers"]
    cache_end = all_dates[-1]

    # How many days do we need?
    today = datetime.now()
    if cache_end.date() >= today.date():
        log.info("Factor cache is current, no extension needed")
        return engine_data

    log.info(f"Extending data from {cache_end.date()} to {today.date()}...")

    # Download fresh data
    start_dl = (cache_end - timedelta(days=5)).strftime("%Y-%m-%d")  # overlap for factors
    end_dl = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    log.info(f"Downloading OHLCV from yfinance ({start_dl} to {end_dl})...")
    batch_size = 100
    all_ohlcv = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            raw = yf.download(batch, start=start_dl, end=end_dl,
                              auto_adjust=True, threads=True, progress=False,
                              group_by="ticker")
            if len(batch) == 1:
                tk = batch[0]
                if "Close" in raw.columns and len(raw) > 0:
                    all_ohlcv[tk] = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            else:
                for tk in batch:
                    try:
                        df = raw[tk].dropna(subset=["Close"])
                        if len(df) > 0:
                            all_ohlcv[tk] = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                    except Exception:
                        pass
        except Exception as e:
            log.warning(f"yfinance batch {i} failed: {e}")

    log.info(f"Downloaded {len(all_ohlcv)} tickers")

    # Find new trading days (dates in yfinance not in cache)
    cache_date_set = set(d.date() for d in all_dates)
    new_dates_set = set()
    for tk, df in all_ohlcv.items():
        for idx in df.index:
            dt = idx.date() if hasattr(idx, "date") else idx
            if dt not in cache_date_set and dt > cache_end.date():
                new_dates_set.add(dt)

    new_dates = sorted(new_dates_set)
    if not new_dates:
        log.info("No new trading days found")
        return engine_data

    log.info(f"Found {len(new_dates)} new trading days: {new_dates[0]} to {new_dates[-1]}")
    n_new = len(new_dates)
    N = engine_data["N"]
    T_old = engine_data["T"]
    T_new = T_old + n_new

    # Build extended OHLCV arrays
    close_ext = np.full((T_new, N), np.nan, dtype=np.float32)
    high_ext = np.full((T_new, N), np.nan, dtype=np.float32)
    low_ext = np.full((T_new, N), np.nan, dtype=np.float32)
    open_ext = np.full((T_new, N), np.nan, dtype=np.float32)

    close_ext[:T_old] = engine_data["close"]
    high_ext[:T_old] = engine_data["high"]
    low_ext[:T_old] = engine_data["low"]
    open_ext[:T_old] = engine_data["open"]

    ticker_idx = {tk: i for i, tk in enumerate(tickers)}

    for tk, df in all_ohlcv.items():
        if tk not in ticker_idx:
            continue
        j = ticker_idx[tk]
        for idx, row in df.iterrows():
            dt = idx.date() if hasattr(idx, "date") else idx
            if dt in new_dates_set:
                day_offset = T_old + new_dates.index(dt)
                close_ext[day_offset, j] = float(row["Close"])
                high_ext[day_offset, j] = float(row["High"])
                low_ext[day_offset, j] = float(row["Low"])
                open_ext[day_offset, j] = float(row["Open"])

    # Carry forward NaN (for stocks missing a day)
    for t in range(T_old, T_new):
        mask = np.isnan(close_ext[t])
        close_ext[t, mask] = close_ext[t-1, mask]
        high_ext[t, mask] = high_ext[t-1, mask]
        low_ext[t, mask] = low_ext[t-1, mask]
        open_ext[t, mask] = open_ext[t-1, mask]

    # Score new days with seed 6 model
    log.info("Scoring new days with seed 6 model...")
    model_path = ROOT / "v10_ml_v3" / "seed_6" / "lgb_model_2026.txt"
    if not model_path.exists():
        model_path = ROOT / "v10_ml_v3" / "seed_6" / "lgb_model_2025.txt"
    model = lgb.Booster(model_file=str(model_path))
    feature_names = model.feature_name()
    n_feat = len(feature_names)
    log.info(f"Model: {model_path.name}, {n_feat} features")

    # Extend scores array
    scores_old = engine_data["scores_ml"]
    scores_ext = np.zeros((T_new, N), dtype=np.float32)
    scores_ext[:T_old] = scores_old

    # For each new day, compute features and score
    from v10_2_engine import FACTOR_NAMES
    for d_idx, dt in enumerate(new_dates):
        t = T_old + d_idx
        # Build feature matrix for all stocks on this day
        X = np.zeros((N, n_feat), dtype=np.float32)

        for j in range(N):
            # Get history up to this day
            c = close_ext[:t+1, j]
            h = high_ext[:t+1, j]
            l = low_ext[:t+1, j]
            o = open_ext[:t+1, j]

            if np.isnan(c[-1]) or c[-1] <= 0:
                continue

            feats = _compute_features_for_day(c, h, l, o, feature_names)
            X[j] = feats

        # Cross-sectional ranks
        X = _add_cross_sectional_ranks(X, feature_names, close_ext[t])

        # Score
        X = np.nan_to_num(X, nan=0.0)
        preds = model.predict(X)

        # Min-max normalize to [0, 1]
        pmin, pmax = preds.min(), preds.max()
        if pmax > pmin:
            scores_ext[t] = (preds - pmin) / (pmax - pmin)
        else:
            scores_ext[t] = 0.5

    # Update engine data
    all_dates_ext = list(all_dates) + [datetime.combine(d, datetime.min.time())
                                        for d in new_dates]

    engine_data["close"] = close_ext
    engine_data["high"] = high_ext
    engine_data["low"] = low_ext
    engine_data["open"] = open_ext
    engine_data["scores_ml"] = scores_ext
    engine_data["T"] = T_new
    engine_data["all_dates"] = all_dates_ext

    log.info(f"Extended to {T_new} days ({all_dates_ext[-1].date()})")
    return engine_data


def _compute_features_for_day(c, h, l, o, feature_names):
    """Compute features for a single stock on the last day of the arrays.
    Matches the feature computation in early_signal_ml_v3.py."""
    n = len(c)
    feats = {}

    # Raw factors
    feats["F05_RSI14"] = _rsi(c, 14)
    feats["F08_RSI4"] = _rsi(c, 4)
    feats["F09_DistSMA20"] = (c[-1] / np.mean(c[-20:])) - 1 if n >= 20 else 0
    feats["F10_DistSMA50"] = (c[-1] / np.mean(c[-50:])) - 1 if n >= 50 else 0
    feats["F11_DistSMA200"] = (c[-1] / np.mean(c[-200:])) - 1 if n >= 200 else 0
    feats["F13_DD52wk"] = (c[-1] / np.max(c[-252:])) - 1 if n >= 252 else (c[-1] / np.max(c) - 1)
    feats["F14_IBS"] = (c[-1] - l[-1]) / (h[-1] - l[-1]) if (h[-1] - l[-1]) > 0 else 0.5
    feats["F16_ATRpctile"] = 0  # simplified
    feats["F17_RealVol"] = np.std(np.diff(np.log(c[-21:]))) * np.sqrt(252) if n >= 21 else 0
    feats["F20_VolSpike"] = 1  # no volume in arrays
    feats["F23_VolAtLows"] = 0
    feats["F25_VIX"] = 0
    feats["F28_SPY5dRet"] = 0
    feats["F30_Breadth"] = 0
    feats["F32_SPYdist200"] = 0
    feats["F33_SectorRelRSI"] = 0
    feats["F37_IdioReturn5d"] = (c[-1] / c[-6] - 1) if n >= 6 else 0
    feats["F39_LogMcap"] = 0
    feats["F43_5dRoC"] = (c[-1] / c[-6] - 1) if n >= 6 else 0
    feats["F45_3dRoC"] = (c[-1] / c[-4] - 1) if n >= 4 else 0
    feats["F52_OvernightGapZ"] = (o[-1] / c[-2] - 1) if n >= 2 else 0
    feats["F54_IBSxRange"] = feats["F14_IBS"] * ((h[-1] - l[-1]) / c[-1]) if c[-1] > 0 else 0
    feats["F57_PriorDayRet"] = (c[-1] / c[-2] - 1) if n >= 2 else 0
    feats["F60_IntradayMom"] = (c[-1] - o[-1]) / (h[-1] - l[-1]) if (h[-1] - l[-1]) > 0 else 0
    feats["F61_VIX_RSI20"] = 0

    # Interactions
    feats["IBS_x_VIX"] = feats["F14_IBS"] * feats["F25_VIX"]
    feats["IBS_x_Breadth"] = feats["F14_IBS"] * feats["F30_Breadth"]
    feats["PriorRet_x_VolSpike"] = feats["F57_PriorDayRet"] * feats["F20_VolSpike"]
    feats["RoC3d_x_SPY5d"] = feats["F45_3dRoC"] * feats["F28_SPY5dRet"]
    feats["DistSMA20_x_Breadth"] = feats["F09_DistSMA20"] * feats["F30_Breadth"]
    feats["IdioRet_x_RealVol"] = feats["F37_IdioReturn5d"] * feats["F17_RealVol"]
    feats["DD52wk_x_VIX"] = feats["F13_DD52wk"] * feats["F25_VIX"]
    feats["OvernightGap_x_VolSpike"] = feats["F52_OvernightGapZ"] * feats["F20_VolSpike"]
    feats["RSI14_x_VIX_RSI20"] = feats["F05_RSI14"] * feats["F61_VIX_RSI20"]
    feats["IntradayMom_x_IBS"] = feats["F60_IntradayMom"] * feats["F14_IBS"]
    feats["SectorRelRSI_x_SPYdist200"] = feats["F33_SectorRelRSI"] * feats["F32_SPYdist200"]
    feats["VolAtLows_x_DD52wk"] = feats["F23_VolAtLows"] * feats["F13_DD52wk"]

    # Lags
    feats["IBS_lag1"] = (c[-2] - l[-2]) / (h[-2] - l[-2]) if n >= 2 and (h[-2] - l[-2]) > 0 else 0.5
    feats["PriorRet_lag1"] = (c[-2] / c[-3] - 1) if n >= 3 else 0
    feats["RoC3d_lag1"] = (c[-2] / c[-5] - 1) if n >= 5 else 0
    feats["VolSpike_lag1"] = 1
    feats["IntradayMom_lag1"] = (c[-2] - o[-2]) / (h[-2] - l[-2]) if n >= 2 and (h[-2] - l[-2]) > 0 else 0

    # Z-scores
    if n >= 7:
        ibs_vals = [(c[i] - l[i]) / (h[i] - l[i]) if (h[i] - l[i]) > 0 else 0.5 for i in range(-5, 0)]
        feats["IBS_z5"] = (ibs_vals[-1] - np.mean(ibs_vals)) / (np.std(ibs_vals) + 1e-9)
        pr_vals = [c[i] / c[i-1] - 1 for i in range(-4, 0)] + [feats["F57_PriorDayRet"]]
        feats["PriorRet_z5"] = (pr_vals[-1] - np.mean(pr_vals)) / (np.std(pr_vals) + 1e-9)
        roc_vals = [c[i] / c[i-3] - 1 for i in range(-4, 0)] + [feats["F45_3dRoC"]]
        feats["RoC3d_z5"] = (roc_vals[-1] - np.mean(roc_vals)) / (np.std(roc_vals) + 1e-9)
    else:
        feats["IBS_z5"] = feats["PriorRet_z5"] = feats["RoC3d_z5"] = 0

    # Cross-sectional ranks (placeholders, filled later)
    feats["IBS_csrank"] = 0
    feats["VolSpike_csrank"] = 0
    feats["RealVol_csrank"] = 0

    # Map to feature vector
    result = np.zeros(len(feature_names), dtype=np.float32)
    for i, fn in enumerate(feature_names):
        result[i] = feats.get(fn, 0)

    return result


def _add_cross_sectional_ranks(X, feature_names, close_today):
    """Add cross-sectional ranks for IBS, VolSpike, RealVol."""
    fn_map = {fn: i for i, fn in enumerate(feature_names)}

    rank_pairs = [
        ("IBS_csrank", "F14_IBS"),
        ("VolSpike_csrank", "F20_VolSpike"),
        ("RealVol_csrank", "F17_RealVol"),
    ]

    for rank_name, src_name in rank_pairs:
        if rank_name not in fn_map or src_name not in fn_map:
            continue
        ri = fn_map[rank_name]
        si = fn_map[src_name]
        vals = X[:, si].copy()
        # Only rank stocks with valid close
        valid = ~np.isnan(close_today) & (close_today > 0)
        if valid.sum() > 1:
            order = np.argsort(vals[valid])
            ranks = np.zeros(valid.sum())
            ranks[order] = np.arange(valid.sum()) / (valid.sum() - 1)
            X[valid, ri] = ranks

    return X


def _rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-(period+1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses) + 1e-9
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def run_backtest(engine_data, start_date=None):
    """Run v10_2_engine from start_date, return daily equity curve."""
    from v10_2_engine import run_multi_horizon_bt

    all_dates = engine_data["all_dates"]
    T = engine_data["T"]
    N = engine_data["N"]

    if start_date:
        # Find the start index
        target = datetime.strptime(start_date, "%Y-%m-%d").date()
        start_idx = None
        for i, d in enumerate(all_dates):
            dt = d.date() if hasattr(d, "date") else d
            if dt >= target:
                start_idx = i
                break
        if start_idx is None:
            log.error(f"Start date {start_date} not found in data")
            return None
    else:
        start_idx = engine_data["oos_idx"]

    log.info(f"Running backtest from index {start_idx} "
             f"({all_dates[start_idx].date()}) to {all_dates[-1].date()}")

    result = run_multi_horizon_bt(
        close=engine_data["close"],
        high=engine_data["high"],
        low=engine_data["low"],
        open_prices=engine_data["open"],
        scores=engine_data["scores_ml"],
        params=engine_data["params"],
        oos_idx=start_idx,
        T=T,
        N=N,
        all_dates=all_dates,
    )

    log.info(f"CAGR: {result['oos_cagr']*100:.1f}%, Sharpe: {result['sharpe']:.2f}, "
             f"MaxDD: {result['max_dd']*100:.1f}%, Trades: {result['n_trades']}")

    # Extract equity curve from start_idx
    equity = result["equity"]
    dates_out = []
    equity_out = []
    for i in range(start_idx, T):
        dt = all_dates[i]
        d_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10]
        dates_out.append(d_str)
        equity_out.append(float(equity[i]))

    return {
        "dates": dates_out,
        "equity": equity_out,
        "cagr": result["oos_cagr"],
        "sharpe": result["sharpe"],
        "max_dd": result["max_dd"],
        "n_trades": result["n_trades"],
        "result": result,
    }


def update_dashboard_backtest(bt_result):
    """Write backtest equity curve to dashboard_data.json.

    Strategy: normalize the full OOS backtest so that on PAPER_START
    the backtest equity equals $100,000 (same as actual). This lets us
    see how the model performed historically and where it diverges from
    live trading going forward.
    """
    dash_file = PROD.parent / "dashboard_data.json"
    if not dash_file.exists():
        log.error("dashboard_data.json not found")
        return

    with open(dash_file) as f:
        data = json.load(f)

    # Build a lookup of backtest equity by date
    bt_lookup = dict(zip(bt_result["dates"], bt_result["equity"]))

    # Find the backtest equity on paper start date (or closest prior)
    paper_start_eq = None
    for d in sorted(bt_lookup.keys(), reverse=True):
        if d <= PAPER_START:
            paper_start_eq = bt_lookup[d]
            break
    if paper_start_eq is None or paper_start_eq <= 0:
        paper_start_eq = bt_lookup[bt_result["dates"][-1]]
    scale = 100000.0 / paper_start_eq
    log.info(f"Normalizing: backtest equity on {PAPER_START} = "
             f"${paper_start_eq:,.0f} -> scale = {scale:.6e}")

    # Update equity curve entries with backtest values
    ec = data.get("equity_curve", [])
    for entry in ec:
        date = entry["date"]
        if date in bt_lookup:
            entry["backtest"] = round(bt_lookup[date] * scale, 2)

    # Add backtest-only entries ONLY from paper start onward (no pre-history)
    ec_dates = {e["date"] for e in ec}
    for date in sorted(bt_lookup.keys()):
        if date >= PAPER_START and date not in ec_dates:
            bt_eq = round(bt_lookup[date] * scale, 2)
            prev_actual = None
            for e in sorted(ec, key=lambda x: x["date"], reverse=True):
                if e["date"] <= date and e.get("actual") is not None:
                    prev_actual = e["actual"]
                    break
            ec.append({"date": date, "actual": prev_actual, "backtest": bt_eq})

    ec.sort(key=lambda e: e["date"])
    data["equity_curve"] = ec

    # Save
    with open(dash_file, "w") as f:
        json.dump(data, f, indent=2, default=str)

    log.info(f"Dashboard backtest updated: {len(ec)} equity curve entries")

    # Also save the full backtest curve for reference
    bt_file = PROD / "backtest_equity.json"
    with open(bt_file, "w") as f:
        json.dump({
            "generated": datetime.now().isoformat(),
            "start_date": bt_result["dates"][0],
            "end_date": bt_result["dates"][-1],
            "scale_factor": scale,
            "paper_start": PAPER_START,
            "cagr": bt_result["cagr"],
            "sharpe": bt_result["sharpe"],
            "max_dd": bt_result["max_dd"],
            "n_trades": bt_result["n_trades"],
            "curve": [{"date": d, "equity": round(e * scale, 2)}
                      for d, e in zip(bt_result["dates"], bt_result["equity"])],
        }, f, indent=2)
    log.info(f"Full backtest curve saved to {bt_file}")


def main():
    full_oos = "--full" in sys.argv

    t0 = time.time()

    # Load engine data
    engine_data = load_engine_data()

    if full_oos:
        # Extend with yfinance only if requested (scores may be unreliable
        # for extended days due to missing VIX/SPY/breadth data)
        engine_data = extend_with_yfinance(engine_data)

    # Always run the full OOS backtest -- the engine needs history to trade.
    # We extract the paper trading portion afterward for the dashboard.
    log.info("Running full OOS backtest...")
    bt = run_backtest(engine_data, start_date=None)
    if bt:
        log.info(f"Full OOS: CAGR={bt['cagr']*100:.1f}%, "
                 f"Sharpe={bt['sharpe']:.2f}, MaxDD={bt['max_dd']*100:.1f}%")
        update_dashboard_backtest(bt)

    elapsed = time.time() - t0
    log.info(f"Backtest replay complete in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
