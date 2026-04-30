#!/usr/bin/env python3
"""
v10_2_engine.py — V10.2 engine: expanded universe, 22 factors, up to 4 positions.
Changes from V10.1:
  CHANGE 1: Expanded universe (~1400+ tickers) from precomputed_v4_expanded.parquet
  CHANGE 2: 22nd factor F61_VIX_RSI20 (RSI(20) of VIX as bond-stress proxy)
  CHANGE 3: Max 4 simultaneous positions
  All V10.1 features preserved: fractional shares, MOC/next_open, score exit, etc.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, time, pickle, math
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path("c:/ai-research-team")))

N_WORKERS = max(1, mp.cpu_count() - 1)
ROOT = Path("c:/ai-research-team")
CACHE_DIR = ROOT / "factor_tournament"
CACHE_DIR.mkdir(exist_ok=True)
FACTOR_CACHE = CACHE_DIR / "v10_2_factors.pkl"
V3_PARQUET = ROOT / "market_data" / "precomputed_v3.parquet"
V4_PARQUET = ROOT / "market_data" / "precomputed_v4_expanded.parquet"
VIX_CSV = ROOT / "market_data" / "macro" / "vix_daily.csv"
MCAP_CSV = ROOT / "backtest_results_v4" / "mcap_cache.csv"

FACTOR_NAMES = [
    # V9 stock-level (12)
    "F14_IBS", "F57_PriorDayRet", "F45_3dRoC", "F37_IdioReturn5d",
    "F33_SectorRelRSI", "F09_DistSMA20", "F20_VolSpike", "F23_VolAtLows",
    "F17_RealVol", "F39_LogMcap", "F16_ATRpctile", "F52_OvernightGapZ",
    # V8 market context (7)
    "F25_VIX", "F28_SPY5dRet", "F30_Breadth", "F32_SPYdist200",
    "F43_5dRoC", "F05_RSI14", "F13_DD52wk",
    # V9 extras (2)
    "F54_IBSxRange", "F60_IntradayMom",
    # V10.2 new (1)
    "F61_VIX_RSI20",
]


def rsi_arr(prices, n):
    s = pd.Series(prices)
    delta = s.diff()
    up = delta.clip(lower=0).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    dn = (-delta).clip(lower=0).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    return (100 - 100 / (1 + up / (dn + 1e-9))).values


def _compute_one_ticker(args):
    j, c_j, o_j, h_j, lo_j, v_j, T = args
    out = {}
    stk_ret = np.full(T, np.nan)
    stk_ret[1:] = c_j[1:] / (c_j[:-1] + 1e-15) - 1

    hl = h_j - lo_j
    out['ibs'] = np.where(hl > 0, (c_j - lo_j) / hl, 0.5)

    c5 = np.roll(c_j, 5); c5[:5] = np.nan
    out['roc_5d'] = np.where((c5 > 0) & ~np.isnan(c5), c_j / c5 - 1, np.nan)

    c3 = np.roll(c_j, 3); c3[:3] = np.nan
    out['roc_3d'] = np.where((c3 > 0) & ~np.isnan(c3), c_j / c3 - 1, np.nan)

    out['real_vol_20d'] = (pd.Series(stk_ret).rolling(20, min_periods=20).std() * np.sqrt(252)).values
    out['rsi14'] = rsi_arr(c_j, 14)

    down_mask = stk_ret < 0
    dv = pd.Series(np.where(down_mask, v_j, 0.0)).rolling(5, min_periods=1).sum().values
    tv = pd.Series(v_j).rolling(5, min_periods=1).sum().values
    out['vol_at_lows'] = np.where(tv > 0, dv / tv, np.nan)

    hi252 = pd.Series(h_j).rolling(252, min_periods=20).max().values
    out['dd_52wk'] = np.where(hi252 > 0, (c_j - hi252) / hi252, np.nan)

    out['prior_day_ret'] = stk_ret

    prev_c = np.roll(c_j, 1); prev_c[0] = np.nan
    og = np.where((prev_c > 0) & ~np.isnan(prev_c), o_j / prev_c - 1, np.nan)
    og_mean = pd.Series(og).rolling(20, min_periods=5).mean().values
    og_std = pd.Series(og).rolling(20, min_periods=5).std().values
    out['overnight_gap_z'] = np.where(og_std > 0, (og - og_mean) / og_std, np.nan)

    tr_vals = np.maximum(h_j - lo_j, np.maximum(
        np.abs(h_j - np.roll(c_j, 1)), np.abs(lo_j - np.roll(c_j, 1))))
    atr14 = pd.Series(tr_vals).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    range_exp = np.where(atr14 > 0, (h_j - lo_j) / atr14, np.nan)
    out['ibs_x_range'] = (1 - out['ibs']) * np.nan_to_num(range_exp, nan=1.0)

    out['intraday_mom'] = np.where(o_j > 0, (c_j - o_j) / o_j, np.nan)

    sma20 = pd.Series(c_j).rolling(20, min_periods=20).mean().values
    out['dist_sma20'] = np.where(sma20 > 0, (c_j - sma20) / sma20, np.nan)

    vol20 = pd.Series(v_j).rolling(20, min_periods=5).mean().values
    out['vol_spike'] = np.where(vol20 > 0, v_j / vol20, np.nan)

    out['log_price'] = np.log10(np.maximum(c_j, 1e-9))
    out['rsi5'] = rsi_arr(c_j, 5)

    return j, out


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING — Expanded universe
# ═══════════════════════════════════════════════════════════════════════════

def _build_expanded_parquet():
    """Build precomputed_v4_expanded.parquet from v3 + new ticker downloads."""
    import yfinance as yf

    print("  Loading V3 parquet (972 tickers)...")
    v3_cols = ["close", "open", "high", "low", "volume", "atr14"]
    v3 = pd.read_parquet(V3_PARQUET, columns=v3_cols).reset_index()
    v3["date"] = pd.to_datetime(v3["date"])
    v3_tickers = set(v3["ticker"].unique())
    print(f"    V3: {len(v3_tickers)} tickers, {v3['date'].min().date()} to {v3['date'].max().date()}")

    # Get expanded ticker list from the scan
    scan_pq = pd.read_parquet(ROOT / "market_data" / "expanded_scan" / "ohlcv_2018_2026.parquet")
    scan_tickers = sorted(set(c.rsplit("_", 1)[0] for c in scan_pq.columns if c.endswith("_c")))
    new_tickers = sorted(set(scan_tickers) - v3_tickers)
    print(f"    Scan: {len(scan_tickers)} tickers, {len(new_tickers)} new to download")

    # Download new tickers from yfinance
    new_frames = []
    batch_sz = 50
    for i in range(0, len(new_tickers), batch_sz):
        batch = new_tickers[i:i + batch_sz]
        batch_end = min(i + batch_sz, len(new_tickers))
        print(f"    Downloading batch {i+1}-{batch_end} of {len(new_tickers)}...")
        try:
            data = yf.download(batch, start="2001-01-01", end="2026-03-17",
                               auto_adjust=True, threads=True, progress=False,
                               group_by="ticker")
            for t in batch:
                try:
                    if len(batch) == 1:
                        df_t = data.copy()
                    else:
                        if t not in data.columns.get_level_values(0):
                            continue
                        df_t = data[t].copy()
                    df_t = df_t.dropna(subset=["Close"])
                    if len(df_t) < 252:
                        continue
                    df_t = df_t.reset_index()
                    frame = pd.DataFrame({
                        "date": pd.to_datetime(df_t["Date"]),
                        "ticker": t,
                        "open": df_t["Open"].values,
                        "high": df_t["High"].values,
                        "low": df_t["Low"].values,
                        "close": df_t["Close"].values,
                        "volume": df_t["Volume"].values,
                    })
                    new_frames.append(frame)
                except Exception:
                    pass
        except Exception as e:
            print(f"      Batch error: {e}")
        time.sleep(1)

    print(f"    Downloaded {len(new_frames)} new tickers with >= 252 days")

    # Combine v3 + new
    v3_core = v3[["date", "ticker", "open", "high", "low", "close", "volume"]].copy()
    if new_frames:
        new_df = pd.concat(new_frames, ignore_index=True)
        combined = pd.concat([v3_core, new_df], ignore_index=True)
    else:
        combined = v3_core

    # Drop tickers with < 252 days
    days_per = combined.groupby("ticker").size()
    good_tickers = days_per[days_per >= 252].index
    combined = combined[combined["ticker"].isin(good_tickers)]

    # Compute derived columns needed by load_data compatibility
    result_frames = []
    for ticker, gdf in combined.groupby("ticker"):
        gdf = gdf.sort_values("date").copy()
        # RSI(4)
        delta = gdf["close"].diff()
        up = delta.clip(lower=0).ewm(alpha=1/4, min_periods=4, adjust=False).mean()
        dn = (-delta).clip(lower=0).ewm(alpha=1/4, min_periods=4, adjust=False).mean()
        gdf["rsi4"] = 100 - 100 / (1 + up / (dn + 1e-9))
        # relative_volume
        vol20 = gdf["volume"].rolling(20, min_periods=5).mean()
        gdf["relative_volume"] = gdf["volume"] / (vol20 + 1)
        # ATR14
        tr = pd.concat([gdf["high"] - gdf["low"],
                         (gdf["high"] - gdf["close"].shift()).abs(),
                         (gdf["low"] - gdf["close"].shift()).abs()], axis=1).max(axis=1)
        gdf["atr14"] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        result_frames.append(gdf)

    combined = pd.concat(result_frames, ignore_index=True)
    combined = combined.set_index(["date", "ticker"]).sort_index()

    final_tickers = combined.index.get_level_values("ticker").nunique()
    print(f"    Final expanded parquet: {final_tickers} tickers")
    combined.to_parquet(V4_PARQUET)
    print(f"    Saved to {V4_PARQUET}")
    return final_tickers


def _load_expanded_data(parquet_path, start="2001-01-01", end="2026-03-17"):
    """Load expanded universe OHLCV from parquet into (T,N) arrays."""
    import yfinance as yf

    print(f"  Loading {parquet_path}...")
    cols = ["close", "open", "high", "low", "volume", "atr14"]
    import pyarrow.parquet as paq
    schema_fields = [f.name for f in paq.read_schema(str(parquet_path))]
    use_cols = [c for c in cols if c in schema_fields]
    raw = pd.read_parquet(parquet_path, columns=use_cols).reset_index()
    raw["date"] = pd.to_datetime(raw["date"])
    # Filter to date range early to save memory
    raw = raw[(raw["date"] >= start) & (raw["date"] <= end)]

    # Drop tickers with < 252 days
    days_per = raw.groupby("ticker").size()
    good_tickers = days_per[days_per >= 252].index
    dropped = len(days_per) - len(good_tickers)
    if dropped > 0:
        print(f"    Dropped {dropped} tickers with < 252 trading days")
    raw = raw[raw["ticker"].isin(good_tickers)]

    def piv(col):
        return raw.pivot(index="date", columns="ticker", values=col).sort_index()

    close_p = piv("close"); open_p = piv("open")
    high_p = piv("high"); low_p = piv("low")
    vol_p = piv("volume")
    atr14_p = piv("atr14") if "atr14" in raw.columns else None

    tickers = list(close_p.columns); N = len(tickers)
    all_dates = close_p.loc[start:end].index; T = len(all_dates)

    def bt(df): return df.loc[start:end].values.astype(float)
    c = bt(close_p); o = bt(open_p); h = bt(high_p); lo = bt(low_p); v = bt(vol_p)

    if atr14_p is not None:
        atr14 = bt(atr14_p)
    else:
        # Compute ATR14 from OHLCV
        atr14 = np.full((T, N), np.nan)
        for j in range(N):
            tr = np.maximum(h[:, j] - lo[:, j], np.maximum(
                np.abs(h[:, j] - np.roll(c[:, j], 1)),
                np.abs(lo[:, j] - np.roll(c[:, j], 1))))
            atr14[:, j] = pd.Series(tr).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values

    # ETF data for market context
    print("  Downloading SPY/VIX reference data...")
    spy_data = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    spy_close_s = spy_data["Close"].squeeze()
    spy_close_s.index = pd.to_datetime(spy_close_s.index)
    spy = spy_close_s.reindex(all_dates).ffill().values.astype(float).ravel()

    # VIX
    vix_df = pd.read_csv(VIX_CSV, parse_dates=["Date"]).set_index("Date")["VIX_Close"]
    vix = vix_df.reindex(all_dates).ffill().values.astype(float).ravel()

    print(f"    {N} tickers, {T} days ({all_dates[0].date()} to {all_dates[-1].date()})")

    return {
        "close": c, "open_": o, "high": h, "low": lo, "volume": v,
        "atr14": atr14, "tickers": tickers, "all_dates": all_dates,
        "T": T, "N": N, "spy": spy, "vix": vix,
    }


def compute_and_cache_factors(force=False, use_v3=False):
    """Compute all V10.2 factors on expanded universe. Cache to pickle."""
    cache = FACTOR_CACHE
    if use_v3:
        cache = CACHE_DIR / "v10_2_factors_v3.pkl"

    if cache.exists() and not force:
        print(f"Loading cached factors from {cache}...")
        t0 = time.time()
        with open(cache, "rb") as f:
            data = pickle.load(f)
        print(f"  Loaded in {time.time()-t0:.1f}s")
        return data

    if use_v3:
        print("Computing V10.2 factors on V3 universe (972 tickers)...")
        parquet = V3_PARQUET
    else:
        print("Computing V10.2 factors on EXPANDED universe...")
        # Build expanded parquet if needed
        if not V4_PARQUET.exists():
            print("  Expanded parquet not found. Building it now...")
            _build_expanded_parquet()
        parquet = V4_PARQUET

    t0 = time.time()
    d = _load_expanded_data(parquet)
    T, N = d["T"], d["N"]
    c, o, h, lo, v = d["close"], d["open_"], d["high"], d["low"], d["volume"]
    spy, vix = d["spy"], d["vix"]

    # Parallel per-ticker factor computation
    print(f"  Computing per-ticker factors for {N} tickers...")
    args_list = [(j, c[:, j].copy(), o[:, j].copy(), h[:, j].copy(),
                  lo[:, j].copy(), v[:, j].copy(), T) for j in range(N)]

    ticker_results = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for j, out in pool.map(_compute_one_ticker, args_list):
            ticker_results[j] = out

    per_ticker_keys = list(ticker_results[0].keys())
    factors = {}
    for key in per_ticker_keys:
        arr = np.full((T, N), np.nan)
        for j in range(N):
            arr[:, j] = ticker_results[j][key]
        factors[key] = arr

    # Market-level factors
    spy_ret = np.zeros(T); spy_ret[1:] = spy[1:] / (spy[:-1] + 1e-15) - 1
    spy_5d = np.zeros(T); spy_5d[5:] = spy[5:] / (spy[:-5] + 1e-15) - 1
    spy_200 = pd.Series(spy).rolling(200, min_periods=200).mean().values
    spy_dist200 = np.where(spy_200 > 0, (spy - spy_200) / spy_200, 0.0)

    sma20_arr = np.zeros((T, N))
    for j in range(N):
        sma20_arr[:, j] = pd.Series(c[:, j]).rolling(20, min_periods=20).mean().values
    breadth = np.nanmean(c > sma20_arr, axis=1)

    for key, vals in [('vix_level', vix), ('spy_5d_ret', spy_5d),
                      ('breadth', breadth), ('spy_dist200', spy_dist200)]:
        factors[key] = np.broadcast_to(vals[:, None], (T, N)).copy()

    # Sector relative RSI
    spy_rsi5 = rsi_arr(spy, 5)
    factors['sector_rel_rsi'] = factors['rsi5'] - spy_rsi5[:, None]

    # Idiosyncratic 5d return
    factors['idio_ret_5d'] = factors['roc_5d'] - spy_5d[:, None]

    # Log market cap
    try:
        mcap_d = pd.read_csv(MCAP_CSV, index_col=0)["marketCap"].to_dict()
        ticker_mcap = np.array([mcap_d.get(t, np.nan) for t in d["tickers"]], dtype=float)
        factors['log_mcap'] = np.broadcast_to(np.log10(np.maximum(ticker_mcap, 1))[None, :], (T, N)).copy()
    except Exception:
        factors['log_mcap'] = np.full((T, N), np.nan)

    # ATR percentile (cross-sectional)
    atr14 = d["atr14"]
    atr_rank = np.full((T, N), np.nan)
    for day in range(T):
        vals = atr14[day]; valid = ~np.isnan(vals); n_v = valid.sum()
        if n_v > 1:
            ranks = np.full(N, np.nan)
            ranks[valid] = vals[valid].argsort().argsort().astype(float) / (n_v - 1)
            atr_rank[day] = ranks
    factors['atr_pctile'] = atr_rank

    # CHANGE 2: F61_VIX_RSI20 — RSI(20) of VIX, broadcast to all stocks
    print("  Computing F61_VIX_RSI20...")
    vix_rsi20 = rsi_arr(vix, 20)
    factors['vix_rsi20'] = np.broadcast_to(vix_rsi20[:, None], (T, N)).copy()

    # Factor name -> data key mapping
    FKEY_MAP = {
        "F14_IBS": "ibs", "F43_5dRoC": "roc_5d", "F17_RealVol": "real_vol_20d",
        "F05_RSI14": "rsi14", "F23_VolAtLows": "vol_at_lows", "F13_DD52wk": "dd_52wk",
        "F57_PriorDayRet": "prior_day_ret", "F52_OvernightGapZ": "overnight_gap_z",
        "F54_IBSxRange": "ibs_x_range", "F60_IntradayMom": "intraday_mom",
        "F45_3dRoC": "roc_3d", "F09_DistSMA20": "dist_sma20",
        "F20_VolSpike": "vol_spike", "F33_SectorRelRSI": "sector_rel_rsi",
        "F37_IdioReturn5d": "idio_ret_5d",
        "F25_VIX": "vix_level", "F28_SPY5dRet": "spy_5d_ret",
        "F30_Breadth": "breadth", "F32_SPYdist200": "spy_dist200",
        "F39_LogMcap": "log_mcap", "F16_ATRpctile": "atr_pctile",
        "F61_VIX_RSI20": "vix_rsi20",
    }

    factor_arrays = {}
    for fname in FACTOR_NAMES:
        factor_arrays[fname] = factors[FKEY_MAP[fname]]

    # OOS index
    oos_idx = 0
    for i, dt in enumerate(d["all_dates"]):
        if str(dt.date()) >= "2019-01-01":
            oos_idx = i; break

    # Forward 1d returns for IC analysis
    fwd1 = np.full((T, N), np.nan)
    fwd1[:-1] = c[1:] / (c[:-1] + 1e-15) - 1

    data = {
        "close": c, "open": o, "high": h, "low": lo,
        "volume": v, "factor_arrays": factor_arrays,
        "T": T, "N": N, "oos_idx": oos_idx,
        "all_dates": d["all_dates"], "tickers": d["tickers"],
        "fwd1": fwd1, "vix": vix,
    }

    print(f"  Saving cache to {cache}...")
    with open(cache, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  V10.2 factors computed and cached in {time.time()-t0:.1f}s")
    return data


# ═══════════════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════════════

def precompute_scores(factor_arrays, weights, close, T, N):
    active = [(fname, w) for fname, w in weights.items() if w > 0 and fname in factor_arrays]
    if not active:
        return np.full((T, N), np.nan)

    fnames = [f for f, _ in active]
    w_arr = np.array([w for _, w in active])
    total_w = w_arr.sum()
    cube = np.stack([factor_arrays[f] for f in fnames], axis=2)
    n_f = len(fnames)

    scores = np.full((T, N), np.nan)
    for day in range(T):
        valid = ~np.isnan(close[day]) & (close[day] > 0)
        vi = np.where(valid)[0]
        if len(vi) < 2:
            continue
        fvals = cube[day, vi, :]
        weighted_sum = np.zeros(len(vi))
        for f_idx in range(n_f):
            col = fvals[:, f_idx]
            not_nan = ~np.isnan(col)
            if not_nan.sum() < 2:
                continue
            ranks = np.full(len(vi), np.nan)
            order = col[not_nan].argsort().argsort()
            n_v = not_nan.sum()
            ranks[not_nan] = order.astype(float) / max(n_v - 1, 1)
            weighted_sum += np.nan_to_num(ranks) * w_arr[f_idx]
        scores[day, vi] = weighted_sum / total_w
    return scores


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-HORIZON CASH ENGINE — V10.2
# Same as V10.1 but supports up to 4 positions
# ═══════════════════════════════════════════════════════════════════════════

def run_multi_horizon_bt(close, high, low, open_prices, scores, params,
                         oos_idx, T, N, all_dates=None, real_vol_arr=None):
    """V10.2 multi-horizon $1K cash backtest. Supports up to 4 positions."""
    slip = params.get("slippage", 0.001)
    capital = 1000.0

    max_positions = params.get("max_positions", 1)
    capital_split = params.get("capital_split", "all_in_one")
    hold_mode = params.get("hold_mode", "fixed")
    base_hold = params.get("base_hold_days", 1)
    short_hold = params.get("short_hold", 1)
    long_hold = params.get("long_hold", 3)
    hold_score_split = params.get("hold_score_split", 0.60)
    low_vol_hold = params.get("low_vol_hold", 3)
    high_vol_hold = params.get("high_vol_hold", 1)
    vol_split_pctile = params.get("vol_split_pctile", 0.50)
    score_threshold = params.get("score_threshold", 0.55)
    max_price = params.get("max_price", 9999)
    ibs_max = params.get("ibs_max", 0.30)
    stop_pct = params.get("stop_pct", 0.02)
    use_profit_target = params.get("use_profit_target", False)
    profit_target_pct = params.get("profit_target_pct", 0.05)
    use_trailing = params.get("use_trailing", False)
    trail_pct = params.get("trail_pct", 0.03)
    trail_activation = params.get("trail_activation", 0.04)
    min_deploy_pct = params.get("min_deploy_pct", 1.0)
    skip_monday = params.get("skip_monday", False)
    skip_friday = params.get("skip_friday", False)
    fractional = params.get("fractional_shares", True)
    entry_mode = params.get("entry_mode", "moc")
    use_score_exit = params.get("use_score_exit", False)
    score_exit_floor = params.get("score_exit_floor", 0.20)
    max_hold_days = params.get("max_hold_days", None)

    hl = high - low

    vol_threshold_per_day = None
    if hold_mode == "volatility_based" and real_vol_arr is not None:
        vol_threshold_per_day = np.full(T, np.nan)
        for day in range(T):
            vals = real_vol_arr[day]
            valid = vals[~np.isnan(vals)]
            if len(valid) > 1:
                vol_threshold_per_day[day] = np.percentile(valid, vol_split_pctile * 100)

    settled_cash = capital
    settling = []
    positions = []  # each: [tk, sh, ep, ed, th, hc, last_price, nan_days]
    equity = np.zeros(T)
    trade_rets = []
    # Track signals per day and idle days for diagnostics
    signals_per_day = []
    idle_days = 0
    MAX_NAN_DAYS = 3  # force-exit after this many consecutive NaN closes

    for day in range(T):
        new_settling = []
        for amt, avail_day in settling:
            if day >= avail_day:
                settled_cash += amt
            else:
                new_settling.append((amt, avail_day))
        settling = new_settling

        pv = settled_cash + sum(a for a, _ in settling)
        for pos in positions:
            tk, sh = pos[0], pos[1]
            p = close[day, tk]
            if not np.isnan(p) and p > 0:
                pv += sh * p
            else:
                # Carry forward last known price for equity valuation
                pv += sh * pos[6]
        equity[day] = pv

        if day == 0:
            continue

        # ── Exits ──
        new_positions = []
        for pos in positions:
            tk, sh, ep, ed, th, hc = pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]
            last_price, nan_days = pos[6], pos[7]
            p = close[day, tk]

            if np.isnan(p) or p <= 0:
                nan_days += 1
                if nan_days >= MAX_NAN_DAYS:
                    # Force-exit at last known price with slippage
                    fill = last_price * (1 - slip)
                    proceeds = sh * fill
                    settling.append((proceeds, day + 1))
                    held = day - ed
                    cost_basis = sh * ep * (1 + slip)
                    pnl_pct = (proceeds / cost_basis - 1) * 100 if cost_basis > 0 else 0
                    trade_rets.append((pnl_pct, held, "nan_exit", ed, th))
                else:
                    new_positions.append([tk, sh, ep, ed, th, hc, last_price, nan_days])
                continue

            # Valid price — reset NaN counter, update last known price
            nan_days = 0
            last_price = p

            held = day - ed
            today_low = low[day, tk] if not np.isnan(low[day, tk]) else p
            new_hc = max(hc, p)
            reason = None
            fill = None

            if stop_pct > 0 and (today_low <= ep * (1 - stop_pct) or p <= ep * (1 - stop_pct)):
                fill = p * (1 - slip)  # Realistic MOC fill: exit at close, not fictional stop level
                reason = "stop"
            elif use_profit_target and p >= ep * (1 + profit_target_pct):
                fill = p * (1 - slip)
                reason = "profit_target"
            elif use_trailing and new_hc >= ep * (1 + trail_activation) and p <= new_hc * (1 - trail_pct):
                fill = new_hc * (1 - trail_pct) * (1 - slip)
                reason = "trailing"
            elif use_score_exit and held >= 1:
                sc_now = scores[day, tk]
                if not np.isnan(sc_now) and sc_now < score_exit_floor:
                    fill = p * (1 - slip)
                    reason = "score_exit"
            if reason is None:
                effective_hold = max_hold_days if (use_score_exit and max_hold_days) else th
                if held >= effective_hold:
                    fill = p * (1 - slip)
                    reason = "hold_expiry"

            if reason:
                proceeds = sh * fill
                settling.append((proceeds, day + 1))
                cost_basis = sh * ep * (1 + slip)
                pnl_pct = (proceeds / cost_basis - 1) * 100 if cost_basis > 0 else 0
                trade_rets.append((pnl_pct, held, reason, ed, th))
            else:
                new_positions.append([tk, sh, ep, ed, th, new_hc, last_price, nan_days])

        positions = new_positions

        # ── Entries ──
        if settled_cash < 10:
            if day >= oos_idx:
                idle_days += 1
            continue
        open_slots = max_positions - len(positions)
        if open_slots <= 0:
            continue

        if all_dates is not None:
            wd = all_dates[day].weekday()
        else:
            wd = 2
        if skip_monday and wd == 0:
            continue
        if skip_friday and wd == 4:
            continue

        if entry_mode == "next_open":
            entry_day = day + 1
            if entry_day >= T:
                continue
        else:
            entry_day = day

        day_scores = scores[day]
        day_close = close[day]
        held_set = {pos[0] for pos in positions}

        cands = []
        for j in range(N):
            sc = day_scores[j]
            if np.isnan(sc) or sc < score_threshold:
                continue
            p = day_close[j]
            if np.isnan(p) or p <= 0 or p > max_price:
                continue
            if j in held_set:
                continue
            ibs_val = (p - low[day, j]) / hl[day, j] if hl[day, j] > 0 else 0.5
            if ibs_val > ibs_max:
                continue
            cands.append((j, sc))

        if day >= oos_idx:
            signals_per_day.append(len(cands))
            if len(cands) == 0:
                idle_days += 1

        if not cands:
            continue

        cands.sort(key=lambda x: -x[1])
        selected = cands[:open_slots]

        for tk, sc in selected:
            if hold_mode == "fixed":
                target_hold = base_hold
            elif hold_mode == "score_based":
                target_hold = long_hold if sc >= hold_score_split else short_hold
            elif hold_mode == "volatility_based":
                if vol_threshold_per_day is not None and not np.isnan(vol_threshold_per_day[day]) and real_vol_arr is not None:
                    rv = real_vol_arr[day, tk]
                    thresh = vol_threshold_per_day[day]
                    target_hold = high_vol_hold if (not np.isnan(rv) and rv >= thresh) else low_vol_hold
                else:
                    target_hold = low_vol_hold
            else:
                target_hold = base_hold

            if capital_split == "all_in_one":
                invest = settled_cash * min_deploy_pct
            elif capital_split == "equal":
                invest = settled_cash * min_deploy_pct / max(len(selected), 1)
            elif capital_split == "tiered":
                if tk == selected[0][0]:
                    invest = settled_cash * min_deploy_pct * 0.6
                else:
                    invest = settled_cash * min_deploy_pct * 0.4 / max(len(selected) - 1, 1)
            else:
                invest = settled_cash * min_deploy_pct

            if entry_mode == "next_open":
                raw_entry = open_prices[entry_day, tk]
                if np.isnan(raw_entry) or raw_entry <= 0:
                    continue
            else:
                raw_entry = close[day, tk]

            fill_p = raw_entry * (1 + slip)

            if fractional:
                shares = invest / fill_p
            else:
                shares = int(invest / fill_p)
                if shares < 1:
                    continue

            cost = shares * fill_p
            if cost > settled_cash:
                if fractional:
                    shares = settled_cash / fill_p
                else:
                    shares = int(settled_cash / fill_p)
                    if shares < 1:
                        continue
                cost = shares * fill_p

            if shares <= 0 or cost <= 0:
                continue

            settled_cash -= cost
            positions.append([tk, shares, raw_entry, entry_day, target_hold, raw_entry, raw_entry, 0])

            if capital_split == "all_in_one":
                break

    # ── OOS metrics ──
    oos_eq = equity[oos_idx:]
    n_years = len(oos_eq) / 252

    if oos_eq[0] > 0 and len(oos_eq) > 1 and oos_eq[-1] > 0:
        cagr = (oos_eq[-1] / oos_eq[0]) ** (1 / max(n_years, 0.01)) - 1
    else:
        cagr = -1

    oos_rets = np.diff(oos_eq) / (oos_eq[:-1] + 1e-15)
    oos_rets = oos_rets[np.isfinite(oos_rets)]
    sharpe = (np.mean(oos_rets) / (np.std(oos_rets, ddof=1) + 1e-15) * np.sqrt(252)) if len(oos_rets) > 1 else 0

    peak = np.maximum.accumulate(oos_eq)
    dd = (oos_eq - peak) / (peak + 1e-15)
    max_dd = dd.min()

    oos_trades = [t for t in trade_rets if t[3] >= oos_idx]
    n_trades = len(oos_trades)
    wins = [t for t in oos_trades if t[0] > 0]
    losses = [t for t in oos_trades if t[0] <= 0]
    wr = len(wins) / n_trades * 100 if n_trades else 0
    tpy = n_trades / max(n_years, 0.01)
    avg_win = np.mean([t[0] for t in wins]) if wins else 0
    avg_loss = np.mean([t[0] for t in losses]) if losses else 0

    hold_dist = {}
    for t in oos_trades:
        hd = t[1]
        hold_dist[hd] = hold_dist.get(hd, 0) + 1

    full_hold_count = sum(1 for t in oos_trades if t[2] == "hold_expiry")
    pct_full_hold = full_hold_count / n_trades * 100 if n_trades else 0

    exit_reasons = {}
    for t in oos_trades:
        r = t[2]
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    yearly = {}
    if all_dates is not None:
        for i in range(len(oos_eq) - 1):
            yr = all_dates[oos_idx + i].year
            if yr not in yearly:
                yearly[yr] = []
            daily_ret = oos_eq[i + 1] / (oos_eq[i] + 1e-15) - 1
            yearly[yr].append(daily_ret)
        yearly_cagr = {}
        for yr, rets in yearly.items():
            cum = 1.0
            for r in rets:
                cum *= (1 + r)
            yearly_cagr[yr] = cum - 1
    else:
        yearly_cagr = {}

    avg_signals = np.mean(signals_per_day) if signals_per_day else 0
    idle_per_yr = idle_days / max(n_years, 0.01)

    # Capital utilization: % of days with at least 1 position
    oos_days_with_pos = 0
    oos_total_days = len(oos_eq) - 1
    # Approximate: use trade count and hold durations
    total_position_days = sum(t[1] for t in oos_trades)
    cap_util = total_position_days / (max_positions * max(oos_total_days, 1)) * 100

    return {
        "oos_cagr": cagr, "sharpe": sharpe, "max_dd": max_dd,
        "win_rate": wr, "n_trades": n_trades, "trades_per_yr": tpy,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "pct_full_hold": pct_full_hold, "exit_reasons": exit_reasons,
        "final_equity": oos_eq[-1] if len(oos_eq) > 0 else 0,
        "equity": equity, "oos_trades": oos_trades, "hold_dist": hold_dist,
        "yearly_cagr": yearly_cagr, "avg_signals_per_day": avg_signals,
        "idle_days_per_yr": idle_per_yr, "cap_utilization": cap_util,
    }


def run_backtest(params_dict, factor_data):
    fa = factor_data['factor_arrays']
    c = factor_data['close']
    T, N = factor_data['T'], factor_data['N']
    weights = params_dict.get('weights', {fname: 0.5 for fname in FACTOR_NAMES})
    scores = precompute_scores(fa, weights, c, T, N)
    return run_multi_horizon_bt(
        close=c, high=factor_data['high'], low=factor_data['low'],
        open_prices=factor_data['open'], scores=scores, params=params_dict,
        oos_idx=factor_data['oos_idx'], T=T, N=N,
        all_dates=factor_data['all_dates'],
    )


# ═══════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════

V10_WINNER_WEIGHTS = {
    "F14_IBS": 0.0, "F57_PriorDayRet": 0.6, "F45_3dRoC": 0.5,
    "F37_IdioReturn5d": 0.6, "F33_SectorRelRSI": 0.6, "F09_DistSMA20": 0.9,
    "F20_VolSpike": 0.9, "F23_VolAtLows": 0.8, "F17_RealVol": 0.8,
    "F39_LogMcap": 0.3, "F16_ATRpctile": 1.0, "F52_OvernightGapZ": 0.2,
    "F25_VIX": 0.8, "F28_SPY5dRet": 1.0, "F30_Breadth": 0.0,
    "F32_SPYdist200": 1.0, "F43_5dRoC": 0.9, "F05_RSI14": 0.5,
    "F13_DD52wk": 0.3, "F54_IBSxRange": 0.5, "F60_IntradayMom": 0.7,
}

V10_BASE_PARAMS = {
    "hold_mode": "fixed", "base_hold_days": 3, "max_positions": 2,
    "capital_split": "equal", "stop_pct": 0.005, "score_threshold": 0.35,
    "max_price": 9999, "ibs_max": 0.10, "skip_monday": False,
    "skip_friday": True, "min_deploy_pct": 1.0, "fractional_shares": True,
    "entry_mode": "moc",
    "use_profit_target": True, "profit_target_pct": 0.12,
    "use_trailing": True, "trail_pct": 0.035, "trail_activation": 0.04,
}


def fmt_res(r):
    return (f"CAGR={r['oos_cagr']*100:+.1f}%  Sharpe={r['sharpe']:.2f}  "
            f"MaxDD={r['max_dd']*100:.1f}%  WR={r['win_rate']:.1f}%  "
            f"Tr/yr={r['trades_per_yr']:.0f}")


def run_diagnostics():
    print("\n" + "="*80)
    print("V10.2 ENGINE DIAGNOSTICS — EXPANDED UNIVERSE")
    print("="*80)

    results_text = []

    # Load both universes
    print("\n[1/3] Loading V3 universe (972 tickers)...")
    data_v3 = compute_and_cache_factors(force=False, use_v3=True)
    T3, N3 = data_v3['T'], data_v3['N']

    print("\n[2/3] Loading EXPANDED universe...")
    data_exp = compute_and_cache_factors(force=False, use_v3=False)
    Te, Ne = data_exp['T'], data_exp['N']

    print(f"\n  V3: {N3} tickers x {T3} days")
    print(f"  Expanded: {Ne} tickers x {Te} days")

    # ── DIAG A: Universe Expansion Impact ──
    print("\n── DIAG A: UNIVERSE EXPANSION IMPACT ──")
    results_text.append("DIAG A — UNIVERSE EXPANSION IMPACT")
    results_text.append(f"{'Universe':>20s}  {'CAGR':>8s}  {'Sharpe':>7s}  {'MaxDD':>7s}  {'Tr/yr':>6s}  {'Sig/day':>8s}  {'Idle/yr':>8s}")
    results_text.append("-" * 75)

    # V3 universe
    print("  Running V3 backtest...")
    scores_v3 = precompute_scores(data_v3['factor_arrays'], V10_WINNER_WEIGHTS, data_v3['close'], T3, N3)
    p = {**V10_BASE_PARAMS, "weights": V10_WINNER_WEIGHTS}
    r_v3 = run_multi_horizon_bt(data_v3['close'], data_v3['high'], data_v3['low'],
                                 data_v3['open'], scores_v3, p,
                                 data_v3['oos_idx'], T3, N3, data_v3['all_dates'])
    line = (f"{'V3 ('+str(N3)+' tk)':>20s}  {r_v3['oos_cagr']*100:>+7.1f}%  {r_v3['sharpe']:>7.2f}  "
            f"{r_v3['max_dd']*100:>6.1f}%  {r_v3['trades_per_yr']:>5.0f}  "
            f"{r_v3['avg_signals_per_day']:>7.1f}  {r_v3['idle_days_per_yr']:>7.0f}")
    print(f"  {line}")
    results_text.append(line)

    # Expanded universe
    print("  Running expanded backtest...")
    scores_exp = precompute_scores(data_exp['factor_arrays'], V10_WINNER_WEIGHTS, data_exp['close'], Te, Ne)
    r_exp = run_multi_horizon_bt(data_exp['close'], data_exp['high'], data_exp['low'],
                                  data_exp['open'], scores_exp, p,
                                  data_exp['oos_idx'], Te, Ne, data_exp['all_dates'])
    line = (f"{'Exp ('+str(Ne)+' tk)':>20s}  {r_exp['oos_cagr']*100:>+7.1f}%  {r_exp['sharpe']:>7.2f}  "
            f"{r_exp['max_dd']*100:>6.1f}%  {r_exp['trades_per_yr']:>5.0f}  "
            f"{r_exp['avg_signals_per_day']:>7.1f}  {r_exp['idle_days_per_yr']:>7.0f}")
    print(f"  {line}")
    results_text.append(line)
    results_text.append("")

    # ── DIAG B: 4 Positions Test ──
    print("\n── DIAG B: POSITION COUNT TEST (expanded universe) ──")
    results_text.append("DIAG B — POSITION COUNT (expanded universe)")
    results_text.append(f"{'Positions':>10s}  {'CAGR':>8s}  {'Sharpe':>7s}  {'MaxDD':>7s}  {'Tr/yr':>6s}  {'CapUtil%':>9s}")
    results_text.append("-" * 55)

    for n_pos in [1, 2, 3, 4]:
        split = "all_in_one" if n_pos == 1 else "equal"
        p_pos = {**V10_BASE_PARAMS, "max_positions": n_pos, "capital_split": split,
                 "weights": V10_WINNER_WEIGHTS}
        r = run_multi_horizon_bt(data_exp['close'], data_exp['high'], data_exp['low'],
                                  data_exp['open'], scores_exp, p_pos,
                                  data_exp['oos_idx'], Te, Ne, data_exp['all_dates'])
        line = (f"{n_pos:>10d}  {r['oos_cagr']*100:>+7.1f}%  {r['sharpe']:>7.2f}  "
                f"{r['max_dd']*100:>6.1f}%  {r['trades_per_yr']:>5.0f}  "
                f"{r['cap_utilization']:>8.1f}%")
        print(f"  {line}")
        results_text.append(line)
    results_text.append("")

    # ── DIAG C: F61 VIX RSI Factor IC ──
    print("\n── DIAG C: F61_VIX_RSI20 FACTOR IC ──")
    results_text.append("DIAG C — F61_VIX_RSI20 INFORMATION COEFFICIENT")
    results_text.append("-" * 60)

    fa = data_exp['factor_arrays']
    fwd1 = data_exp['fwd1']
    c = data_exp['close']
    oos = data_exp['oos_idx']

    # Compute daily rank IC for F61 and compare to top V9 factors
    test_factors = ["F61_VIX_RSI20", "F14_IBS", "F09_DistSMA20", "F43_5dRoC",
                    "F17_RealVol", "F57_PriorDayRet", "F54_IBSxRange"]

    results_text.append(f"{'Factor':>20s}  {'MeanIC':>8s}  {'IC>0%':>6s}  {'t-stat':>7s}  {'Stable':>7s}")
    results_text.append("-" * 55)

    for fname in test_factors:
        if fname not in fa:
            continue
        f_arr = fa[fname]
        ics = []
        for day in range(oos, Te - 1):
            valid = ~np.isnan(c[day]) & (c[day] > 0) & ~np.isnan(fwd1[day]) & ~np.isnan(f_arr[day])
            vi = np.where(valid)[0]
            if len(vi) < 20:
                continue
            from scipy.stats import spearmanr
            rho, _ = spearmanr(f_arr[day, vi], fwd1[day, vi])
            if not np.isnan(rho):
                ics.append(rho)

        if len(ics) < 10:
            line = f"{fname:>20s}  {'N/A':>8s}  {'N/A':>6s}  {'N/A':>7s}  {'N/A':>7s}"
        else:
            ics = np.array(ics)
            mean_ic = np.mean(ics)
            ic_pos = np.mean(ics > 0) * 100
            t_stat = mean_ic / (np.std(ics, ddof=1) / np.sqrt(len(ics)))

            # Stability: IC across 5 equal periods
            n_per = len(ics) // 5
            period_ics = []
            for i in range(5):
                chunk = ics[i * n_per:(i + 1) * n_per]
                if len(chunk) > 0:
                    period_ics.append(np.mean(chunk))
            stability = np.mean(np.array(period_ics) > 0) * 100 if period_ics else 0

            line = (f"{fname:>20s}  {mean_ic:>+8.4f}  {ic_pos:>5.1f}%  {t_stat:>7.2f}  "
                    f"{stability:>5.0f}/5")

        print(f"  {line}")
        results_text.append(line)

    # Print 5-period breakdown for F61
    if "F61_VIX_RSI20" in fa:
        f_arr = fa["F61_VIX_RSI20"]
        ics = []
        for day in range(oos, Te - 1):
            valid = ~np.isnan(c[day]) & (c[day] > 0) & ~np.isnan(fwd1[day]) & ~np.isnan(f_arr[day])
            vi = np.where(valid)[0]
            if len(vi) < 20:
                continue
            rho, _ = spearmanr(f_arr[day, vi], fwd1[day, vi])
            if not np.isnan(rho):
                ics.append(rho)
        if len(ics) >= 50:
            ics = np.array(ics)
            n_per = len(ics) // 5
            period_labels = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-26"]
            results_text.append("\n  F61 period breakdown:")
            for i in range(5):
                chunk = ics[i * n_per:(i + 1) * n_per]
                lbl = period_labels[i] if i < len(period_labels) else f"P{i+1}"
                results_text.append(f"    {lbl}: IC={np.mean(chunk):+.4f}  IC>0={np.mean(chunk>0)*100:.0f}%")
    results_text.append("")

    # Save
    full_text = "\n".join(results_text)
    results_path = ROOT / "market_data" / "scores" / "v10_2_diagnostics.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("V10.2 ENGINE DIAGNOSTICS — EXPANDED UNIVERSE\n" + "="*80 + "\n\n" + full_text)
    print(f"\nResults saved to {results_path}")

    return full_text


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # Step 1: Build expanded parquet if needed
    if not V4_PARQUET.exists():
        print("Building expanded parquet...")
        n_tickers = _build_expanded_parquet()
    else:
        import pandas as pd
        pq = pd.read_parquet(V4_PARQUET, columns=[]).reset_index() if False else None

    # Step 2: Compute and cache factors for both universes
    print("\n[EXPANDED UNIVERSE]")
    data_exp = compute_and_cache_factors(force=True, use_v3=False)
    Ne = data_exp['N']
    n_factors = len(data_exp['factor_arrays'])

    print("\n[V3 UNIVERSE]")
    data_v3 = compute_and_cache_factors(force=True, use_v3=True)

    msg = (f"V10.2 engine ready. {Ne} tickers. {n_factors} factors cached. "
           f"Workers can start.")
    print(f"\n{msg}")

    # Step 3: Run diagnostics
    diag_text = run_diagnostics()

    # Step 4: Post to Slack
    try:
        from slack_sdk import WebClient
        client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
        channel = "C0AK6PHTGD8"
        thread_ts = "1773765570.266679"

        chunks = []
        lines = diag_text.split("\n")
        current = ""
        for line in lines:
            if len(current) + len(line) + 1 > 2900:
                chunks.append(current)
                current = line
            else:
                current += "\n" + line if current else line
        if current:
            chunks.append(current)

        for i, chunk in enumerate(chunks):
            label = f"V10.2 Diagnostics ({i+1}/{len(chunks)})" if len(chunks) > 1 else "V10.2 Diagnostics"
            client.chat_postMessage(
                channel=channel, thread_ts=thread_ts,
                text=f":telescope: *{label}*\n```\n{chunk}\n```",
                username="V10.2 Engine", icon_emoji=":telescope:",
            )
        print("  Posted diagnostics to Slack.")
    except Exception as e:
        print(f"  Slack error: {e}")
