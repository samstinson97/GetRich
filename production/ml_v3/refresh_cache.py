#!/usr/bin/env python3
"""
Daily Factor Cache Refresh
Downloads latest OHLCV + VIX from yfinance, appends to parquet/CSV,
rebuilds factor cache, expands features, rescores with seed 6, and
updates scores_ml.npy. Then runs backtest replay for the dashboard.

Add to run_live.bat BEFORE backtest_replay.py.
Runtime: ~5-8 minutes (download + rebuild + score).

Usage:
    python refresh_cache.py           # Incremental refresh (new days only)
    python refresh_cache.py --force   # Full rebuild from scratch
"""
import sys, time, logging, pickle, json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("c:/ai-research-team")
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("refresh_cache")

PARQUET_PATH = ROOT / "market_data" / "precomputed_v4_expanded.parquet"
VIX_PATH = ROOT / "market_data" / "macro" / "vix_daily.csv"
FACTOR_CACHE = ROOT / "factor_tournament" / "v10_2_factors.pkl"
FEATURES_PKL = ROOT / "factor_tournament" / "v10_ml_v3_features.pkl"
SCORES_PATH = ROOT / "v10_ml_v3" / "seed_6" / "scores_ml.npy"
METADATA_PATH = ROOT / "factor_tournament" / "v10_ml_v3_metadata.json"


def get_parquet_last_date():
    """Get the last date in the parquet file."""
    df = pd.read_parquet(PARQUET_PATH, columns=["close"])
    return df.index.get_level_values("date").max()


def download_new_ohlcv(tickers, start_date):
    """Download OHLCV from yfinance for dates after start_date."""
    import yfinance as yf

    start = (start_date - timedelta(days=5)).strftime("%Y-%m-%d")
    end = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    log.info(f"Downloading OHLCV from {start} to {end} for {len(tickers)} tickers...")

    all_data = []
    batch_size = 100
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            raw = yf.download(batch, start=start, end=end,
                              auto_adjust=True, threads=True, progress=False,
                              group_by="ticker")
            if len(batch) == 1:
                tk = batch[0]
                if "Close" in raw.columns and len(raw) > 0:
                    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
                    df.columns = ["open", "high", "low", "close", "volume"]
                    df["ticker"] = tk
                    df.index.name = "date"
                    all_data.append(df.reset_index())
            else:
                for tk in batch:
                    try:
                        df = raw[tk].dropna(subset=["Close"])
                        if len(df) == 0:
                            continue
                        df2 = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                        df2.columns = ["open", "high", "low", "close", "volume"]
                        df2["ticker"] = tk
                        df2.index.name = "date"
                        all_data.append(df2.reset_index())
                    except Exception:
                        pass
        except Exception as e:
            log.warning(f"Batch {i} failed: {e}")

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    # Filter to only new dates
    result["date"] = pd.to_datetime(result["date"])
    result = result[result["date"] > pd.Timestamp(start_date)]
    log.info(f"Downloaded {len(result)} new rows for "
             f"{result['ticker'].nunique()} tickers, "
             f"{result['date'].nunique()} new days")
    return result


def download_new_vix(start_date):
    """Download VIX data from yfinance."""
    import yfinance as yf

    start = start_date.strftime("%Y-%m-%d")
    end = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    log.info(f"Downloading VIX from {start} to {end}...")

    vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
    if len(vix) == 0:
        return pd.DataFrame()

    vix_df = pd.DataFrame({
        "Date": vix.index.strftime("%Y-%m-%d"),
        "VIX_Close": vix["Close"].values.flatten(),
        "VIX_High": vix["High"].values.flatten(),
        "VIX_Low": vix["Low"].values.flatten(),
        "VIX_Open": vix["Open"].values.flatten(),
        "Volume": 0,
        "VIX_SMA_20": np.nan,
        "VIX_Regime": "unknown",
    })
    # Filter to new dates only
    vix_df = vix_df[vix_df["Date"] > start_date.strftime("%Y-%m-%d")]
    log.info(f"Downloaded {len(vix_df)} new VIX rows")
    return vix_df


def append_to_parquet(new_data):
    """Append new OHLCV rows to the parquet file."""
    if new_data.empty:
        log.info("No new data to append to parquet")
        return

    log.info(f"Loading existing parquet...")
    existing = pd.read_parquet(PARQUET_PATH)

    # Compute derived columns for new data
    new_data = new_data.copy()
    new_data["rsi4"] = np.nan  # Will be recomputed by factor cache
    new_data["relative_volume"] = np.nan
    new_data["atr14"] = np.nan

    # Set multi-index
    new_data["date"] = pd.to_datetime(new_data["date"])
    new_indexed = new_data.set_index(["date", "ticker"])
    new_indexed = new_indexed[existing.columns]  # match column order

    # Remove any dates that already exist
    existing_dates = set(existing.index.get_level_values("date").unique())
    new_dates = set(new_indexed.index.get_level_values("date").unique())
    truly_new = new_dates - existing_dates

    if not truly_new:
        log.info("All dates already in parquet, skipping append")
        return

    new_indexed = new_indexed[new_indexed.index.get_level_values("date").isin(truly_new)]
    log.info(f"Appending {len(new_indexed)} rows ({len(truly_new)} new dates)")

    combined = pd.concat([existing, new_indexed])
    combined.sort_index(inplace=True)
    combined.to_parquet(PARQUET_PATH)
    log.info(f"Parquet updated: {combined.index.get_level_values('date').max()}")


def append_to_vix(new_vix):
    """Append new VIX rows to the CSV."""
    if new_vix.empty:
        return

    existing = pd.read_csv(VIX_PATH)
    existing_dates = set(existing["Date"])
    new_rows = new_vix[~new_vix["Date"].isin(existing_dates)]

    if new_rows.empty:
        log.info("All VIX dates already in CSV")
        return

    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined.sort_values("Date", inplace=True)

    # Recompute SMA_20
    combined["VIX_SMA_20"] = combined["VIX_Close"].rolling(20).mean()
    combined["VIX_Regime"] = combined["VIX_Close"].apply(
        lambda x: "low_vol" if x < 20 else ("mid_vol" if x < 30 else "high_vol"))

    combined.to_csv(VIX_PATH, index=False)
    log.info(f"VIX CSV updated: {combined['Date'].iloc[-1]} ({len(new_rows)} new rows)")


def rebuild_factor_cache():
    """Rebuild v10_2_factors.pkl from updated parquet.
    Uses v10_2_engine_daily.py (copy with dynamic end date) to avoid
    modifying the production engine."""
    # Import from the daily copy which has the dynamic end date
    PROD_DIR = str(Path(__file__).parent)
    if PROD_DIR not in sys.path:
        sys.path.insert(0, PROD_DIR)
    from v10_2_engine_daily import compute_and_cache_factors

    log.info("Rebuilding factor cache (force=True, dynamic end date)...")
    t0 = time.time()
    data = compute_and_cache_factors(force=True)
    log.info(f"Factor cache rebuilt in {time.time()-t0:.1f}s: "
             f"T={data['T']}, N={data['N']}, "
             f"dates {data['all_dates'][0].date()} to {data['all_dates'][-1].date()}")
    return data


def expand_features(data):
    """Expand 22 raw factors to 77 features (RANK, ZSCORE, INT, SQ, RATIO).
    Mirrors v10_ml_v3_feature_expansion.py exactly."""
    from scipy.stats import rankdata

    fa = data["factor_arrays"]
    T, N = data["T"], data["N"]
    log.info(f"Expanding features: T={T}, N={N}")

    expanded = {}

    # Copy 22 raw factors
    from v10_2_engine_daily import FACTOR_NAMES
    for f in FACTOR_NAMES:
        expanded[f] = fa[f]

    # Interactions
    interactions = [
        ('F14_IBS', 'F17_RealVol'), ('F14_IBS', 'F54_IBSxRange'),
        ('F57_PriorDayRet', 'F60_IntradayMom'), ('F14_IBS', 'F60_IntradayMom'),
        ('F17_RealVol', 'F57_PriorDayRet'), ('F14_IBS', 'F45_3dRoC'),
        ('F20_VolSpike', 'F14_IBS'), ('F23_VolAtLows', 'F17_RealVol'),
        ('F37_IdioReturn5d', 'F33_SectorRelRSI'), ('F09_DistSMA20', 'F14_IBS'),
    ]
    for a, b in interactions:
        expanded[f"INT_{a}_{b}"] = fa[a] * fa[b]

    # Squared
    sq_factors = ['F14_IBS', 'F57_PriorDayRet', 'F60_IntradayMom', 'F54_IBSxRange',
                  'F17_RealVol', 'F45_3dRoC', 'F37_IdioReturn5d', 'F09_DistSMA20',
                  'F20_VolSpike', 'F23_VolAtLows']
    for f in sq_factors:
        expanded[f"SQ_{f}"] = fa[f] ** 2

    # Cross-sectional ranks
    def cs_rank(arr):
        result = np.full((T, N), np.nan)
        for day in range(T):
            row = arr[day]
            valid = ~np.isnan(row)
            n_valid = valid.sum()
            if n_valid > 1:
                result[day, valid] = rankdata(row[valid]) / n_valid
        return result

    for f in FACTOR_NAMES:
        expanded[f"RANK_{f}"] = cs_rank(fa[f])

    # Cross-sectional z-scores
    zs_factors = ['F14_IBS', 'F57_PriorDayRet', 'F60_IntradayMom', 'F54_IBSxRange',
                  'F17_RealVol', 'F45_3dRoC', 'F37_IdioReturn5d', 'F20_VolSpike',
                  'F23_VolAtLows', 'F33_SectorRelRSI']
    for f in zs_factors:
        result = np.full((T, N), np.nan)
        for day in range(T):
            row = fa[f][day]
            valid = ~np.isnan(row)
            if valid.sum() > 2:
                mu = np.nanmean(row)
                std = np.nanstd(row)
                if std > 1e-9:
                    result[day, valid] = (row[valid] - mu) / std
        expanded[f"ZSCORE_{f}"] = result

    # Ratios
    ratios = [('F14_IBS', 'F17_RealVol'), ('F57_PriorDayRet', 'F17_RealVol'),
              ('F45_3dRoC', 'F17_RealVol')]
    for a, b in ratios:
        denom = fa[b].copy()
        denom[np.abs(denom) < 1e-9] = np.nan
        expanded[f"RATIO_{a}_over_{b}"] = fa[a] / denom

    log.info(f"Expanded to {len(expanded)} features")

    # Save
    with open(FEATURES_PKL, "wb") as f:
        pickle.dump(expanded, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"Saved expanded features to {FEATURES_PKL}")

    return expanded


def rescore_with_seed6(expanded, data):
    """Extend scores_ml.npy for new days only.

    The original scores (from in-memory training) are the ground truth.
    Saved .txt models produce slightly different predictions (LightGBM
    serialization loses some state), so we NEVER rescore historical days.
    We only score NEW days beyond the original scores_ml.npy range.
    """
    import lightgbm as lgb

    T, N = data["T"], data["N"]
    close = data["close"]
    all_dates = data["all_dates"]

    # Load original scores from backup (ground truth from training run)
    BACKUP_SCORES = ROOT / "BEST_MODELS" / "ML_v3_seed6" / "scores_ml.npy"
    if BACKUP_SCORES.exists():
        orig_scores = np.load(BACKUP_SCORES)
        T_orig = orig_scores.shape[0]
        log.info(f"Loaded original scores: {orig_scores.shape} (ground truth)")
    else:
        # Fall back to current scores file
        orig_scores = np.load(SCORES_PATH)
        T_orig = orig_scores.shape[0]
        log.info(f"Using current scores as base: {orig_scores.shape}")

    if T <= T_orig:
        log.info(f"No new days to score (T={T} <= T_orig={T_orig})")
        # Just copy original to the right path
        np.save(SCORES_PATH, orig_scores[:T])
        return orig_scores[:T]

    n_new = T - T_orig
    log.info(f"Extending scores: {T_orig} -> {T} ({n_new} new days)")

    # Build extended scores: original + new
    scores = np.full((T, N), np.nan, dtype=np.float32)
    scores[:T_orig] = orig_scores

    # Load models for new days
    model_dir = ROOT / "v10_ml_v3" / "seed_6"
    models_by_year = {}
    for yr in range(2006, 2028):
        mp = model_dir / f"lgb_model_{yr}.txt"
        if mp.exists():
            models_by_year[yr] = lgb.Booster(model_file=str(mp))
    log.info(f"Loaded {len(models_by_year)} seed 6 models")

    # Use all 77 features in metadata order
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]
    log.info(f"Using {len(feature_names)} features from metadata")

    feat_arrays = []
    for fn in feature_names:
        if fn in expanded:
            feat_arrays.append(expanded[fn])
        else:
            log.warning(f"Feature {fn} not in expanded set, using zeros")
            feat_arrays.append(np.zeros((T, N), dtype=np.float32))

    # Score ONLY new days
    t0 = time.time()
    for day in range(T_orig, T):
        yr = all_dates[day].year if day < len(all_dates) else 2026
        model = models_by_year.get(yr)
        if model is None:
            for offset in range(1, 5):
                model = models_by_year.get(yr - offset) or models_by_year.get(yr + offset)
                if model:
                    break
        if model is None:
            continue

        valid = ~np.isnan(close[day]) & (close[day] > 0)
        vi = np.where(valid)[0]
        if len(vi) < 5:
            continue

        feats = np.column_stack([fa[day, vi] for fa in feat_arrays])
        feats = np.nan_to_num(feats, nan=0.0)
        preds = model.predict(feats, predict_disable_shape_check=True)

        pmin, pmax = preds.min(), preds.max()
        if pmax > pmin:
            scores[day, vi] = (preds - pmin) / (pmax - pmin)
        else:
            scores[day, vi] = 0.5

        dt = all_dates[day].strftime("%Y-%m-%d") if day < len(all_dates) else f"day_{day}"
        n_valid = len(vi)
        log.info(f"  Scored day {day} ({dt}): {n_valid} stocks")

    elapsed = time.time() - t0
    log.info(f"New days scored in {elapsed:.1f}s")

    # Save
    np.save(SCORES_PATH, scores)
    log.info(f"Saved scores_ml.npy: {scores.shape}")
    return scores


def main():
    force = "--force" in sys.argv
    t_total = time.time()

    # Step 1: Check what's new
    last_date = get_parquet_last_date()
    log.info(f"Parquet last date: {last_date.date()}")

    today = datetime.now().date()
    if last_date.date() >= today and not force:
        log.info("Parquet is current, nothing to refresh")
        return

    # Step 2: Download new data
    # Get ticker list from existing parquet
    existing = pd.read_parquet(PARQUET_PATH, columns=["close"])
    tickers = list(existing.index.get_level_values("ticker").unique())

    new_ohlcv = download_new_ohlcv(tickers, last_date)
    new_vix = download_new_vix(last_date)

    # Step 3: Append to parquet + VIX
    append_to_parquet(new_ohlcv)
    append_to_vix(new_vix)

    # Step 4: Rebuild factor cache
    data = rebuild_factor_cache()

    # Step 5: Expand features
    expanded = expand_features(data)

    # Step 6: Rescore with seed 6
    rescore_with_seed6(expanded, data)

    elapsed = time.time() - t_total
    log.info(f"Full refresh complete in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
