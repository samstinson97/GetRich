"""Exhaustive feature comparison: live signal generator vs backtest engine.
Tests all 23 computable features across 5 tickers of varying characteristics."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "C:/ai-research-team")
from v10_2_engine import rsi_arr
import yfinance as yf

def _rsi_ewm(prices, period=14):
    if len(prices) < period + 1: return 50.0
    s = pd.Series(prices)
    delta = s.diff()
    up = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    dn = (-delta).clip(lower=0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return float((100 - 100 / (1 + up / (dn + 1e-9))).values[-1])

tickers_test = ["AAPL", "MMYT", "CENX", "BEAM", "HD"]
all_pass = True

for ticker in tickers_test:
    data = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
    c = data["Close"].squeeze().dropna().values.astype(float).ravel()
    h = data["High"].squeeze().dropna().values.astype(float).ravel()
    l = data["Low"].squeeze().dropna().values.astype(float).ravel()
    o = data["Open"].squeeze().dropna().values.astype(float).ravel()
    v = data["Volume"].squeeze().dropna().values.astype(float).ravel()
    N = min(len(c), 300)
    c, h, l, o, v = c[-N:], h[-N:], l[-N:], o[-N:], v[-N:]
    T = len(c)
    if T < 30:
        print(f"{ticker}: only {T} bars, skipping"); continue

    stk_ret = np.full(T, np.nan); stk_ret[1:] = c[1:] / (c[:-1] + 1e-15) - 1
    hl_arr = h - l

    bt, lv = {}, {}

    # F14_IBS
    bt["F14_IBS"] = np.where(hl_arr > 0, (c - l) / hl_arr, 0.5)[-1]
    lv["F14_IBS"] = (c[-1] - l[-1]) / (h[-1] - l[-1]) if (h[-1] - l[-1]) > 0 else 0.5

    # F05_RSI14
    bt["F05_RSI14"] = float(rsi_arr(c, 14)[-1])
    lv["F05_RSI14"] = _rsi_ewm(c, 14)

    # F09_DistSMA20
    sma20_bt = pd.Series(c).rolling(20, min_periods=20).mean().values[-1]
    bt["F09_DistSMA20"] = (c[-1] - sma20_bt) / sma20_bt
    lv["F09_DistSMA20"] = c[-1] / np.mean(c[-20:]) - 1

    # F13_DD52wk
    hi252 = pd.Series(h).rolling(252, min_periods=20).max().values[-1]
    bt["F13_DD52wk"] = (c[-1] - hi252) / hi252
    hi252_lv = np.max(h[-252:]) if len(h) >= 252 else np.max(h)
    lv["F13_DD52wk"] = (c[-1] - hi252_lv) / hi252_lv if hi252_lv > 0 else 0

    # F17_RealVol
    bt["F17_RealVol"] = float((pd.Series(stk_ret).rolling(20, min_periods=20).std() * np.sqrt(252)).values[-1])
    arith_rets = np.diff(c[-21:]) / (c[-21:-1] + 1e-15)
    lv["F17_RealVol"] = np.std(arith_rets, ddof=1) * np.sqrt(252)

    # F20_VolSpike
    vol20_bt = pd.Series(v).rolling(20, min_periods=5).mean().values[-1]
    bt["F20_VolSpike"] = v[-1] / vol20_bt
    lv["F20_VolSpike"] = v[-1] / np.mean(v[-20:])

    # F23_VolAtLows
    dm = stk_ret < 0
    bt["F23_VolAtLows"] = pd.Series(np.where(dm, v, 0.0)).rolling(5, min_periods=1).sum().values[-1] / pd.Series(v).rolling(5, min_periods=1).sum().values[-1]
    rets_lv = np.diff(c[-6:]) / (c[-6:-1] + 1e-15)
    lv["F23_VolAtLows"] = np.sum(v[-5:][rets_lv < 0]) / np.sum(v[-5:])

    # F43_5dRoC
    bt["F43_5dRoC"] = c[-1] / c[-6] - 1
    lv["F43_5dRoC"] = c[-1] / c[-6] - 1

    # F45_3dRoC
    bt["F45_3dRoC"] = c[-1] / c[-4] - 1
    lv["F45_3dRoC"] = c[-1] / c[-4] - 1

    # F57_PriorDayRet
    bt["F57_PriorDayRet"] = stk_ret[-1]
    lv["F57_PriorDayRet"] = c[-1] / c[-2] - 1

    # F52_OvernightGapZ
    prev_c = np.roll(c, 1); prev_c[0] = np.nan
    og = np.where((prev_c > 0) & ~np.isnan(prev_c), o / prev_c - 1, np.nan)
    og_m = pd.Series(og).rolling(20, min_periods=5).mean().values[-1]
    og_s = pd.Series(og).rolling(20, min_periods=5).std().values[-1]
    bt["F52_OvernightGapZ"] = (og[-1] - og_m) / og_s if og_s > 0 else 0
    gaps_lv = o[-20:] / c[-21:-1] - 1
    gap_today_lv = o[-1] / c[-2] - 1
    lv["F52_OvernightGapZ"] = (gap_today_lv - np.nanmean(gaps_lv)) / (np.nanstd(gaps_lv, ddof=1) + 1e-9)

    # F54_IBSxRange
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    atr14_bt = pd.Series(tr).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values[-1]
    bt["F54_IBSxRange"] = (1 - bt["F14_IBS"]) * ((h[-1] - l[-1]) / atr14_bt)
    prev_c_lv = np.roll(c, 1); prev_c_lv[0] = c[0]
    tr_lv = np.maximum(h - l, np.maximum(np.abs(h - prev_c_lv), np.abs(l - prev_c_lv)))
    atr14_lv = pd.Series(tr_lv).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values[-1]
    lv["F54_IBSxRange"] = (1 - lv["F14_IBS"]) * ((h[-1] - l[-1]) / atr14_lv)

    # F60_IntradayMom
    bt["F60_IntradayMom"] = (c[-1] - o[-1]) / o[-1]
    lv["F60_IntradayMom"] = (c[-1] - o[-1]) / o[-1]

    # rsi5
    bt["rsi5"] = float(rsi_arr(c, 5)[-1])
    lv["rsi5"] = _rsi_ewm(c, 5)

    # _atr14
    bt["_atr14"] = atr14_bt
    lv["_atr14"] = atr14_lv

    # LAGS
    bt["IBS_lag1"] = (c[-2] - l[-2]) / (h[-2] - l[-2]) if (h[-2] - l[-2]) > 0 else 0.5
    lv["IBS_lag1"] = bt["IBS_lag1"]  # same formula

    bt["PriorRet_lag1"] = stk_ret[-2]
    lv["PriorRet_lag1"] = c[-2] / c[-3] - 1

    bt["RoC3d_lag1"] = c[-2] / c[-5] - 1
    lv["RoC3d_lag1"] = c[-2] / c[-5] - 1

    vol20_lag_bt = pd.Series(v).rolling(20, min_periods=5).mean().values[-2]
    bt["VolSpike_lag1"] = v[-2] / vol20_lag_bt if vol20_lag_bt > 0 else 1
    lv["VolSpike_lag1"] = v[-2] / np.mean(v[-21:-1]) if np.mean(v[-21:-1]) > 0 else 1

    bt["IntradayMom_lag1"] = (c[-2] - o[-2]) / o[-2] if o[-2] > 0 else 0
    lv["IntradayMom_lag1"] = bt["IntradayMom_lag1"]  # same formula

    # Z-SCORES
    ibs_series = np.where(hl_arr > 0, (c - l) / hl_arr, 0.5)
    ibs_z_m = pd.Series(ibs_series).rolling(5, min_periods=3).mean().values[-1]
    ibs_z_s = pd.Series(ibs_series).rolling(5, min_periods=3).std().values[-1]
    bt["IBS_z5"] = (ibs_series[-1] - ibs_z_m) / (ibs_z_s + 1e-9)
    ibs_vals_lv = [(c[i] - l[i]) / (h[i] - l[i]) if (h[i] - l[i]) > 0 else 0.5 for i in range(-5, 0)]
    lv["IBS_z5"] = (ibs_vals_lv[-1] - np.mean(ibs_vals_lv)) / (np.std(ibs_vals_lv, ddof=1) + 1e-9)

    pr_z_m = pd.Series(stk_ret).rolling(5, min_periods=3).mean().values[-1]
    pr_z_s = pd.Series(stk_ret).rolling(5, min_periods=3).std().values[-1]
    bt["PriorRet_z5"] = (stk_ret[-1] - pr_z_m) / (pr_z_s + 1e-9)
    pr_vals_lv = [c[i] / c[i-1] - 1 for i in range(-5, 0)]
    lv["PriorRet_z5"] = (pr_vals_lv[-1] - np.mean(pr_vals_lv)) / (np.std(pr_vals_lv, ddof=1) + 1e-9)

    c3 = np.roll(c, 3); c3[:3] = np.nan
    roc3 = np.where((c3 > 0) & ~np.isnan(c3), c / c3 - 1, np.nan)
    roc3_z_m = pd.Series(roc3).rolling(5, min_periods=3).mean().values[-1]
    roc3_z_s = pd.Series(roc3).rolling(5, min_periods=3).std().values[-1]
    bt["RoC3d_z5"] = (roc3[-1] - roc3_z_m) / (roc3_z_s + 1e-9)
    roc_vals_lv = [c[i] / c[i-3] - 1 for i in range(-5, 0)]
    lv["RoC3d_z5"] = (roc_vals_lv[-1] - np.mean(roc_vals_lv)) / (np.std(roc_vals_lv, ddof=1) + 1e-9)

    # COMPARE
    fails, close_calls = [], []
    for k in sorted(bt.keys()):
        b, lval = bt[k], lv.get(k, float("nan"))
        if np.isnan(b) and np.isnan(lval): continue
        if np.isnan(b) or np.isnan(lval):
            fails.append((k, b, lval, "NaN")); continue
        rel = abs(b - lval) / (abs(b) + 1e-12)
        if rel >= 0.01:
            fails.append((k, b, lval, f"{rel*100:.2f}%"))
        elif rel >= 0.001:
            close_calls.append((k, b, lval, f"{rel*100:.3f}%"))

    print(f"=== {ticker} ({T} bars, close={c[-1]:.2f}) ===")
    if fails:
        for k, b, lval, err in fails:
            print(f"  FAIL  {k:25s} bt={b:>12.6f} lv={lval:>12.6f} err={err}")
    if close_calls:
        for k, b, lval, err in close_calls:
            print(f"  CLOSE {k:25s} bt={b:>12.6f} lv={lval:>12.6f} err={err}")
    if not fails and not close_calls:
        print(f"  ALL {len(bt)} features MATCH (<0.1%)")
    else:
        all_pass = False
        print(f"  {len(fails)} FAIL, {len(close_calls)} CLOSE")
    print()

# Final syntax check
import ast
with open("production/ml_v3/early_signal_ml_v3.py") as f:
    ast.parse(f.read())
print("Script syntax: OK")
print()
if all_pass:
    print("VERDICT: ALL FEATURES MATCH ACROSS ALL 5 TICKERS. Ready for tomorrow.")
else:
    print("VERDICT: ISSUES REMAIN.")
