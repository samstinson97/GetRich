#!/usr/bin/env python3
"""
SHAP Alpha Decay Monitor + Rolling Sharpe Tracker
Computes daily SHAP feature importance and detects signal decay.

Usage:
    # Standalone test with seed 6 model
    python shap_monitor.py --test

    # Called from early_signal_ml_v3.py after scoring
    from shap_monitor import run_shap_monitor, compute_rolling_sharpe
"""
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("shap_monitor")

PROD = Path(__file__).parent          # production/ml_v4/
_stop_mode = os.environ.get("V4_STOP_MODE", "moc")
SHAP_HISTORY_FILE = PROD / f"shap_history_{_stop_mode}.json"
DASHBOARD_DATA_FILE = PROD.parent / f"dashboard_data_v4_{_stop_mode}.json"

MAX_HISTORY_DAYS = 90  # ~4.5 months of trading days


# =====================================================================
# SHAP COMPUTATION
# =====================================================================
def compute_daily_shap(model, X_scored, feature_names):
    """
    Compute mean(|SHAP|) per feature for today's scored universe.
    Returns a pd.Series of SHAP importance percentages (sums to 100).
    """
    import shap

    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scored)

    # Mean absolute SHAP value per feature
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_names,
    )

    # Normalize to percentages
    total = mean_abs_shap.sum()
    if total > 0:
        shap_pct = (mean_abs_shap / total * 100).round(2)
    else:
        shap_pct = mean_abs_shap

    elapsed = time.time() - t0
    log.info(f"SHAP computed in {elapsed:.1f}s for {X_scored.shape[0]} stocks, "
             f"{X_scored.shape[1]} features")
    return shap_pct


# =====================================================================
# SHAP HISTORY
# =====================================================================
def load_shap_history():
    """Load rolling SHAP history from JSON."""
    if SHAP_HISTORY_FILE.exists():
        with open(SHAP_HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_shap_history(history):
    """Save rolling SHAP history to JSON."""
    with open(SHAP_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def update_shap_history(shap_pct, history=None):
    """
    Append today's SHAP importance to the rolling history.
    Keep last 90 trading days.
    """
    if history is None:
        history = load_shap_history()

    importance_dict = {k: round(float(v), 2) for k, v in shap_pct.items()}
    top_5 = dict(shap_pct.nlargest(5).items())
    top_5 = {k: round(float(v), 2) for k, v in top_5.items()}

    # Herfindahl-Hirschman Index (concentration measure)
    hhi = float(((shap_pct / 100) ** 2).sum())

    today_record = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "shap_importance": importance_dict,
        "top_5": top_5,
        "hhi": round(hhi, 4),
    }

    # Replace today's entry if it already exists (re-run)
    history = [h for h in history if h["date"] != today_record["date"]]
    history.append(today_record)

    # Keep last N trading days
    if len(history) > MAX_HISTORY_DAYS:
        history = history[-MAX_HISTORY_DAYS:]

    save_shap_history(history)
    return history


# =====================================================================
# ALERT GENERATION
# =====================================================================
def check_shap_alerts(history):
    """
    Generate alerts based on SHAP importance shifts.
    Returns list of (severity, message) tuples.
    """
    alerts = []

    if len(history) < 2:
        alerts.append(("info", "SHAP monitor initialized. Need 5+ days for trend alerts."))
        return alerts

    today = history[-1]["shap_importance"]

    # Week-ago comparison (5 trading days back)
    week_ago = history[-5]["shap_importance"] if len(history) >= 5 else None

    # Month-ago comparison (20 trading days back)
    month_ago = history[-20]["shap_importance"] if len(history) >= 20 else None

    # Alert 1: RANK_RealVol (or RealVol_csrank) SHAP drops >30% week-over-week
    realvol_keys = ["RANK_F17_RealVol", "RealVol_csrank", "F17_RealVol"]
    if week_ago:
        for rk in realvol_keys:
            rv_today = today.get(rk, 0)
            rv_week = week_ago.get(rk, 0)
            if rv_week > 2:  # only alert if it was meaningful
                drop_pct = (rv_week - rv_today) / rv_week * 100
                if drop_pct > 30:
                    alerts.append(("critical",
                        f"{rk} SHAP dropped {drop_pct:.0f}% week-over-week "
                        f"({rv_week:.1f}% -> {rv_today:.1f}%). "
                        f"Investigate possible signal crowding or regime shift."))
                break  # only alert on the first matching key

    # Alert 2: Any single feature rises above 25%
    for feature, importance in today.items():
        if importance > 25:
            alerts.append(("warning",
                f"{feature} SHAP at {importance:.1f}% (above 25% threshold). "
                f"Check for overfitting to recent noise."))

    # Alert 3: Top-5 feature set changes by 2+ features vs last month
    if month_ago:
        top5_today = set(sorted(today, key=lambda k: today[k], reverse=True)[:5])
        top5_month = set(sorted(month_ago, key=lambda k: month_ago[k], reverse=True)[:5])
        changed = len(top5_today - top5_month)
        if changed >= 2:
            new_features = top5_today - top5_month
            dropped_features = top5_month - top5_today
            alerts.append(("warning",
                f"Top-5 features changed by {changed} vs last month. "
                f"New: {new_features}. Dropped: {dropped_features}. "
                f"Possible regime shift."))

    # Alert 4: HHI concentration rising above threshold
    hhi_today = history[-1]["hhi"]
    if hhi_today > 0.15:
        alerts.append(("warning",
            f"Feature concentration HHI at {hhi_today:.3f} (above 0.15 threshold). "
            f"Model becoming dominated by fewer features."))

    # Alert 5: HHI increasing trend (3 consecutive rises)
    if len(history) >= 4:
        recent_hhi = [h["hhi"] for h in history[-4:]]
        if all(recent_hhi[i] < recent_hhi[i+1] for i in range(3)):
            alerts.append(("warning",
                f"HHI rising 3 days straight: {recent_hhi[-3]:.3f} -> "
                f"{recent_hhi[-2]:.3f} -> {recent_hhi[-1]:.3f}. "
                f"Feature concentration trending up."))

    return alerts


# =====================================================================
# ROLLING SHARPE MONITOR
# =====================================================================
def compute_rolling_sharpe(equity_curve, window=60):
    """
    Compute rolling Sharpe over the last `window` trading days.
    Uses daily equity returns from the equity curve.
    Returns (sharpe_value, status_color) or (None, None) if insufficient data.
    """
    if not equity_curve or len(equity_curve) < 2:
        return None, None

    # Extract actual equity values
    equities = [e.get("actual", e.get("equity", 0)) for e in equity_curve]
    equities = [e for e in equities if e and e > 0]

    if len(equities) < 2:
        return None, None

    # Use available window (up to requested)
    use_window = min(len(equities), window)
    recent = equities[-use_window:]

    # Daily returns
    daily_returns = []
    for i in range(1, len(recent)):
        if recent[i-1] > 0:
            daily_returns.append(recent[i] / recent[i-1] - 1)

    if not daily_returns:
        return None, None

    mean_ret = np.mean(daily_returns)
    std_ret = np.std(daily_returns)

    if std_ret == 0:
        return 0.0, "neutral"

    # Annualize
    sharpe = (mean_ret / std_ret) * np.sqrt(252)
    sharpe = round(sharpe, 2)

    # Color coding
    if sharpe >= 1.5:
        color = "ok"     # green - healthy
    elif sharpe >= 1.0:
        color = "warn"   # yellow - watch
    else:
        color = "bad"    # red - critical

    return sharpe, color


def check_sharpe_alerts(sharpe_value):
    """Generate alerts based on rolling Sharpe."""
    alerts = []
    if sharpe_value is None:
        return alerts
    if sharpe_value < 0.5:
        alerts.append(("critical",
            f"Rolling 60-day Sharpe at {sharpe_value:.2f}. "
            f"Model edge may have decayed. Consider pausing live trading."))
    elif sharpe_value < 1.0:
        alerts.append(("critical",
            f"Rolling 60-day Sharpe at {sharpe_value:.2f} (below 1.0). "
            f"Consider pausing live trading until signal recovers."))
    return alerts


# =====================================================================
# DASHBOARD DATA UPDATE
# =====================================================================
def update_dashboard_data(shap_pct, history, shap_alerts, sharpe_value, sharpe_color):
    """Add SHAP + rolling Sharpe data to dashboard_data.json."""
    # Load existing dashboard data
    if DASHBOARD_DATA_FILE.exists():
        with open(DASHBOARD_DATA_FILE, "r") as f:
            data = json.load(f)
    else:
        data = {}

    # Build top 10 for display
    top_10 = dict(shap_pct.nlargest(10).items())
    top_10 = {k: round(float(v), 2) for k, v in top_10.items()}

    # Build history for chart (last 90 days, top_5 + hhi per day)
    chart_history = []
    for h in history[-90:]:
        chart_history.append({
            "date": h["date"],
            "top_5": h["top_5"],
            "hhi": h["hhi"],
        })

    # Build alert list
    alert_records = []
    now_str = datetime.now().strftime("%m/%d %H:%M")
    for severity, message in shap_alerts:
        alert_records.append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": now_str,
            "severity": severity,
            "message": message,
        })

    data["shap"] = {
        "latest": {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "top_10": top_10,
            "hhi": round(float(history[-1]["hhi"]), 4) if history else 0,
        },
        "history": chart_history,
        "alerts": alert_records,
    }

    data["rolling_sharpe"] = {
        "value": sharpe_value,
        "color": sharpe_color,
        "window": 60,
        "as_of": datetime.now().strftime("%Y-%m-%d"),
    }

    # Merge SHAP alerts into main alerts list
    existing_alerts = data.get("alerts", [])
    for severity, message in shap_alerts:
        existing_alerts.insert(0, {
            "time": now_str,
            "level": severity,
            "msg": f"[SHAP] {message}",
        })

    # Merge Sharpe alerts
    sharpe_alerts = check_sharpe_alerts(sharpe_value)
    for severity, message in sharpe_alerts:
        existing_alerts.insert(0, {
            "time": now_str,
            "level": severity,
            "msg": f"[SHARPE] {message}",
        })

    data["alerts"] = existing_alerts[:50]  # keep last 50

    with open(DASHBOARD_DATA_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)

    log.info(f"Dashboard data updated: {len(top_10)} SHAP features, "
             f"HHI={history[-1]['hhi']:.4f}, Sharpe={sharpe_value}")


# =====================================================================
# MAIN ENTRY POINT (called from signal generator)
# =====================================================================
def run_shap_monitor(model, X_scored, feature_names, equity_curve=None):
    """
    Full SHAP monitoring pipeline. Call after ML scoring.
    Returns (shap_alerts, sharpe_alerts) for logging.
    """
    # Step 1: Compute daily SHAP
    shap_pct = compute_daily_shap(model, X_scored, feature_names)

    # Step 2: Update history
    history = update_shap_history(shap_pct)

    # Step 3: Check SHAP alerts
    shap_alerts = check_shap_alerts(history)

    # Step 4: Rolling Sharpe
    sharpe_value, sharpe_color = compute_rolling_sharpe(equity_curve)
    sharpe_alerts = check_sharpe_alerts(sharpe_value)

    # Step 5: Update dashboard data
    update_dashboard_data(shap_pct, history, shap_alerts, sharpe_value, sharpe_color)

    # Log alerts
    all_alerts = shap_alerts + sharpe_alerts
    for severity, message in all_alerts:
        if severity == "critical":
            log.warning(f"SHAP ALERT [CRITICAL]: {message}")
        elif severity == "warning":
            log.warning(f"SHAP ALERT [WARNING]: {message}")
        else:
            log.info(f"SHAP ALERT [{severity.upper()}]: {message}")

    log.info(f"SHAP top 3: {dict(shap_pct.nlargest(3).items())}")
    log.info(f"HHI: {history[-1]['hhi']:.4f}")
    if sharpe_value is not None:
        log.info(f"Rolling 60d Sharpe: {sharpe_value:.2f} ({sharpe_color})")

    return shap_alerts, sharpe_alerts


# =====================================================================
# STANDALONE TEST
# =====================================================================
def main_test():
    """Standalone test: load seed 6 model, compute SHAP on factor cache."""
    import sys
    sys.path.insert(0, str(PROD.parent.parent))
    import lightgbm as lgb
    import pickle

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    ROOT = PROD.parent.parent
    model_dir = Path(r"C:\ai-research-team\v10_ml_v3\seed_6")
    model_path = model_dir / "lgb_model_2026.txt"
    if not model_path.exists():
        model_path = model_dir / "lgb_model_2025.txt"
    model = lgb.Booster(model_file=str(model_path))
    log.info(f"Loaded model: {model_path} ({model.num_feature()} features)")

    # Load factor cache for feature names
    feature_names = model.feature_name()
    n_features = len(feature_names)
    log.info(f"Model has {n_features} features: {feature_names[:5]}...")

    # Build a synthetic scored universe (random data for testing SHAP mechanics)
    # In production, X_scored comes from score_stocks()
    np.random.seed(42)
    n_stocks = 500
    X_test = np.random.randn(n_stocks, n_features).astype(np.float32)
    log.info(f"Test matrix: {X_test.shape}")

    # Run the full pipeline
    t0 = time.time()
    shap_alerts, sharpe_alerts = run_shap_monitor(
        model, X_test, feature_names,
        equity_curve=[
            {"date": "2026-03-23", "actual": 100000},
            {"date": "2026-03-24", "actual": 98785},
            {"date": "2026-03-25", "actual": 99200},
        ]
    )
    elapsed = time.time() - t0
    log.info(f"Total SHAP monitor runtime: {elapsed:.1f}s")

    # Print results
    history = load_shap_history()
    latest = history[-1]
    print("\n" + "=" * 60)
    print("SHAP MONITOR TEST RESULTS")
    print("=" * 60)
    print(f"Date: {latest['date']}")
    print(f"HHI:  {latest['hhi']:.4f}")
    print(f"\nTop 10 features by SHAP importance:")
    sorted_feats = sorted(latest["shap_importance"].items(),
                          key=lambda x: x[1], reverse=True)
    for i, (feat, imp) in enumerate(sorted_feats[:10]):
        bar = "#" * int(imp / 2)
        flag = " *** ABOVE 25% ***" if imp > 25 else ""
        print(f"  {i+1:2d}. {feat:<30s} {imp:5.1f}% {bar}{flag}")

    print(f"\nAlerts ({len(shap_alerts)} SHAP, {len(sharpe_alerts)} Sharpe):")
    for sev, msg in shap_alerts + sharpe_alerts:
        print(f"  [{sev.upper()}] {msg}")

    print(f"\nRuntime: {elapsed:.1f}s")
    print(f"SHAP history file: {SHAP_HISTORY_FILE}")
    print(f"Dashboard data:    {DASHBOARD_DATA_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        main_test()
    else:
        print("Usage: python shap_monitor.py --test")
        print("  Runs standalone SHAP computation on seed 6 model with synthetic data.")
