"""S3 state sync helpers.

Lambda is stateless — between runs we persist:
  state_fresh.json, state_gtc.json, state_moc.json   (per-account state)
  ohlcv_cache.pkl                                    (yfinance cache, 5d updates)
  shared_scoring_state.pkl                           (v4 GTC -> MOC handoff)

S3 layout: s3://{bucket}/runtime-state/{filename}

At handler start: download_state() copies S3 -> /tmp/state/
At handler end:   upload_state()   copies /tmp/state/ -> S3
"""
import os
from pathlib import Path

import boto3

S3_BUCKET = os.environ.get("STATE_S3_BUCKET", "getrich-runtime-state")
S3_PREFIX = "runtime-state"
LOCAL_STATE_DIR = Path("/tmp/state")

# Files we sync. Map: local-relative-path -> S3 key suffix
SYNCED_FILES = {
    "ml_v3/state_fresh.json":           "state_fresh.json",
    "ml_v3/ohlcv_cache.pkl":            "ohlcv_cache_v3.pkl",
    "ml_v4/state_gtc.json":             "state_gtc.json",
    "ml_v4/state_moc.json":             "state_moc.json",
    "ml_v4/ohlcv_cache_gtc.pkl":        "ohlcv_cache_gtc.pkl",
    "ml_v4/ohlcv_cache_moc.pkl":        "ohlcv_cache_moc.pkl",
    "ml_v4/shap_history_gtc.json":      "shap_history_gtc.json",
    "ml_v4/shap_history.json":          "shap_history.json",
    "shared_scoring_state.pkl":         "shared_scoring_state.pkl",
    "dashboard_data.json":              "dashboard_data.json",
    "dashboard_data_v4_gtc.json":       "dashboard_data_v4_gtc.json",
    "dashboard_data_v4_moc.json":       "dashboard_data_v4_moc.json",
    "tracking/reconciled_trades.csv":   "reconciled_trades.csv",
}


def _s3():
    return boto3.client("s3")


def download_state():
    """Download persisted state from S3 to /tmp/state/. Missing files are OK (first run)."""
    s3 = _s3()
    LOCAL_STATE_DIR.mkdir(parents=True, exist_ok=True)
    n_downloaded = 0
    n_missing = 0
    for local_rel, s3_suffix in SYNCED_FILES.items():
        local_path = LOCAL_STATE_DIR / local_rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            s3.download_file(S3_BUCKET, f"{S3_PREFIX}/{s3_suffix}", str(local_path))
            n_downloaded += 1
        except s3.exceptions.ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code == "404" or code == "NoSuchKey":
                n_missing += 1  # first run for this file — OK
            else:
                raise
    print(f"[s3_state] downloaded {n_downloaded} files, {n_missing} missing (first-run)")
    return LOCAL_STATE_DIR


def upload_state():
    """Upload current /tmp/state/ contents back to S3."""
    s3 = _s3()
    n_uploaded = 0
    n_skipped = 0
    for local_rel, s3_suffix in SYNCED_FILES.items():
        local_path = LOCAL_STATE_DIR / local_rel
        if not local_path.exists():
            n_skipped += 1
            continue
        s3.upload_file(str(local_path), S3_BUCKET, f"{S3_PREFIX}/{s3_suffix}")
        n_uploaded += 1
    print(f"[s3_state] uploaded {n_uploaded} files, {n_skipped} not present (skipped)")


def get_state_path(local_rel):
    """Return /tmp/state/{local_rel} for use by handler."""
    return LOCAL_STATE_DIR / local_rel
