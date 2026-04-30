"""Lambda entrypoint for daily signal generator.

Triggered by EventBridge cron at 19:55 UTC (3:55 PM EST) Mon-Fri.
During EDT (March-Nov) the rule is 19:55 UTC = 3:55 PM EDT.

Flow:
  1. download_state() pulls last-run state.json + ohlcv_cache.pkl from S3
  2. Symlink/copy state files into the locations the v3 + v4 scripts expect
  3. Run v3_fresh, v4_gtc, v4_moc sequentially (Lambda is single-process)
  4. update_dashboard.py + reconcile_fills.py
  5. upload_state() pushes everything back to S3
"""
import json
import os
import shutil
import sys
import time
import traceback
from datetime import date, datetime
from pathlib import Path

# Lambda-specific imports
sys.path.insert(0, "/var/task")
import s3_state

# Use /tmp/state as the runtime state location.
# Lambda's /var/task is read-only, so we copy ml_v3/ and ml_v4/ subtrees to
# /tmp/work/ (writable) and run from there. State files load/save under /tmp/work/.
STATE_DIR = Path("/tmp/state")
WORK_DIR = Path("/tmp/work")
TASK_ROOT = Path("/var/task")  # source: where Dockerfile copied production/, v10_ml_v3/, etc.

# 2026 NYSE holidays (matches run_all_signals.py)
NYSE_HOLIDAYS_2026 = {
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
}


def is_market_day(d=None):
    if d is None:
        d = date.today()
    if d.weekday() >= 5:
        return False
    return d not in NYSE_HOLIDAYS_2026


def link_state_into_module(module_dir, state_files):
    """For each state file, copy from /tmp/state/{rel} into the module dir
    where the early_signal script expects to find it."""
    for rel_in_state, rel_in_module in state_files.items():
        src = STATE_DIR / rel_in_state
        dst = Path(module_dir) / rel_in_module
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy(src, dst)


def link_state_back(module_dir, state_files):
    """After run, copy module dir state back to /tmp/state for upload."""
    for rel_in_state, rel_in_module in state_files.items():
        src = Path(module_dir) / rel_in_module
        dst = STATE_DIR / rel_in_state
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy(src, dst)


def _ensure_work_module(name):
    """Mirror production/<name>/ from read-only /var/task to writable /tmp/work."""
    src = TASK_ROOT / "production" / name
    dst = WORK_DIR / "production" / name
    if not dst.exists():
        shutil.copytree(src, dst)
    return dst


def run_v3_fresh():
    """Run v3 fresh signal generator inline (no subprocess, Lambda is one process)."""
    print(f"[{datetime.now():%H:%M:%S}] === v3_fresh ===")
    module_dir = _ensure_work_module("ml_v3")
    state_map = {
        "ml_v3/state_fresh.json": "state_fresh.json",
        "ml_v3/ohlcv_cache.pkl": "ohlcv_cache.pkl",
    }
    link_state_into_module(module_dir, state_map)

    # Set env for v3 account from secrets
    os.environ["APCA_API_KEY_ID"] = os.environ["V3_FRESH_API_KEY"]
    os.environ["APCA_API_SECRET_KEY"] = os.environ["V3_FRESH_API_SECRET"]
    os.environ["V3_STATE_FILE"] = str(module_dir / "state_fresh.json")

    # Run as module
    os.chdir(str(TASK_ROOT))
    sys.path.insert(0, str(module_dir))
    try:
        # Force fresh module load each invocation (Lambda may reuse container)
        for mod_name in list(sys.modules):
            if mod_name.startswith("early_signal_ml_v3"):
                del sys.modules[mod_name]
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "early_signal_ml_v3", str(module_dir / "early_signal_ml_v3.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        m.run_signal(dry_run=False)
    finally:
        link_state_back(module_dir, state_map)
        sys.path.remove(str(module_dir))


def run_v4(account):
    """account: 'gtc' or 'moc'."""
    print(f"[{datetime.now():%H:%M:%S}] === v4_{account} ===")
    module_dir = _ensure_work_module("ml_v4")
    state_map = {
        f"ml_v4/state_{account}.json": f"state_{account}.json",
        f"ml_v4/ohlcv_cache_{account}.pkl": f"ohlcv_cache_{account}.pkl",
        f"ml_v4/shap_history_{'gtc' if account == 'gtc' else ''}.json":
            f"shap_history{'_gtc' if account == 'gtc' else ''}.json",
        "shared_scoring_state.pkl": "../shared_scoring_state.pkl",
    }
    link_state_into_module(module_dir, state_map)

    if account == "gtc":
        os.environ["APCA_API_KEY_ID"] = os.environ["V4_GTC_API_KEY"]
        os.environ["APCA_API_SECRET_KEY"] = os.environ["V4_GTC_API_SECRET"]
        os.environ["V4_STOP_MODE"] = "gtc"
        os.environ["V4_SCORING_MODE"] = "write"
    else:  # moc
        os.environ["APCA_API_KEY_ID"] = os.environ["V4_MOC_API_KEY"]
        os.environ["APCA_API_SECRET_KEY"] = os.environ["V4_MOC_API_SECRET"]
        os.environ["V4_STOP_MODE"] = "moc"
        os.environ["V4_SCORING_MODE"] = "read"

    os.chdir(str(TASK_ROOT))
    sys.path.insert(0, str(module_dir))
    try:
        for mod_name in list(sys.modules):
            if mod_name.startswith("early_signal_ml_v4"):
                del sys.modules[mod_name]
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "early_signal_ml_v4", str(module_dir / "early_signal_ml_v4.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        m.run_signal(dry_run=False)
    finally:
        link_state_back(module_dir, state_map)
        sys.path.remove(str(module_dir))


def handler(event, context):
    """Lambda entrypoint. EventBridge passes empty event."""
    print(f"=== Signal generator Lambda starting at {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC ===")
    print(f"Local date: {date.today()}")

    if not is_market_day():
        print(f"Market closed today ({date.today():%A %Y-%m-%d}). Skipping.")
        return {"status": "skipped", "reason": "market_closed"}

    t0 = time.time()
    errors = []

    # Step 1: Download state from S3
    try:
        s3_state.download_state()
    except Exception as e:
        print(f"[ERROR] state download: {e}")
        errors.append(("state_download", str(e)))
        # Continue — first run may have no state

    # Step 2: Run all 3 signal generators sequentially (Lambda is one process)
    for fn, name in [(run_v3_fresh, "v3_fresh"), (lambda: run_v4("gtc"), "v4_gtc"),
                     (lambda: run_v4("moc"), "v4_moc")]:
        try:
            fn()
        except SystemExit as e:
            # early_signal scripts may sys.exit() — treat 0 as success
            if e.code == 0:
                print(f"  {name}: clean exit")
            else:
                print(f"  {name}: SystemExit({e.code})")
                errors.append((name, f"SystemExit({e.code})"))
        except Exception as e:
            print(f"  {name}: ERROR {e}")
            traceback.print_exc()
            errors.append((name, str(e)))

    # Step 3: Always try to upload state, even if some accounts failed
    try:
        s3_state.upload_state()
    except Exception as e:
        print(f"[ERROR] state upload: {e}")
        errors.append(("state_upload", str(e)))

    elapsed = time.time() - t0
    result = {
        "status": "ok" if not errors else "partial",
        "elapsed_sec": elapsed,
        "errors": errors,
    }
    print(f"=== Lambda done in {elapsed:.0f}s, status={result['status']} ===")
    return result


if __name__ == "__main__":
    # Local test (won't have S3 access unless AWS creds set)
    handler({}, None)
