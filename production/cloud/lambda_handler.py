"""Lambda entrypoint for daily signal generator.

Triggered by EventBridge cron at 19:55 UTC weekdays.

Flow:
  1. download_state() pulls last-run state.json + ohlcv_cache.pkl from S3 → /tmp/state/
  2. Mirror /var/task/production/{ml_v3,ml_v4}/ → /tmp/work/ (writable)
  3. Copy state files into /tmp/work/.../ where the scripts expect them
  4. Spawn v3_fresh + v4_gtc as parallel subprocesses (each with isolated env)
  5. Wait for v4_gtc, then spawn v4_moc (it reads v4_gtc's shared scoring state)
  6. Wait for all to finish
  7. Copy state files back to /tmp/state/, upload to S3
"""
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import date, datetime
from pathlib import Path

# Lambda-specific imports
sys.path.insert(0, "/var/task")
import s3_state

STATE_DIR = Path("/tmp/state")
WORK_DIR = Path("/tmp/work")
TASK_ROOT = Path("/var/task")
PYTHON = sys.executable

# 2026 NYSE holidays
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


def _ensure_work_module(name):
    """Mirror production/<name>/ from /var/task to writable /tmp/work/.
    Always force-fresh. Also creates logs/state/tracking subdirs."""
    src = TASK_ROOT / "production" / name
    dst = WORK_DIR / "production" / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    (WORK_DIR / "production" / "logs").mkdir(parents=True, exist_ok=True)
    (WORK_DIR / "production" / "state").mkdir(parents=True, exist_ok=True)
    (WORK_DIR / "production" / "tracking").mkdir(parents=True, exist_ok=True)
    return dst


def _stage_state_into_module(module_dir, state_files):
    """Copy state files from /tmp/state/ into the module dir before running."""
    for rel_in_state, rel_in_module in state_files.items():
        src = STATE_DIR / rel_in_state
        dst = Path(module_dir) / rel_in_module
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy(src, dst)


def _stage_state_back(module_dir, state_files):
    """Copy state files back from module dir to /tmp/state/ after running."""
    for rel_in_state, rel_in_module in state_files.items():
        src = Path(module_dir) / rel_in_module
        dst = STATE_DIR / rel_in_state
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy(src, dst)


def _make_env(extra):
    """Return a fresh os.environ copy with `extra` overrides applied."""
    env = dict(os.environ)
    env.update(extra)
    return env


def _spawn(name, script_path, env_overrides, log_path, dry_run=False):
    """Spawn a subprocess for one signal generator account."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w")
    cmd = [PYTHON, script_path]
    if dry_run:
        cmd.append("--dry-run")
    proc = subprocess.Popen(
        cmd,
        env=_make_env(env_overrides),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        cwd=str(WORK_DIR),
    )
    print(f"[{datetime.now():%H:%M:%S}] {name}: PID {proc.pid} -> {log_path.name} (dry_run={dry_run})")
    return proc, log_fh


def handler(event, context):
    dry_run = bool((event or {}).get("dry_run", False))
    print(f"=== Signal generator Lambda starting at {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC ===")
    print(f"Local date: {date.today()}  dry_run={dry_run}")

    if not is_market_day():
        print(f"Market closed today ({date.today():%A %Y-%m-%d}). Skipping.")
        return {"status": "skipped", "reason": "market_closed"}

    t0 = time.time()
    errors = []

    # 1. Download state from S3
    try:
        s3_state.download_state()
    except Exception as e:
        print(f"[ERROR] state download: {e}")
        errors.append(("state_download", str(e)))

    # 2. Mirror module dirs to writable /tmp/work/
    v3_module = _ensure_work_module("ml_v3")
    v4_module = _ensure_work_module("ml_v4")

    # 3. Stage state files into each module
    v3_state_map = {
        "ml_v3/state_fresh.json": "state_fresh.json",
        "ml_v3/ohlcv_cache.pkl": "ohlcv_cache.pkl",
    }
    v4_gtc_state_map = {
        "ml_v4/state_gtc.json": "state_gtc.json",
        "ml_v4/ohlcv_cache_gtc.pkl": "ohlcv_cache_gtc.pkl",
        "ml_v4/shap_history_gtc.json": "shap_history_gtc.json",
    }
    v4_moc_state_map = {
        "ml_v4/state_moc.json": "state_moc.json",
        "ml_v4/ohlcv_cache_moc.pkl": "ohlcv_cache_moc.pkl",
        "ml_v4/shap_history.json": "shap_history.json",
    }
    shared_scoring_map = {
        "shared_scoring_state.pkl": "../shared_scoring_state.pkl",
    }
    _stage_state_into_module(v3_module, v3_state_map)
    _stage_state_into_module(v4_module, v4_gtc_state_map)
    _stage_state_into_module(v4_module, v4_moc_state_map)
    _stage_state_into_module(v4_module, shared_scoring_map)

    # 4. Common Alpaca env for all subprocesses
    base_env = {"PYTHONPATH": str(TASK_ROOT)}

    # 5. Launch v3_fresh + v4_gtc in PARALLEL via subprocess
    v3_log = WORK_DIR / "production" / "logs" / "v3_fresh.log"
    v4_gtc_log = WORK_DIR / "production" / "logs" / "v4_gtc.log"
    v4_moc_log = WORK_DIR / "production" / "logs" / "v4_moc.log"

    v3_env = {
        **base_env,
        "ALPACA_API_KEY": os.environ["V3_FRESH_API_KEY"],
        "ALPACA_SECRET_KEY": os.environ["V3_FRESH_API_SECRET"],
        "APCA_API_KEY_ID": os.environ["V3_FRESH_API_KEY"],
        "APCA_API_SECRET_KEY": os.environ["V3_FRESH_API_SECRET"],
        "V3_STATE_FILE": str(v3_module / "state_fresh.json"),
    }
    v4_gtc_env = {
        **base_env,
        "ALPACA_API_KEY": os.environ["V4_GTC_API_KEY"],
        "ALPACA_SECRET_KEY": os.environ["V4_GTC_API_SECRET"],
        "APCA_API_KEY_ID": os.environ["V4_GTC_API_KEY"],
        "APCA_API_SECRET_KEY": os.environ["V4_GTC_API_SECRET"],
        "V4_STOP_MODE": "gtc",
        "V4_SCORING_MODE": "write",
    }
    v4_moc_env = {
        **base_env,
        "ALPACA_API_KEY": os.environ["V4_MOC_API_KEY"],
        "ALPACA_SECRET_KEY": os.environ["V4_MOC_API_SECRET"],
        "APCA_API_KEY_ID": os.environ["V4_MOC_API_KEY"],
        "APCA_API_SECRET_KEY": os.environ["V4_MOC_API_SECRET"],
        "V4_STOP_MODE": "moc",
        "V4_SCORING_MODE": "read",
    }

    v3_script = str(v3_module / "early_signal_ml_v3.py")
    v4_script = str(v4_module / "early_signal_ml_v4.py")

    print(f"[{datetime.now():%H:%M:%S}] Launching v3_fresh + v4_gtc in parallel...")
    p_v3, fh_v3 = _spawn("v3_fresh", v3_script, v3_env, v3_log, dry_run=dry_run)
    time.sleep(5)  # stagger to avoid yfinance rate limits
    p_gtc, fh_gtc = _spawn("v4_gtc", v4_script, v4_gtc_env, v4_gtc_log, dry_run=dry_run)

    # Wait for v4_gtc (v4_moc reads its shared scoring state)
    try:
        rc_gtc = p_gtc.wait(timeout=600)
        fh_gtc.close()
        print(f"[{datetime.now():%H:%M:%S}] v4_gtc finished rc={rc_gtc}")
        if rc_gtc != 0:
            errors.append(("v4_gtc", f"rc={rc_gtc}"))
    except subprocess.TimeoutExpired:
        p_gtc.kill()
        fh_gtc.close()
        errors.append(("v4_gtc", "timeout 600s"))

    # Now spawn v4_moc (depends on v4_gtc's shared scoring state)
    print(f"[{datetime.now():%H:%M:%S}] Launching v4_moc...")
    p_moc, fh_moc = _spawn("v4_moc", v4_script, v4_moc_env, v4_moc_log, dry_run=dry_run)

    # Wait for v3 (might still be running) and v4_moc
    try:
        rc_v3 = p_v3.wait(timeout=600)
        fh_v3.close()
        print(f"[{datetime.now():%H:%M:%S}] v3_fresh finished rc={rc_v3}")
        if rc_v3 != 0:
            errors.append(("v3_fresh", f"rc={rc_v3}"))
    except subprocess.TimeoutExpired:
        p_v3.kill()
        fh_v3.close()
        errors.append(("v3_fresh", "timeout 600s"))

    try:
        rc_moc = p_moc.wait(timeout=600)
        fh_moc.close()
        print(f"[{datetime.now():%H:%M:%S}] v4_moc finished rc={rc_moc}")
        if rc_moc != 0:
            errors.append(("v4_moc", f"rc={rc_moc}"))
    except subprocess.TimeoutExpired:
        p_moc.kill()
        fh_moc.close()
        errors.append(("v4_moc", "timeout 600s"))

    # Print last lines of each log for visibility in CloudWatch
    for name, log in [("v3_fresh", v3_log), ("v4_gtc", v4_gtc_log), ("v4_moc", v4_moc_log)]:
        if log.exists():
            print(f"--- last 20 lines of {name} log ---")
            try:
                lines = log.read_text(errors="replace").splitlines()[-20:]
                print("\n".join(lines))
            except Exception as e:
                print(f"  could not read log: {e}")

    # 6. Stage state back to /tmp/state/ for S3 upload
    _stage_state_back(v3_module, v3_state_map)
    _stage_state_back(v4_module, v4_gtc_state_map)
    _stage_state_back(v4_module, v4_moc_state_map)
    _stage_state_back(v4_module, shared_scoring_map)

    # 7. Upload state to S3
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
    handler({}, None)
