#!/usr/bin/env python3
"""
Wrapper: warm cache then launch all 3 signal generators in parallel.
Run via Task Scheduler at 3:48 PM ET daily.

Accounts:
  1. v3 production (fresh) - MOC stops
  2. v4 #1188 seed 6 - GTC stops
  3. v4 #1188 seed 6 - MOC stops
"""
import subprocess, sys, time, os
from pathlib import Path
from datetime import datetime, date

ROOT = Path(__file__).parent.parent
PROD = Path(__file__).parent
PYTHON = sys.executable

LOG_DIR = PROD / "logs"
LOG_DIR.mkdir(exist_ok=True)

today = datetime.now().strftime("%Y%m%d")

# --- Market day check: skip weekends and known NYSE holidays ---
def is_market_day(d=None):
    """Return True if NYSE is open on the given date."""
    if d is None:
        d = date.today()
    # Weekends
    if d.weekday() >= 5:
        return False
    # Known NYSE holidays (fixed + observed). Update annually.
    # 2026 calendar: https://www.nyse.com/markets/hours-calendars
    nyse_holidays_2026 = {
        date(2026, 1, 1),   # New Year's Day
        date(2026, 1, 19),  # MLK Day
        date(2026, 2, 16),  # Presidents' Day
        date(2026, 4, 3),   # Good Friday
        date(2026, 5, 25),  # Memorial Day
        date(2026, 6, 19),  # Juneteenth
        date(2026, 7, 3),   # Independence Day (observed)
        date(2026, 9, 7),   # Labor Day
        date(2026, 11, 26), # Thanksgiving
        date(2026, 12, 25), # Christmas
    }
    return d not in nyse_holidays_2026

if not is_market_day():
    print(f"[{datetime.now():%H:%M:%S}] Market closed today ({date.today():%A %Y-%m-%d}). Skipping.")
    sys.exit(0)

print(f"[{datetime.now():%H:%M:%S}] Signal runner starting...")

# Step 1: Warm the OHLCV cache (shared by all generators)
print(f"[{datetime.now():%H:%M:%S}] Warming OHLCV cache...")
# Import and run the cache warm-up from v4 (v3 uses same cache location)
cache_script = f"""
import sys; sys.path.insert(0, r'{ROOT}')
from pathlib import Path
import pickle, time

# Check if v4 cache exists and is fresh
cache_v4 = Path(r'{PROD / "ml_v4" / "ohlcv_cache.pkl"}')
cache_v3 = Path(r'{PROD / "ml_v3" / "ohlcv_cache.pkl"}')
from datetime import date
needs_warm = True

for cf in [cache_v4, cache_v3]:
    if cf.exists():
        try:
            with open(cf, 'rb') as f:
                d = pickle.load(f)
            if d.get('last_date') == date.today():
                print(f'Cache {{cf}} already warm (today)')
                needs_warm = False
                break
        except:
            pass

if needs_warm:
    print('Cache needs warming - first signal generator will handle it')
else:
    print('Cache is warm')
"""
subprocess.run([PYTHON, "-c", cache_script], cwd=str(ROOT), timeout=30)

# Step 2: Launch all 3 signal generators in parallel
print(f"[{datetime.now():%H:%M:%S}] Launching signal generators (5s stagger)...")

processes = []

# Load env files for each account
def load_env_file(path):
    env = dict(os.environ)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env

accounts_dir = PROD / "accounts"

# V3 fresh account
# Override the .env keys so v3's load_dotenv doesn't use the old account
v3_env = load_env_file(accounts_dir / "v3_fresh.env")
v3_env["APCA_API_KEY_ID"] = v3_env.get("ALPACA_API_KEY", "")
v3_env["APCA_API_SECRET_KEY"] = v3_env.get("ALPACA_SECRET_KEY", "")
v3_env["V3_STATE_FILE"] = str(PROD / "ml_v3" / "state_fresh.json")
p1 = subprocess.Popen(
    [PYTHON, str(PROD / "ml_v3" / "early_signal_ml_v3.py")],
    cwd=str(ROOT),
    stdout=open(LOG_DIR / f"v3_fresh_{today}.log", "w"),
    stderr=subprocess.STDOUT,
    env=v3_env,
)
processes.append(("v3_fresh", p1))
print(f"  v3_fresh: PID {p1.pid}")

time.sleep(5)  # stagger to avoid yfinance rate limits

# V4 GTC stops (runs first, saves shared scoring state for MOC)
v4_gtc_env = load_env_file(accounts_dir / "v4_gtc.env")
v4_gtc_env["V4_STOP_MODE"] = "gtc"
v4_gtc_env["V4_SCORING_MODE"] = "write"
p2 = subprocess.Popen(
    [PYTHON, str(PROD / "ml_v4" / "early_signal_ml_v4.py")],
    cwd=str(ROOT),
    stdout=open(LOG_DIR / f"v4_gtc_{today}.log", "w"),
    stderr=subprocess.STDOUT,
    env=v4_gtc_env,
)
processes.append(("v4_gtc", p2))
print(f"  v4_gtc: PID {p2.pid}")

# Wait for v4 GTC to finish so its shared scoring state is available for MOC.
# Using process.wait() instead of a fixed sleep -- robust against slow runs.
p2.wait(timeout=600)
gtc_status = "OK" if p2.returncode == 0 else f"FAILED (rc={p2.returncode})"
print(f"  v4_gtc finished: {gtc_status}")

# V4 MOC stops (loads GTC's shared scoring state = identical features/scores)
v4_moc_env = load_env_file(accounts_dir / "v4_moc.env")
v4_moc_env["V4_STOP_MODE"] = "moc"
v4_moc_env["V4_SCORING_MODE"] = "read"
p3 = subprocess.Popen(
    [PYTHON, str(PROD / "ml_v4" / "early_signal_ml_v4.py")],
    cwd=str(ROOT),
    stdout=open(LOG_DIR / f"v4_moc_{today}.log", "w"),
    stderr=subprocess.STDOUT,
    env=v4_moc_env,
)
processes.append(("v4_moc", p3))
print(f"  v4_moc: PID {p3.pid}")

# Step 3: Wait for all to finish
print(f"[{datetime.now():%H:%M:%S}] Waiting for completion...")
for name, proc in processes:
    proc.wait(timeout=600)  # 10 min max
    status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
    print(f"  {name}: {status}")

print(f"[{datetime.now():%H:%M:%S}] All signal generators complete.")

# Step 4: Update dashboard and push to S3
print(f"[{datetime.now():%H:%M:%S}] Updating dashboard...")
try:
    dash_result = subprocess.run(
        [PYTHON, str(PROD / "update_dashboard.py")],
        cwd=str(ROOT), capture_output=True, text=True, timeout=60)
    if dash_result.returncode == 0:
        print(f"[{datetime.now():%H:%M:%S}] Dashboard updated and pushed to S3")
    else:
        print(f"[{datetime.now():%H:%M:%S}] Dashboard update failed: {dash_result.stderr[:200]}")
except Exception as e:
    print(f"[{datetime.now():%H:%M:%S}] Dashboard update error: {e}")

# Step 5: Reconcile fills (fetch actual Alpaca fill prices, compute slippage)
# Runs immediately -- for MOC orders, fills may not be available yet.
# The reconciler is idempotent: re-running later will pick up any fills it missed.
print(f"[{datetime.now():%H:%M:%S}] Reconciling fills...")
try:
    recon_result = subprocess.run(
        [PYTHON, str(PROD / "reconcile_fills.py")],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    if recon_result.returncode == 0:
        print(f"[{datetime.now():%H:%M:%S}] Fill reconciliation complete")
    else:
        print(f"[{datetime.now():%H:%M:%S}] Reconciliation failed: {recon_result.stderr[:200]}")
except Exception as e:
    print(f"[{datetime.now():%H:%M:%S}] Reconciliation error: {e}")
