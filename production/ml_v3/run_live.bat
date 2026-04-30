@echo off
REM ML v3 Paper Trading - Run at 3:50 PM ET
REM Schedule this in Windows Task Scheduler

cd /d C:\ai-research-team
call venv\Scripts\activate

REM Run signal generator (LIVE mode - no --dry-run flag)
python production\ml_v3\early_signal_ml_v3.py

REM Refresh factor cache with today's data (downloads yfinance, rebuilds cache, rescores)
python production\ml_v3\refresh_cache.py

REM Run backtest replay for dashboard equity curve
python production\ml_v3\backtest_replay.py

REM Update dashboard with Alpaca state + S3 sync
python production\ml_v3\update_dashboard.py

echo Done at %date% %time%
