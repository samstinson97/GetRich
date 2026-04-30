@echo off
REM ML v3 Paper Trading - Dry Run (no real orders)

cd /d C:\ai-research-team
call venv\Scripts\activate

python production\ml_v3\early_signal_ml_v3.py --dry-run

echo Done at %date% %time%
pause
