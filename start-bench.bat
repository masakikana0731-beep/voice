@echo off
chcp 65001 > nul
rem 精度検証（studio/bench.py）を続きから実行する。Claude アプリとは別に動くよう、タスクスケジューラから起動される
cd /d "%~dp0"
set PYTHONUTF8=1
".venv\Scripts\python.exe" -m studio.bench run >> "bench\logs\run.log" 2>&1
