@echo off
rem Voice Studio をダブルクリックで起動（ブラウザが開きます）
cd /d "%~dp0"
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
".venv\Scripts\python.exe" -m studio.app
pause
