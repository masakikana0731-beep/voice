@echo off
rem Voice Studio をダブルクリックで起動（ブラウザが開きます）
cd /d "%~dp0"
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

rem すでに起動中なら、画面を開くだけ（二重に起動しない）
netstat -ano | findstr ":7860 " | findstr "LISTENING" >nul
if not errorlevel 1 (
  start "" "https://kumoshibito.tailfd662b.ts.net/"
  exit /b
)

title Voice Studio（この画面を閉じると止まります）
".venv\Scripts\python.exe" -m studio.app
pause
