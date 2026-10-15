@echo off
chcp 65001 >nul
rem Voice Studio Live（PC 専用）: リアルタイム声変換を通話アプリのマイクに流す。本人の同意がない声は使えません
cd /d "%~dp0"
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
title Voice Studio Live（この画面を閉じると止まります）
".venv\Scripts\python.exe" -m studio.live
pause
