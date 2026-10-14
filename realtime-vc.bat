@echo off
rem （テスト機能）リアルタイム声変換。PC のマイクに話すと、その場で声が変わる（PC のみ）
rem 画面で「Reference Audio」に voices\<名前>\ref_vc.wav を選び、入力（マイク）と出力（スピーカー）を選んで Start
cd /d "%~dp0third_party\seed-vc"
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
"%~dp0.venv-vc\Scripts\python.exe" "%~dp0studio\vc_run.py" real-time-gui.py
pause
