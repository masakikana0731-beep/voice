@echo off
chcp 65001 > nul
rem あゆみんの再学習（素材の準備＝自動選別込み・読み上げ・声変換）。リアルタイム用は前回のまま
rem 学習の記録は voices\ayumin\train.log（学習側が書く）。ここの出力は別のファイルへ（同じファイルだと書き込みがぶつかる）
cd /d "%~dp0"
set PYTHONUTF8=1
".venv\Scripts\python.exe" -m studio.train ayumin --skip rt >> "voices\ayumin\train_run.log" 2>&1
