@echo off
chcp 65001 > nul
rem あゆみんの再学習（読み上げ・声変換）。素材の準備とリアルタイム用は前回のまま
rem 学習の記録は voices\ayumin\train.log（学習側が書く）。ここの出力は別のファイルへ（同じファイルだと書き込みがぶつかる）
cd /d "%~dp0"
set PYTHONUTF8=1
".venv\Scripts\python.exe" -m studio.train ayumin --skip prep --skip rt >> "voices\ayumin\train_run.log" 2>&1
