# Voice Studio（学習 + 声変換）用の追加セットアップ。先に setup.ps1 を実行しておくこと。
# 使い方: powershell -ExecutionPolicy Bypass -File setup\setup-studio.ps1
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
function Invoke-Checked { & $args[0] $args[1..($args.Count - 1)]; if ($LASTEXITCODE) { throw "失敗: $args" } }

# 学習用の追加ライブラリ（メモリ節約の 8bit 最適化など）
Invoke-Checked uv pip install --python .\.venv\Scripts\python.exe bitsandbytes tensorboard

# ffmpeg（スマホ録音 m4a やブラウザ録音の読み込み用）
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    winget install --id Gyan.FFmpeg -e --source winget --accept-package-agreements --accept-source-agreements --disable-interactivity
}

# 公式の学習コードと声変換（Seed-VC）
New-Item -ItemType Directory -Force third_party | Out-Null
if (-not (Test-Path third_party\Qwen3-TTS)) { Invoke-Checked git clone --depth 1 https://github.com/QwenLM/Qwen3-TTS.git third_party\Qwen3-TTS }
if (-not (Test-Path third_party\seed-vc)) { Invoke-Checked git clone --depth 1 https://github.com/Plachtaa/seed-vc.git third_party\seed-vc }

# Seed-VC は依存が古いので別環境（.venv-vc）。torch は本体と同じ 2.11 を使う（2.4 はこの PC で DLL エラー）
uv python install 3.10
$py = Get-ChildItem "$env:APPDATA\uv\python\cpython-3.10.*-windows-x86_64-none\python.exe" |
    Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
Invoke-Checked uv venv --allow-existing --python $py .venv-vc
Invoke-Checked uv pip install --python .\.venv-vc\Scripts\python.exe torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
# torch 系は上で入れたので除外。resemblyzer / jiwer は評価用で、C++ コンパイラが要るため除外
Get-Content third_party\seed-vc\requirements.txt |
    Where-Object { $_ -and $_ -notmatch '^(torch|torchvision|torchaudio|FreeSimpleGUI|sounddevice|resemblyzer|jiwer)|index-url' } |
    Set-Content -Encoding utf8 third_party\seed-vc-req.txt
Invoke-Checked uv pip install --python .\.venv-vc\Scripts\python.exe -r third_party\seed-vc-req.txt hf_xet

Write-Host ""
Write-Host "[setup] 完了。start-studio.bat をダブルクリックすると Voice Studio が開きます。"
