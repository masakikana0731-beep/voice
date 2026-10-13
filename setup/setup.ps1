# Qwen3-TTS ローカル環境セットアップ（Windows PowerShell）
# 使い方: PowerShell で  powershell -ExecutionPolicy Bypass -File setup\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "[setup] uv をインストールします"
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

uv venv --python 3.12 .venv
. .\.venv\Scripts\Activate.ps1

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "[setup] NVIDIA GPU を検出 -> CUDA 版 PyTorch"
    uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
} else {
    Write-Host "[setup] GPU なし -> CPU 版 PyTorch（かなり遅くなります。AivisSpeech の利用を推奨）"
    uv pip install torch torchaudio
}

uv pip install -r requirements.txt

python -c "import torch;print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"
Write-Host ""
Write-Host "[setup] 完了。次を試してください:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python tts.py speak `"こんにちは、ローカル音声合成のテストです。`""
Write-Host "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --port 8000   # Web UI"
