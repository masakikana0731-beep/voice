# Qwen3-TTS ローカル環境セットアップ（Windows PowerShell）
# 使い方: PowerShell で  powershell -ExecutionPolicy Bypass -File setup\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "[setup] uv をインストールします"
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

# uv が出す進捗メッセージ(stderr)を PowerShell 5.1 がエラー扱いしないようにする
$ErrorActionPreference = "Continue"
function Invoke-Checked { & $args[0] $args[1..($args.Count - 1)]; if ($LASTEXITCODE) { throw "失敗: $args" } }

# uv の 3.12 ショートカット(ジャンクション)作成が失敗する環境があるため、実体のパスを直接使う
uv python install 3.12
$py = Get-ChildItem "$env:APPDATA\uv\python\cpython-3.12.*-windows-x86_64-none\python.exe" -ErrorAction SilentlyContinue |
    Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
if (-not $py) { throw "Python 3.12 が見つかりません" }

Invoke-Checked uv venv --allow-existing --python $py .venv
$venvPy = ".\.venv\Scripts\python.exe"

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "[setup] NVIDIA GPU を検出 -> CUDA 版 PyTorch"
    Invoke-Checked uv pip install --python $venvPy torch torchaudio --index-url https://download.pytorch.org/whl/cu128
} else {
    Write-Host "[setup] GPU なし -> CPU 版 PyTorch（かなり遅くなります。AivisSpeech の利用を推奨）"
    Invoke-Checked uv pip install --python $venvPy torch torchaudio
}

Invoke-Checked uv pip install --python $venvPy -r requirements.txt

# 既に CPU 版が入っていると上の install は何もしないので、GPU があるのに CUDA 無効なら入れ直す
if ((Get-Command nvidia-smi -ErrorAction SilentlyContinue) -and
    ((& $venvPy -c "import torch;print(torch.cuda.is_available())") -ne "True")) {
    Write-Host "[setup] CPU 版 PyTorch が入っているので CUDA 版に入れ直します"
    Invoke-Checked uv pip install --python $venvPy --reinstall-package torch --reinstall-package torchaudio torch torchaudio --index-url https://download.pytorch.org/whl/cu128
}

& $venvPy -c "import torch;print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"
Write-Host ""
Write-Host "[setup] 完了。次を試してください:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python tts.py speak `"こんにちは、ローカル音声合成のテストです。`""
Write-Host "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --port 8000   # Web UI"
