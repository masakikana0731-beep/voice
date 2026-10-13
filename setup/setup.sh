#!/usr/bin/env bash
# Qwen3-TTS ローカル環境セットアップ（macOS / Linux / WSL）
# 使い方: bash setup/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] uv をインストールします"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[setup] NVIDIA GPU を検出 → CUDA 版 PyTorch"
  uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
else
  echo "[setup] GPU なし（または Apple Silicon）→ 標準版 PyTorch"
  uv pip install torch torchaudio
fi

uv pip install -r requirements.txt

python -c "import torch;print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '| mps:', torch.backends.mps.is_available())"
echo
echo "[setup] 完了。次を試してください:"
echo "  source .venv/bin/activate"
echo "  python tts.py speak \"こんにちは、ローカル音声合成のテストです。\""
echo "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --port 8000   # Web UI"
