"""Seed-VC のスクリプトを .venv-vc で動かすための小さな入口。

新しい torchaudio は save/load に torchcodec が必要なので、soundfile で置き換えてから実行する。
  .venv-vc/Scripts/python.exe studio/vc_run.py inference.py --source ...   （cwd は third_party/seed-vc）
"""

import runpy
import sys

import soundfile as sf
import torch
import torchaudio


def _save(path, wav, sr, *a, **kw):
    sf.write(str(path), wav.detach().cpu().numpy().T, sr)


def _load(path, *a, **kw):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


torchaudio.save = _save
torchaudio.load = _load

sys.path.insert(0, ".")
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
