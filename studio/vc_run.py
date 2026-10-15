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
if script.endswith("real-time-gui.py"):
    # リアルタイム変換の画面は日本語に差し替えてから動かす（元のファイルは書き換えない）
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rt_ja import patch

    path = Path(script).resolve()
    code = compile(patch(path.read_text(encoding="utf-8")), str(path), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(path)})
else:
    runpy.run_path(script, run_name="__main__")
