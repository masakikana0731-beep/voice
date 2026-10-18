"""AI クリーンアップ（.venv-sep で動かす）: 変換後の音に残るシャリシャリ・ザラつきなどの雑音を取る。

  python -m studio.cleanup 入力.wav 出力.wav

audio-separator の雑音除去モデル（抽出で使っているものと同じ）で、声（dry）だけを残す。
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from studio.common import ffmpeg  # noqa: E402
from studio.separate import MODELS_DIR  # noqa: E402

MODEL = "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"


def main() -> int:
    os.environ["PATH"] = f"{Path(ffmpeg()).parent};{os.environ['PATH']}"
    from audio_separator.separator import Separator

    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        sep = Separator(model_file_dir=str(MODELS_DIR), output_dir=tmp, output_format="WAV")
        sep.load_model(model_filename=MODEL)
        files = [Path(tmp) / Path(f).name for f in sep.separate(a.src)]
        dry = next(f for f in files if "(dry)" in f.name.lower())
        shutil.move(str(dry), a.dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
