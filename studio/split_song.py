"""曲を歌声と伴奏に分ける（.venv-sep で動かす）。歌声変換で使う。

  python -m studio.split_song 曲のファイル -o 出力フォルダ   → vocals.wav / accompaniment.wav
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from studio.common import ffmpeg  # noqa: E402
from studio.separate import MODELS_DIR  # noqa: E402

MODEL = "vocals_mel_band_roformer.ckpt"


def main() -> int:
    os.environ["PATH"] = f"{Path(ffmpeg()).parent};{os.environ['PATH']}"
    from audio_separator.separator import Separator

    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wav = tmp / "song.wav"
        subprocess.run([ffmpeg(), "-v", "error", "-y", "-i", a.source, "-vn", "-ac", "2", "-ar", "44100", str(wav)],
                       check=True)
        sep = Separator(model_file_dir=str(MODELS_DIR), output_dir=str(tmp), output_format="WAV")
        sep.load_model(model_filename=MODEL)
        files = [tmp / Path(f).name for f in sep.separate(str(wav))]
        for f in files:
            name = f.name.lower()
            if "(vocals)" in name:
                shutil.move(str(f), out / "vocals.wav")
            elif "(other)" in name or "(instrumental)" in name:
                shutil.move(str(f), out / "accompaniment.wav")
    print("[split] 完了", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
