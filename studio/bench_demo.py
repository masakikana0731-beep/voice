"""精度検証のデモ音声: 同じ文を、レベルの低い順に並べて 1 本にする（各レベルの前に内蔵の声で紹介）。

  python -m studio.bench_demo   → bench/demo/レベル別デモ.mp3（と各レベルの mp3）

素材は精度検証（studio/bench.py）で作って保存してある音声（bench/runs/）。JSUT の音声を含むので個人で聞く用。
"""

import subprocess

import librosa
import numpy as np
import soundfile as sf
import torch

from studio import bench, engine
from studio.common import ffmpeg

SR = 24000
SENTENCES = ["003.wav", "007.wav"]  # 「あいにくですが、本日は予約でいっぱいです。」ほか
# (紹介の文, 実験フォルダ, ラベル)
LEVELS = [
    ("レベル1。修正前のプログラムで、学習率を上げすぎて壊れたモデルです。0点。",
     "tts_d24_lr1e-05_const_spk0_sub0.3_ga4_e4", "epoch-002"),
    ("レベル2。修正前のプログラムで、一番良かったモデルです。83点。",
     "tts_d24_lr2e-06_const_spk0_sub0.3_ga4_e4", "epoch-004"),
    ("レベル3。学習なし。参照音声を聞かせただけのクローンです。84点。", "tts_zero", "zero"),
    ("レベル4。素材10分で学習。修正後のプログラムです。91点。",
     "tts_d10_lr1e-05_const_spk0_sub0.3_ga8_fix_e4", "epoch-002"),
    ("レベル5。素材24分、一番良い設定で学習。94.5点。",
     "tts_d24_lr1e-05_const_spk0_sub0.3_ga8_fix_e4", "epoch-003"),
    ("レベル6。本物の録音です。100点。", "real", "real"),
]


def load(path):
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y / (np.abs(y).max() + 1e-9) * 0.8


def gap(sec):
    return np.zeros(int(sec * SR), dtype=np.float32)


def main():
    from tts import fetch_model

    out = bench.BENCH / "demo"
    out.mkdir(exist_ok=True)
    model = engine._load(("custom",), fetch_model("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"))

    def say(text):
        torch.manual_seed(0)
        w, sr = model.generate_custom_voice(text=text, language="Japanese", speaker="Ono_Anna",
                                            max_new_tokens=engine._max_tokens(text))
        return load_arr(w[0], sr)

    def load_arr(w, sr):
        y = librosa.resample(np.asarray(w, dtype=np.float32), orig_sr=sr, target_sr=SR)
        return y / (np.abs(y).max() + 1e-9) * 0.6

    whole = [say("精度検証のデモです。同じ文を、レベルの低い順に聞いてください。"), gap(0.8)]
    for i, (intro, exp, label) in enumerate(LEVELS, 1):
        clips = [load(bench.RUNS / exp / f"{label}__{s}") for s in SENTENCES]
        part = []
        for c in clips:
            part += [c, gap(0.6)]
        part = np.concatenate(part)
        whole += [say(intro), gap(0.5), part, gap(0.8)]
        f = out / f"level{i}.wav"
        sf.write(f, part, SR)
        subprocess.run([ffmpeg(), "-v", "error", "-y", "-i", str(f), "-b:a", "160k", str(f.with_suffix(".mp3"))],
                       check=True)
        f.unlink()
    engine.unload()
    f = out / "demo.wav"
    sf.write(f, np.concatenate(whole), SR)
    mp3 = out / "レベル別デモ.mp3"
    subprocess.run([ffmpeg(), "-v", "error", "-y", "-i", str(f), "-b:a", "160k", str(mp3)], check=True)
    f.unlink()
    print(mp3)


if __name__ == "__main__":
    main()
