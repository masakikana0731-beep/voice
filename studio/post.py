"""仕上げ（後処理）: リップノイズ・響き（リバーブ）・エコー・環境（部屋や電話など）・音量そろえ。

  apply(wav_path, preset="部屋", reverb=0, echo=0, ambience=0, lips=30) → 仕上げた wav のパス

スライダーの値はすべて 0〜100。プリセットの上に、スライダーの分を足す。
"""

from pathlib import Path

import numpy as np
import soundfile as sf

PRESETS = ["なし", "部屋", "スタジオ", "ホール", "お風呂", "電話", "ラジオ", "屋外", "やまびこ"]


def _preset_chain(name: str):
    from pedalboard import (Compressor, Delay, Distortion, Gain, HighpassFilter, LowpassFilter,
                            PeakFilter, Reverb)

    return {
        "なし": [],
        "部屋": [Reverb(room_size=0.25, damping=0.5, wet_level=0.12, dry_level=0.9, width=0.8)],
        "スタジオ": [HighpassFilter(80), Compressor(threshold_db=-20, ratio=3, attack_ms=5, release_ms=80),
                    PeakFilter(4500, gain_db=2, q=0.8),
                    Reverb(room_size=0.12, damping=0.6, wet_level=0.05, dry_level=0.95)],
        "ホール": [Reverb(room_size=0.85, damping=0.35, wet_level=0.3, dry_level=0.8, width=1.0)],
        "お風呂": [Reverb(room_size=0.55, damping=0.05, wet_level=0.35, dry_level=0.75),
                  Delay(delay_seconds=0.045, feedback=0.25, mix=0.12)],
        "電話": [HighpassFilter(320), LowpassFilter(3300), Distortion(drive_db=8),
                Compressor(threshold_db=-24, ratio=6), Gain(-4)],
        "ラジオ": [HighpassFilter(120), LowpassFilter(9000),
                  Compressor(threshold_db=-18, ratio=4, attack_ms=3, release_ms=60), PeakFilter(2500, gain_db=3)],
        "屋外": [Reverb(room_size=0.05, damping=0.9, wet_level=0.03, dry_level=1.0)],
        "やまびこ": [Delay(delay_seconds=0.38, feedback=0.4, mix=0.3),
                   Reverb(room_size=0.7, damping=0.4, wet_level=0.12, dry_level=0.9)],
    }[name]


def _lip_noise(y, sr, amount, rng):
    """話し始める直前に、口を開く小さな音（リップノイズ）を足す。"""
    import librosa
    from scipy.signal import butter, sosfilt

    if amount <= 0:
        return y
    sos = butter(2, [1200, 6000], btype="band", fs=sr, output="sos")
    level = np.sqrt(np.mean(y ** 2) + 1e-9) * (0.02 + 0.25 * amount / 100)  # 声の大きさに合わせる
    out = y.copy()
    prev_end = 0
    for s, e in librosa.effects.split(y, top_db=35, frame_length=1024, hop_length=256):
        if s - prev_end > 0.3 * sr and rng.random() < 0.35 + 0.6 * amount / 100:
            for _ in range(rng.integers(1, 3)):  # 1〜2 回の「プチッ」
                n = int(rng.uniform(0.003, 0.009) * sr)
                click = rng.standard_normal(n) * np.exp(-np.linspace(0, rng.uniform(4, 9), n))
                click = sosfilt(sos, click)
                click *= level / (np.abs(click).max() + 1e-9)
                pos = s - int(rng.uniform(0.04, 0.16) * sr)
                if pos > 0:
                    out[pos:pos + n] += click[: len(out) - pos]
        prev_end = e
    return out


def _ambience(n, sr, amount, rng, kind):
    """環境音（部屋の空気音・屋外のざわめき）。ピンクノイズを加工して作る。"""
    if amount <= 0:
        return 0
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1 / sr)
    spec /= np.sqrt(np.maximum(f, 20))  # ピンクノイズ（低い音ほど強い）
    if kind == "屋外":
        spec *= 1 + 0.5 * np.exp(-((f - 300) / 250) ** 2)  # 遠くのざわめき
    noise = np.fft.irfft(spec, n)
    noise /= np.abs(noise).max() + 1e-9
    return noise * (10 ** ((-60 + 30 * amount / 100) / 20))  # 0→ -60dB, 100→ -30dB


def clean(wav) -> Path:
    """AI クリーンアップ（studio/cleanup.py を .venv-sep で実行）。変換で出たザラつき・雑音を取る。"""
    import subprocess

    from studio.common import ROOT, SEP_PY, child_env

    out = Path(wav).with_name(Path(wav).stem + "_clean.wav")
    subprocess.run([str(SEP_PY), "-m", "studio.cleanup", str(Path(wav).resolve()), str(out)],
                   cwd=ROOT, env=child_env(), check=True, capture_output=True)
    return out


def apply(wav, preset="なし", reverb=0, echo=0, ambience=0, lips=0, seed=0, ai_clean=False) -> Path:
    from pedalboard import Delay, Pedalboard, Reverb

    if ai_clean:  # 響きなどを足す前に、変換で出た雑音を取っておく
        wav = clean(wav)
    y, sr = sf.read(wav, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    rng = np.random.default_rng(seed)
    y = _lip_noise(y, sr, lips, rng)

    chain = _preset_chain(preset)
    if reverb > 0:
        chain.append(Reverb(room_size=0.3 + 0.6 * reverb / 100, damping=0.4,
                            wet_level=0.35 * reverb / 100, dry_level=1 - 0.25 * reverb / 100))
    if echo > 0:
        chain.append(Delay(delay_seconds=0.12 + 0.3 * echo / 100, feedback=0.15 + 0.35 * echo / 100,
                           mix=0.35 * echo / 100))
    # 響きやエコーの余韻が切れないよう、後ろに少し無音を足してから処理する
    tail = int((1.5 if chain else 0) * sr)
    y = np.concatenate([y, np.zeros(tail, dtype=np.float32)])
    if chain:
        y = Pedalboard(chain)(y, sr)
    amb_kind = "屋外" if preset == "屋外" else "部屋"
    y = y + _ambience(len(y), sr, ambience + (25 if preset == "屋外" else 0), rng, amb_kind)
    # 音量をそろえ（平均 -22dB 前後）、大きすぎる山だけをやわらかく抑える（最大 -1dB）
    rms = np.sqrt(np.mean(y ** 2) + 1e-12)
    y = y * (0.08 / rms)
    knee, room = 0.7, 0.19
    a = np.abs(y)
    y = np.where(a > knee, np.sign(y) * (knee + room * np.tanh((a - knee) / room)), y).astype(np.float32)
    # 末尾の無音を切りそろえる
    nz = np.nonzero(np.abs(y) > 1e-3)[0]
    if len(nz):
        y = y[: min(len(y), nz[-1] + int(0.3 * sr))]
    import time
    # 設定を変えてかけ直したとき、ブラウザが古い音を出さないよう毎回ちがう名前にする
    out = Path(wav).with_name(f"{Path(wav).stem}_{preset}_{time.strftime('%H%M%S')}.wav")
    sf.write(out, y, sr)
    return out
