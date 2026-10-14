"""学習データの準備: raw/ の録音を無音で区切り、書き起こして clips.jsonl を作る。

  python -m studio.prep <声の名前>

出力（voices/<名前>/ 以下）:
  clips_hq/*.wav  44.1kHz（声変換 Seed-VC の学習用）
  clips/*.wav     24kHz （Qwen3-TTS の学習用）
  clips.jsonl     {"audio", "text", "ref_audio"} の行
  ref.wav / ref_hq.wav  代表の参照音声
"""

import argparse
import json
import shutil
import subprocess

import numpy as np
import soundfile as sf

from studio.common import AUDIO_EXT, VOICES, ffmpeg, write_status

HQ_SR, TTS_SR = 44100, 24000
MIN_SEC, MAX_SEC = 2.0, 15.0
ASR_MODEL = "openai/whisper-large-v3-turbo"


def load(path, sr):
    """ffmpeg で何の形式でも mono / float32 に読み込む（スマホの m4a なども可）。"""
    out = subprocess.run(
        [ffmpeg(), "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def segments(y, sr):
    """無音で区切った区間を MIN_SEC〜MAX_SEC の塊にまとめる。"""
    import librosa

    pad = int(0.15 * sr)
    # 0.3 秒未満の息継ぎではつなげる（文の途中で切らないため）
    utts = []
    for s, e in librosa.effects.split(y, top_db=35, frame_length=2048, hop_length=512):
        if utts and s - utts[-1][1] < 0.3 * sr:
            utts[-1][1] = e
        else:
            utts.append([s, e])

    chunks, start, end = [], None, None
    for s, e in utts:
        # 区切りなしで長すぎる発話は MAX_SEC ごとに切る
        while e - s > MAX_SEC * sr:
            chunks.append((s, s + int(MAX_SEC * sr)))
            s += int(MAX_SEC * sr)
        if start is not None and e - start <= MAX_SEC * sr:
            end = e
            continue
        if start is not None and end - start >= MIN_SEC * sr:
            chunks.append((start, end))
        start, end = s, e
    if start is not None and end - start >= MIN_SEC * sr:
        chunks.append((start, end))
    return [(max(0, s - pad), min(len(y), e + pad)) for s, e in chunks]


def main() -> int:
    import librosa
    import torch
    from transformers import pipeline

    from tts import fetch_model

    ap = argparse.ArgumentParser()
    ap.add_argument("voice")
    ap.add_argument("--language", default="japanese")
    args = ap.parse_args()

    vdir = VOICES / args.voice
    raws = sorted(p for p in (vdir / "raw").iterdir() if p.suffix.lower() in AUDIO_EXT)
    if not raws:
        raise SystemExit(f"{vdir / 'raw'} に録音がありません")
    for d in ("clips", "clips_hq"):
        shutil.rmtree(vdir / d, ignore_errors=True)
        (vdir / d).mkdir()

    write_status(vdir, stage="prep", message="録音を区切っています")
    clips = []
    for i, raw in enumerate(raws):
        y = load(raw, HQ_SR)
        peak = np.abs(y).max()
        if peak > 0.95:
            y = y * (0.95 / peak)
        for j, (s, e) in enumerate(segments(y, HQ_SR)):
            name = f"{i:03d}_{j:04d}.wav"
            seg = y[s:e]
            sf.write(vdir / "clips_hq" / name, seg, HQ_SR, subtype="PCM_16")
            sf.write(vdir / "clips" / name, librosa.resample(seg, orig_sr=HQ_SR, target_sr=TTS_SR), TTS_SR,
                     subtype="PCM_16")
            clips.append((name, (e - s) / HQ_SR))
        print(f"[prep] {raw.name}: 累計 {len(clips)} クリップ", flush=True)

    total = sum(d for _, d in clips)
    write_status(vdir, stage="prep", message=f"{len(clips)} クリップ（{total / 60:.1f}分）を書き起こし中")
    asr = pipeline(
        "automatic-speech-recognition", model=fetch_model(ASR_MODEL),
        torch_dtype=torch.float16, device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    paths = [str(vdir / "clips" / n) for n, _ in clips]
    texts = []
    for k in range(0, len(paths), 16):
        # ファイル名で渡すと transformers が ffmpeg を探しに行くので、波形で渡す
        batch = [{"raw": sf.read(p, dtype="float32")[0], "sampling_rate": TTS_SR} for p in paths[k:k + 16]]
        res = asr(batch, batch_size=16,
                  generate_kwargs={"language": args.language, "task": "transcribe"})
        texts += [r["text"].strip() for r in res]
        print(f"[prep] 書き起こし {len(texts)}/{len(paths)}", flush=True)
        write_status(vdir, stage="prep", message=f"書き起こし {len(texts)}/{len(paths)}")

    # 参照音声: 8 秒に一番近い、書き起こしのあるクリップ
    ok = [(n, d) for (n, d), t in zip(clips, texts) if t]
    ref = min(ok, key=lambda c: abs(c[1] - 8.0))[0]
    shutil.copy(vdir / "clips" / ref, vdir / "ref.wav")
    shutil.copy(vdir / "clips_hq" / ref, vdir / "ref_hq.wav")
    ref_text = texts[[n for n, _ in clips].index(ref)]
    (vdir / "ref.txt").write_text(ref_text, encoding="utf-8")

    with open(vdir / "clips.jsonl", "w", encoding="utf-8") as f:
        for (n, _), t in zip(clips, texts):
            if t:
                f.write(json.dumps({"audio": str(vdir / "clips" / n), "text": t,
                                    "ref_audio": str(vdir / "ref.wav")}, ensure_ascii=False) + "\n")
    write_status(vdir, stage="prep_done", clips=len(ok), minutes=round(total / 60, 1),
                 message=f"準備完了: {len(ok)} クリップ / {total / 60:.1f}分")
    print(f"[prep] 完了 {len(ok)} クリップ / {total / 60:.1f}分", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
