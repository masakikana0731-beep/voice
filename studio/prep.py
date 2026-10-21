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
import re
import shutil
import subprocess

import numpy as np
import soundfile as sf

from studio.common import AUDIO_EXT, VOICES, ffmpeg, write_status

HQ_SR, TTS_SR = 44100, 24000
MIN_SEC, MAX_SEC = 2.0, 10.0  # 長いと書き起こしが崩れやすい
ASR_MODEL = "openai/whisper-large-v3-turbo"


def load(path, sr):
    """ffmpeg で何の形式でも mono / float32 に読み込む（スマホの m4a なども可）。"""
    out = subprocess.run(
        [ffmpeg(), "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def bad_transcript(text: str, sec: float) -> str | None:
    """書き起こしの失敗（Whisper の幻覚）を見つける。理由を返す（問題なければ None）。

    失敗した書き起こしで学習すると「同じ音を繰り返して話し終わらない」モデルになる。
    """
    import zlib

    t = re.sub(r"\s", "", text)
    if not t:
        return "空"
    cps = len(t) / sec
    if cps > 12:
        return f"文字が多すぎる（{cps:.0f}文字/秒）"
    if cps < 1.5:
        return f"文字が少なすぎる（{cps:.1f}文字/秒）"
    if len(t) >= 20 and len(t.encode()) / len(zlib.compress(t.encode())) > 2.4:
        return "同じ言葉の繰り返し"
    if re.search(r"(.)\1{5,}", t):
        return "同じ文字の連続"
    return None


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
    ap.add_argument("--purity", type=float, default=0, help="（テスト機能）登録済みの声との近さがこれ未満のクリップを除外。0 で無効")
    ap.add_argument("--select", type=int, default=1,
                    help="1 なら素材を自動で選ぶ（別の声が混ざった・雑音の強いクリップを外す。24 分で 3〜4 分）")
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

    total = sum(d for _, d in clips)  # 区切った全体の長さ
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

    # 書き起こしに失敗したクリップは学習に使わない
    dropped = []
    if args.purity > 0:
        # （テスト機能）相手の声などが混ざったクリップを、話者の近さで見つけて除外する
        import subprocess

        from studio.common import ROOT, SEP_PY, child_env

        pj = vdir / "purity.json"
        subprocess.run([str(SEP_PY), "-m", "studio.purity", "clips", str(vdir / "clips_hq"),
                        "--ref", *[str(r) for r in raws], "-o", str(pj)], cwd=ROOT, env=child_env(), check=True)
        purity = json.loads(pj.read_text(encoding="utf-8"))
        for i, (n, _) in enumerate(clips):
            if texts[i] and purity.get(n, 1) < args.purity:
                dropped.append(f"{n}: 別の声が混ざっている（近さ {purity[n]:.2f}）")
                texts[i] = ""
    for i, ((n, d), t) in enumerate(zip(clips, texts)):
        why = bad_transcript(t, d)
        if why:
            dropped.append(f"{n}: {why} | {t[:40]}")
            texts[i] = ""
    if args.select:
        # 素材の自動選別（studio/select_clips.py）: 別の声が混ざったクリップ・雑音の強いクリップを外す。
        # 声変換は clips_hq/ のファイルをそのまま学習に使うので、ファイルごと clips_dropped/ へ移す
        from studio.select_clips import choose, score_clips

        write_status(vdir, stage="prep", message="素材を選んでいます（相手の声・雑音の強い部分を外す）")
        try:
            drop = choose(vdir, score_clips(vdir))
        except Exception as e:  # 採点役が動かなくても学習は続ける
            print(f"[prep] 素材の選別を飛ばしました: {e}", flush=True)
            drop = {}
        shutil.rmtree(vdir / "clips_dropped", ignore_errors=True)
        (vdir / "clips_dropped").mkdir()
        names = [n for n, _ in clips]
        for n, why in drop.items():
            dropped.append(f"{n}: {why}")
            texts[names.index(n)] = ""
            shutil.move(str(vdir / "clips_hq" / n), vdir / "clips_dropped" / n)
            (vdir / "clips" / n).unlink(missing_ok=True)
        print(f"[prep] 素材の選別で除外: {len(drop)} クリップ", flush=True)
    (vdir / "dropped.txt").write_text("\n".join(dropped), encoding="utf-8")
    print(f"[prep] 除外（書き起こし失敗・素材の選別）: 合計 {len(dropped)} クリップ", flush=True)

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
    used = sum(d for _, d in ok)
    write_status(vdir, stage="prep_done", clips=len(ok), minutes=round(used / 60, 1),
                 message=f"準備完了: {len(ok)} クリップ / {used / 60:.1f}分（書き起こし失敗 {len(dropped)} 件を除外）")
    print(f"[prep] 完了 {len(ok)} クリップ / {used / 60:.1f}分", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
