"""動画・音声・YouTube から声だけを抜き出し、話者ごとに分ける。.venv-sep で動かす。

  .venv-sep/Scripts/python.exe -m studio.separate 入力（ファイル or URL）... -o 出力フォルダ
        [--no-dereverb] [--no-denoise] [--speakers off|auto|1〜5]

工程:
  0. URL なら yt-dlp で音声をダウンロード
  1. ffmpeg で音声を取り出す（mp4 / mkv / mov / mp3 / m4a など何でも可）
  2. 声と背景音を分離   vocals_mel_band_roformer（Kim）     … audio-separator（UVR のモデル）
  3. 残響（エコー）除去  deverb_bs_roformer                  … --no-dereverb で省略
  4. 雑音除去            denoise_mel_band_roformer           … --no-denoise で省略
  5. 話者分け            ECAPA 話者埋め込み + クラスタリング … 全ファイルまとめて分けるので、
                         別々の動画に出ている同じ人は同じ「話者N」になる
出力: <名前>_voice.wav（全員分）、speakers/speaker_N.wav と speakers.json（話者ごと、長い順）
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from studio.common import ffmpeg  # noqa: E402

MODELS_DIR = ROOT / "models" / "separator"
STEPS = [
    # (工程名, モデル, 残す出力の名前)
    ("声と背景音の分離", "vocals_mel_band_roformer.ckpt", "vocals"),
    ("残響の除去", "deverb_bs_roformer_8_384dim_10depth.ckpt", "noreverb"),
    ("雑音の除去", "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt", "dry"),
]
SR = 44100
MAX_SPEAKERS = 5


STATUS = None  # 出力フォルダの status.json（UI が進み具合を読む）


def set_status(state, message):
    if STATUS:
        STATUS.write_text(json.dumps({"state": state, "message": message, "updated": time.strftime("%H:%M:%S")},
                                     ensure_ascii=False), encoding="utf-8")


def log(msg):
    print(msg, flush=True)
    set_status("running", msg.replace("[sep] ", "").replace("[done] ", "保存: "))


def download(url, tmp):
    """YouTube などの URL から音声だけをダウンロードする。"""
    import yt_dlp

    deno = sorted((Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages").glob("DenoLand.Deno*/deno.exe"))
    opts = {
        "format": "bestaudio/best", "outtmpl": str(tmp / "%(id)s.%(ext)s"), "noplaylist": True,
        "quiet": True, "no_warnings": True, "noprogress": True, "ffmpeg_location": str(Path(ffmpeg()).parent),
    }
    if deno:  # YouTube の再生情報の解読に JavaScript 実行環境が要る
        opts["js_runtimes"] = {"deno": {"path": str(deno[-1])}}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
    title = re.sub(r'[\\/:*?"<>|\s]+', "_", info.get("title") or info["id"])[:60]
    log(f"[sep] ダウンロード: {info.get('title')}（{info.get('duration', 0) // 60}分）")
    return path, title


def pick(files, stem):
    """分離結果のうち、名前に (stem) を含むものを返す。"""
    for f in files:
        if f"({stem})" in Path(f).name.lower():
            return Path(f)
    raise RuntimeError(f"出力 {stem} が見つかりません: {files}")


_ENC = None


def embed(segments):
    """16kHz の音声片のリスト → 正規化した話者埋め込み（ECAPA）。"""
    import numpy as np
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    global _ENC
    if _ENC is None:
        _ENC = EncoderClassifier.from_hparams(
            "speechbrain/spkrec-ecapa-voxceleb", savedir=str(MODELS_DIR / "ecapa"),
            run_opts={"device": "cuda:0" if torch.cuda.is_available() else "cpu"}, local_strategy=LocalStrategy.COPY)
    embs = []
    for k in range(0, len(segments), 64):
        chunk = [torch.from_numpy(np.ascontiguousarray(x)) for x in segments[k:k + 64]]
        lens = torch.tensor([len(c) for c in chunk], dtype=torch.float32)
        batch = torch.nn.utils.rnn.pad_sequence(chunk, batch_first=True)
        with torch.no_grad():
            embs.append(_ENC.encode_batch(batch, lens / lens.max()).squeeze(1).cpu().numpy())
    X = np.concatenate(embs)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def voice_print(paths, max_sec=180):
    """登録済みの声（複数ファイル可）の平均的な特徴。各ファイルの先頭 max_sec 秒を 3 秒ずつに切って使う。"""
    import librosa
    import numpy as np

    segs = []
    for p in paths:
        y, _ = librosa.load(str(p), sr=16000, mono=True, duration=max_sec)
        for s, e in librosa.effects.split(y, top_db=35):
            segs += [y[i:i + 48000] for i in range(s, e - 24000, 48000)]
    if not segs:
        return None
    c = embed(segs).mean(axis=0)
    return c / np.linalg.norm(c)


def diarize(wavs, out_dir, n_speakers, match=None):
    """声を話者ごとに分けて speaker_N.wav に書き出す。n_speakers=None なら人数を自動推定（最大 5）。"""
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from sklearn.cluster import AgglomerativeClustering
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    # 3 秒窓・1.5 秒ずらし（短すぎると話者の特徴が安定しない）
    win, hop = int(3.0 * 16000), int(1.5 * 16000)
    audios, spans = [], []  # spans: (ファイル番号, 発話番号, 16kHz の開始, 終了)
    for fi, w in enumerate(wavs):
        y, sr = sf.read(w, dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        audios.append(y)
        y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
        utts = []
        for s, e in librosa.effects.split(y16, top_db=35, frame_length=1024, hop_length=256):
            if utts and s - utts[-1][1] < 0.3 * 16000:
                utts[-1][1] = e
            else:
                utts.append([s, e])
        for ui, (s, e) in enumerate(utts):
            if e - s < 0.5 * 16000:
                continue
            starts = [s] if e - s <= win else list(range(s, e - win + 1, hop))
            for st in starts:
                spans.append((fi, ui, st, min(st + win, e), y16))
    if not spans:
        raise RuntimeError("声が見つかりませんでした")

    X = embed([sp[4][sp[2]:sp[3]] for sp in spans])

    if len(X) == 1:
        labels = np.zeros(1, dtype=int)
    elif n_speakers:
        labels = AgglomerativeClustering(n_clusters=min(n_speakers, len(X)), metric="cosine", linkage="average").fit_predict(X)
    else:
        labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                         distance_threshold=0.78).fit_predict(X)  # 同じ人 ≈0.3〜0.7、別人 ≈0〜0.25（コサイン類似度）
        if labels.max() + 1 > MAX_SPEAKERS:
            labels = AgglomerativeClustering(n_clusters=MAX_SPEAKERS, metric="cosine", linkage="average").fit_predict(X)

    # 合計 3 秒未満の小さな塊は誤検出とみなし、近い話者にまとめる
    seconds = {c: (labels == c).sum() * 1.5 for c in set(labels)}
    big = [c for c, s in seconds.items() if s >= 3] or [max(seconds, key=seconds.get)]
    cents = {c: X[labels == c].mean(axis=0) for c in big}
    for i, c in enumerate(labels):
        if c not in cents:
            labels[i] = max(cents, key=lambda b: X[i] @ cents[b])

    # 同じ発話の中で前後と違う 1 窓だけの揺れをならす
    for i in range(1, len(labels) - 1):
        same_utt = spans[i - 1][:2] == spans[i][:2] == spans[i + 1][:2]
        if same_utt and labels[i - 1] == labels[i + 1] != labels[i]:
            labels[i] = labels[i - 1]

    # 各窓の担当区間 = 隣の窓との中間点で区切る → 話者ごとに音声をつなげる
    ratio = SR / 16000
    pieces = {c: [] for c in set(labels)}
    for i, (fi, ui, st, en, _) in enumerate(spans):
        lo = st if i == 0 or spans[i - 1][:2] != (fi, ui) else (spans[i - 1][2] + spans[i - 1][3] + st + en) // 4
        hi = en if i == len(spans) - 1 or spans[i + 1][:2] != (fi, ui) else (st + en + spans[i + 1][2] + spans[i + 1][3]) // 4
        pieces[labels[i]].append(audios[fi][int(lo * ratio):int(hi * ratio)])

    out_dir.mkdir(parents=True, exist_ok=True)
    gap = np.zeros(int(0.25 * SR), dtype=np.float32)
    order = sorted(pieces, key=lambda c: -sum(len(p) for p in pieces[c]))
    result = []
    for k, c in enumerate(order, 1):
        audio = np.concatenate([x for p in pieces[c] for x in (p, gap)])
        path = out_dir / f"speaker_{k}.wav"
        sf.write(path, audio, SR, subtype="PCM_16")
        item = {"file": str(path), "seconds": round(len(audio) / SR, 1)}
        if match is not None:  # 登録済みの声との近さ（同じ人 ≈0.6 以上、別人 ≈0.3 以下が目安）
            cen = X[labels == c].mean(axis=0)
            item["similarity"] = round(float(cen @ match / np.linalg.norm(cen)), 3)
        result.append(item)
        log(f"[sep] 話者{k}: {len(audio) / SR:.0f}秒" + (f"（登録済みの声との近さ {item['similarity']:.2f}）" if match is not None else ""))
    (out_dir / "speakers.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    # audio-separator も内部で ffmpeg を呼ぶので PATH に載せる
    os.environ["PATH"] = f"{Path(ffmpeg()).parent};{os.environ['PATH']}"
    from audio_separator.separator import Separator

    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="ファイルのパス、または YouTube などの URL")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--no-dereverb", action="store_true")
    ap.add_argument("--no-denoise", action="store_true")
    ap.add_argument("--speakers", default="auto", help="off / auto / 1〜5")
    ap.add_argument("--match", nargs="*", default=[], help="登録済みの声のファイル。各話者がこの声にどれだけ近いかを出す")
    args = ap.parse_args()

    global STATUS
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    STATUS = out / "status.json"
    (out / "inputs.txt").write_text("\n".join(args.inputs), encoding="utf-8")
    set_status("running", "開始しました")

    steps = [s for s in STEPS if not (args.no_dereverb and s[2] == "noreverb") and not (args.no_denoise and s[2] == "dry")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # 0〜1. ダウンロード・音声の取り出し
        current, names = [], []
        for i, src in enumerate(args.inputs):
            if re.match(r"https?://", src):
                src, name = download(src, tmp)
            else:
                name = Path(src).stem
            wav = tmp / f"{i:03d}.wav"
            subprocess.run([ffmpeg(), "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "2", "-ar", str(SR), str(wav)],
                           check=True)
            current.append(wav)
            names.append(name)
        log(f"[sep] 音声を取り出しました: {len(current)} 本")

        # 2〜4. モデルを 1 つずつ読み込み、全ファイルに順にかける（読み込みは工程ごとに 1 回）
        for n, (label, model, stem) in enumerate(steps, 1):
            t0 = time.time()
            step_dir = tmp / f"step{n}"
            sep = Separator(model_file_dir=str(MODELS_DIR), output_dir=str(step_dir), output_format="WAV")
            sep.load_model(model_filename=model)
            current = [pick([step_dir / Path(f).name for f in sep.separate(str(w))], stem) for w in current]
            del sep
            log(f"[sep] {n}/{len(steps)} {label} 完了（{time.time() - t0:.0f}秒）")

        outs = []
        for name, wav in zip(names, current):
            dst = out_dir / f"{name}_voice.wav"
            shutil.move(str(wav), dst)
            outs.append(dst)
            log(f"[done] {dst}")

    # 5. 話者分け
    if args.speakers != "off":
        t0 = time.time()
        match = voice_print(args.match) if args.match else None
        diarize(outs, out_dir / "speakers", None if args.speakers == "auto" else int(args.speakers), match)
        log(f"[sep] 話者分け完了（{time.time() - t0:.0f}秒）")
    return 0


if __name__ == "__main__":
    try:
        code = main()
        set_status("done", "完了しました")
    except Exception as e:  # UI に失敗の理由を見せる
        import traceback

        traceback.print_exc()
        set_status("error", f"失敗しました: {e}")
        code = 1
    raise SystemExit(code)
