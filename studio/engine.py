"""音声の生成（UI から呼ぶ）。文章→音声 は Qwen3-TTS、録音→声変換 は Seed-VC。"""

import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from studio.common import ROOT, SEEDVC, SPEAKER, VC_CONFIG, VC_PY, VOICES, child_env, ffmpeg, read_status

OUT = ROOT / "outputs"
_lock = threading.Lock()
_cache = {"key": None, "model": None}


def tts_checkpoints(voice: str) -> list[str]:
    """新しい順。採点済みなら一番よく読めるもの（best.txt）を先頭にする。"""
    tdir = VOICES / voice / "tts"
    names = [p.name for p in sorted(tdir.glob("epoch-*"), reverse=True)]
    best = (tdir / "best.txt").read_text(encoding="utf-8").strip() if (tdir / "best.txt").exists() else None
    return [best] + [n for n in names if n != best] if best in names else names


def _load(key, path):
    import torch
    from qwen_tts import Qwen3TTSModel

    if _cache["key"] != key:
        _cache["model"] = None
        torch.cuda.empty_cache()
        _cache["model"] = Qwen3TTSModel.from_pretrained(
            str(path), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
        _cache["key"] = key
    return _cache["model"]


_vc = {"proc": None}
_vc_lock = threading.Lock()


def _vc_request(req: dict) -> None:
    """声変換の常駐係（studio/vc_server.py）に依頼して、終わるまで待つ。

    1 回目だけモデルの読み込みに時間がかかり、2 回目からはすぐに変換が始まる。
    常駐係が落ちていたら起動し直す。
    """
    import json

    with _vc_lock:
        p = _vc["proc"]
        if p is None or p.poll() is not None:
            p = subprocess.Popen(
                [str(VC_PY), str(ROOT / "studio" / "vc_run.py"), str(ROOT / "studio" / "vc_server.py")],
                cwd=SEEDVC, env=child_env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW)
            _vc["proc"] = p
        p.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        p.stdin.flush()
        tail = []
        for line in p.stdout:
            line = line.rstrip()
            if line == "@@DONE":
                return
            if line.startswith("@@ERR"):
                raise RuntimeError("声変換に失敗しました: " + line[5:] + "\n" + "\n".join(tail[-15:]))
            tail.append(line)
        _vc["proc"] = None
        raise RuntimeError("声変換の常駐係が止まりました:\n" + "\n".join(tail[-20:]))


def stop_vc_server() -> None:
    p = _vc["proc"]
    if p is not None and p.poll() is None:
        p.kill()
    _vc["proc"] = None


def unload() -> None:
    """読み込んだモデルを GPU から外す（学習に GPU メモリを譲るため）。声変換の常駐係も止める。"""
    import gc

    import torch

    stop_vc_server()
    with _lock:
        _cache.update(key=None, model=None)
        gc.collect()
        torch.cuda.empty_cache()


def _max_tokens(text: str) -> int:
    """生成の上限（12 トークン/秒）。日本語は 1 秒 6〜8 文字なので、その 2〜3 倍の余裕を持たせる。
    モデルが壊れて話し終わらなくなっても、延々と生成し続けないようにする。"""
    return max(96, int(len(text) * 3.5) + 48)


def speak(voice: str, text: str, checkpoint: str | None, instruct: str = "", sub: str = "tts") -> tuple[Path, str]:
    """学習済みなら学習した声で、未学習なら参照音声からのクローンで読み上げる。"""
    import soundfile as sf

    from tts import fetch_model

    vdir = VOICES / voice
    with _lock:
        t0 = time.time()
        if checkpoint:
            model = _load(("ft", voice, sub, checkpoint), vdir / sub / checkpoint)
            # 公式の学習コードは言語指定なし（Auto）の形式で学習するので、読み上げ時も Auto にそろえる。
            # "Japanese" を指定すると学習時と違う先頭トークンになり、話し終わらない・雑音になる
            wavs, sr = model.generate_custom_voice(text=text, language="Auto", speaker=SPEAKER,
                                                   instruct=instruct or None, max_new_tokens=_max_tokens(text))
            how = f"学習済みモデル {checkpoint}"
        else:
            if not (vdir / "ref.wav").exists():
                raise RuntimeError("まだ学習データの準備（または学習）をしていません")
            model = _load(("base",), fetch_model("Qwen/Qwen3-TTS-12Hz-1.7B-Base"))
            ref_text = (vdir / "ref.txt").read_text(encoding="utf-8")
            wavs, sr = model.generate_voice_clone(text=text, language="Japanese", ref_audio=str(vdir / "ref.wav"),
                                                  ref_text=ref_text, max_new_tokens=_max_tokens(text))
            how = "未学習（参照音声からのクローン）"
        out = OUT / voice / f"tts_{time.strftime('%Y%m%d-%H%M%S')}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, wavs[0], sr)
    return out, f"{how} / {time.time() - t0:.1f}秒"


SR_VC = 44100


def vc_reference(vdir: Path, target_sec: float = 12.0) -> Path:
    """声変換の参照音声（約 12 秒）。1 クリップ（約 8 秒）だけより、声の特徴をつかみやすい。

    Seed-VC は「参照 + 変換する区間」を 30 秒の窓で処理するので、参照は 12 秒程度に抑える。
    """
    import numpy as np
    import soundfile as sf

    ref = vdir / "ref_vc.wav"
    if ref.exists() and ref.stat().st_mtime >= (vdir / "ref_hq.wav").stat().st_mtime:
        return ref
    pieces, total = [], 0.0
    files = [vdir / "ref_hq.wav"] + sorted((vdir / "clips_hq").glob("*.wav"))
    used = set()
    for f in files:
        if f.name in used or total >= target_sec:
            continue
        y, sr = sf.read(f, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if not 3 <= len(y) / sr <= 10 or np.sqrt((y ** 2).mean()) < 0.02:  # 短すぎ・小さすぎる区間は使わない
            continue
        pieces += [y, np.zeros(int(0.2 * sr), dtype=np.float32)]
        total += len(y) / sr
        used.add(f.name)
    sf.write(ref, np.concatenate(pieces)[: int(14 * SR_VC)], SR_VC)
    return ref


def split_at_silence(y, sr, max_sec: float = 15.0):
    """無音のところで区切って max_sec 以下の区間にする（区切りは無音の真ん中）。"""
    import librosa

    iv = librosa.effects.split(y, top_db=35, frame_length=2048, hop_length=512)
    if len(iv) == 0:
        return [(0, len(y))]
    cuts = [0]
    seg_start = 0
    for (s0, e0), (s1, _) in zip(iv[:-1], iv[1:]):
        mid = (e0 + s1) // 2  # 声と声のあいだ（無音）の真ん中
        if s1 - seg_start > max_sec * sr and mid > seg_start:
            cuts.append(mid)
            seg_start = mid
    cuts.append(len(y))
    out = []
    for s0, e0 in zip(cuts[:-1], cuts[1:]):
        # 無音なしで max_sec を超える長い発話は、仕方なく等分する
        n = max(1, int(np.ceil((e0 - s0) / (max_sec * sr))))
        step = (e0 - s0) // n
        out += [(s0 + i * step, e0 if i == n - 1 else s0 + (i + 1) * step) for i in range(n)]
    return out


def convert(voice: str, source: str, steps: int = 50, auto_pitch: bool = True, semitone: int = 0) -> tuple[Path, str]:
    """録音を学習した声に変換する。抑揚（声の高さの動き）・間はそのまま残る。

    無音で 15 秒以下に区切って変換し、つなぎ直す（Seed-VC 内部の区切りで音が途切れるのを防ぐ）。
    """
    import soundfile as sf

    vdir = VOICES / voice
    if not (vdir / "ref_hq.wav").exists():
        raise RuntimeError("まだ学習データの準備（または学習）をしていません")
    ref = vc_reference(vdir)
    from studio.speakers import vc_model

    model = vc_model(voice)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # ブラウザ録音（webm 等）でも読めるよう、先に wav へ変換しておく
        src_wav = tmp / "source.wav"
        subprocess.run([ffmpeg(), "-v", "error", "-y", "-i", str(Path(source).resolve()), "-ac", "1", "-ar", str(SR_VC),
                        str(src_wav)], check=True)
        y, sr = sf.read(src_wav, dtype="float32")
        seg_dir, out_dir = tmp / "segments", tmp / "converted"
        seg_dir.mkdir()
        spans = split_at_silence(y, sr)
        for i, (s0, e0) in enumerate(spans):
            sf.write(seg_dir / f"{i:03d}.wav", y[s0:e0], sr)

        how = "学習済みモデル" if model else "未学習（参照音声からのゼロショット変換）"
        _vc_request({"segments": str(seg_dir), "target": str(ref), "out": str(out_dir), "steps": int(steps),
                     "auto_pitch": bool(auto_pitch), "semitone": int(semitone),
                     "checkpoint": str(model[0]) if model else None, "config": str(model[1]) if model else None})
        outs = [out_dir / f"{i:03d}.wav" for i in range(len(spans))]
        if not all(f.exists() for f in outs):
            raise RuntimeError("声変換に失敗しました（結果のファイルがありません）")

        # つなぎ目は無音の中なので、10ms だけ重ねてなめらかにつなぐ
        fade = int(0.01 * sr)
        joined = None
        for f in outs:
            z, zsr = sf.read(f, dtype="float32")
            if z.ndim > 1:
                z = z.mean(axis=1)
            if joined is None or len(z) <= fade or len(joined) <= fade:
                joined = z if joined is None else np.concatenate([joined, z])
                continue
            w = np.linspace(0, 1, fade, dtype=np.float32)
            joined = np.concatenate([joined[:-fade], joined[-fade:] * (1 - w) + z[:fade] * w, z[fade:]])
        out = OUT / voice / f"vc_{time.strftime('%Y%m%d-%H%M%S')}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, joined, zsr)
    return out, f"{how} / {len(spans)} 区間 / {time.time() - t0:.1f}秒"


def status_text(voice: str) -> str:
    st = read_status(VOICES / voice)
    if not st:
        return "まだ学習していません"
    lines = [st.get("message", "")]
    if st.get("progress") is not None and st.get("stage") not in ("done", "error"):
        lines.append(f"進み具合: {st['progress'] * 100:.0f}%" + (f"（残り約{st['eta_min']}分）" if st.get("eta_min") else ""))
    if st.get("error"):
        lines.append("⚠ " + st["error"])
    if st.get("minutes"):
        lines.append(f"学習データ: {st.get('clips')} クリップ / {st['minutes']}分")
    lines.append(f"更新: {st.get('updated', '')}")
    return "\n".join(l for l in lines if l)
