"""音声の生成（UI から呼ぶ）。文章→音声 は Qwen3-TTS、録音→声変換 は Seed-VC。"""

import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from studio.common import ROOT, SEEDVC, SPEAKER, VC_CONFIG, VC_PY, VOICES, child_env, ffmpeg, read_status

OUT = ROOT / "outputs"
_lock = threading.Lock()
_cache = {"key": None, "model": None}


def tts_checkpoints(voice: str) -> list[str]:
    return [p.name for p in sorted((VOICES / voice / "tts").glob("epoch-*"), reverse=True)]


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


def unload() -> None:
    """読み込んだモデルを GPU から外す（学習に GPU メモリを譲るため）。"""
    import gc

    import torch

    with _lock:
        _cache.update(key=None, model=None)
        gc.collect()
        torch.cuda.empty_cache()


def speak(voice: str, text: str, checkpoint: str | None, instruct: str = "") -> tuple[Path, str]:
    """学習済みなら学習した声で、未学習なら参照音声からのクローンで読み上げる。"""
    import soundfile as sf

    from tts import fetch_model

    vdir = VOICES / voice
    with _lock:
        t0 = time.time()
        if checkpoint:
            model = _load(("ft", voice, checkpoint), vdir / "tts" / checkpoint)
            wavs, sr = model.generate_custom_voice(text=text, language="Japanese", speaker=SPEAKER,
                                                   instruct=instruct or None)
            how = f"学習済みモデル {checkpoint}"
        else:
            if not (vdir / "ref.wav").exists():
                raise RuntimeError("まだ学習データの準備（または学習）をしていません")
            model = _load(("base",), fetch_model("Qwen/Qwen3-TTS-12Hz-1.7B-Base"))
            ref_text = (vdir / "ref.txt").read_text(encoding="utf-8")
            wavs, sr = model.generate_voice_clone(text=text, language="Japanese", ref_audio=str(vdir / "ref.wav"),
                                                  ref_text=ref_text)
            how = "未学習（参照音声からのクローン）"
        out = OUT / voice / f"tts_{time.strftime('%Y%m%d-%H%M%S')}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, wavs[0], sr)
    return out, f"{how} / {time.time() - t0:.1f}秒"


def convert(voice: str, source: str, steps: int = 30, auto_pitch: bool = True, semitone: int = 0) -> tuple[Path, str]:
    """録音を学習した声に変換する。抑揚（声の高さの動き）・間はそのまま残る。"""
    vdir = VOICES / voice
    ref = vdir / "ref_hq.wav"
    if not ref.exists():
        raise RuntimeError("まだ学習データの準備（または学習）をしていません")
    run = SEEDVC / "runs" / voice
    ckpt = run / "ft_model.pth"
    cmd = [str(VC_PY), str(ROOT / "studio" / "vc_run.py"), "inference.py", "--source", source, "--target", str(ref),
           "--diffusion-steps", str(steps), "--f0-condition", "True",
           "--auto-f0-adjust", str(auto_pitch), "--semi-tone-shift", str(semitone)]
    if ckpt.exists():
        cmd += ["--checkpoint", str(ckpt), "--config", str(run / Path(VC_CONFIG).name)]
        how = "学習済みモデル"
    else:
        how = "未学習（参照音声からのゼロショット変換）"
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        # ブラウザ録音（webm 等）でも読めるよう、先に wav へ変換しておく
        src_wav = Path(tmp) / "source.wav"
        subprocess.run([ffmpeg(), "-v", "error", "-y", "-i", str(Path(source).resolve()), "-ac", "1", "-ar", "44100",
                        str(src_wav)], check=True)
        cmd[cmd.index(source)] = str(src_wav)
        p = subprocess.run(cmd + ["--output", tmp], cwd=SEEDVC, env=child_env(),
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        wavs = list(Path(tmp).glob("*.wav"))
        if p.returncode != 0 or not wavs:
            raise RuntimeError("声変換に失敗しました:\n" + (p.stdout + p.stderr)[-1500:])
        out = OUT / voice / f"vc_{time.strftime('%Y%m%d-%H%M%S')}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(wavs[0]), out)
    return out, f"{how} / {time.time() - t0:.1f}秒"


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
