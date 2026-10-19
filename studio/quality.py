"""作った声の「出来」を数字にする共通の道具（.venv で動かす）。精度検証（bench.py）と学習後の採点（eval_tts.py）で使う。

  - 読み間違い（CER）: Whisper で書き起こし、元の文とひらがなに直して比べる（漢字・かなの書き方の違いを無視）
  - 似ている度（sim）: 本物の声の特徴（ECAPA）とのコサイン類似度。同じ人 ≈0.6 以上
  - 自然さ（mos）: UTMOSv2 の予想点 1〜5（英語中心の AI なので、同じ条件での比べっこに使う）
  - 総合点（total）: 本物の録音を 100 としたときの点数
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path

from studio.common import ROOT, SEP_PY, child_env

ASR_MODEL = "openai/whisper-large-v3-turbo"
_asr = {}
_kks = {}


def kana(text: str) -> str:
    """ひらがなに直し、記号・空白を消す（「本当」と「ほんとう」を同じに扱うため）。"""
    import pykakasi

    if "k" not in _kks:
        _kks["k"] = pykakasi.kakasi()
    hira = "".join(x["hira"] for x in _kks["k"].convert(text))
    return re.sub(r"[\s、。，．,.!！?？「」『』…ー〜~・（）()]", "", hira)


def cer(ref: str, hyp: str) -> float:
    r, h = kana(ref), kana(hyp)
    d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, b in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a != b))
    return min(1.0, d[-1] / max(len(r), 1))


def transcribe(paths) -> list[str]:
    import librosa
    import torch
    from transformers import pipeline

    from tts import fetch_model

    if "p" not in _asr:
        _asr["p"] = pipeline("automatic-speech-recognition", model=fetch_model(ASR_MODEL), dtype=torch.float16,
                             device="cuda:0")
    out = []
    for p in paths:
        y, _ = librosa.load(str(p), sr=16000, mono=True)
        out.append(_asr["p"]({"raw": y, "sampling_rate": 16000}, generate_kwargs={"language": "japanese"},
                             return_timestamps=True)["text"].strip())
    return out


def unload_asr() -> None:
    import gc

    import torch

    _asr.clear()
    gc.collect()
    torch.cuda.empty_cache()


def judge(paths, refs, mos: bool = True) -> dict:
    """似ている度と自然さ（studio/judge.py を .venv-sep で）。{ファイル名: {"sim", "mos"}}
    ファイル名が重なると上書きされるので、呼び出し側で名前を分けておく。"""
    with tempfile.TemporaryDirectory() as tmp:
        res, lst = Path(tmp) / "judge.json", Path(tmp) / "paths.txt"
        # ファイルが多いとコマンドの長さの上限（約 3 万 2 千文字）を超えるので、一覧ファイルで渡す
        lst.write_text("\n".join(str(Path(p).resolve()) for p in paths), encoding="utf-8")
        cmd = [str(SEP_PY), "-m", "studio.judge", "--list", str(lst),
               "--ref", *[str(Path(r).resolve()) for r in refs], "-o", str(res)] + ([] if mos else ["--no-mos"])
        r = subprocess.run(cmd, cwd=ROOT, env=child_env(), capture_output=True, text=True, encoding="utf-8",
                           errors="replace", creationflags=subprocess.CREATE_NO_WINDOW)
        if r.returncode != 0:
            raise RuntimeError("採点に失敗しました:\n" + r.stdout[-1500:] + r.stderr[-1500:])
        return json.loads(res.read_text(encoding="utf-8"))


def total(m: dict, base: dict) -> float:
    """本物の録音（base）を 100 とした総合点。似ている度 5 割・自然さ 3 割・読み間違いの少なさ 2 割。

    似ている度は別人でも 0.3〜0.4 出るので、別人の声（base["sim_other"]、なければ 0.35）を 0、本物を満点にする。
    自然さは 1（最低）を 0、本物を満点にする。読み間違いは本物より 30 ポイント多いと 0。"""
    floor = base.get("sim_other", 0.35)
    s = min(1.0, max(0.0, (m["sim"] - floor) / max(1e-6, base["sim"] - floor)))
    if m.get("mos") is not None and base.get("mos"):
        q = min(1.0, max(0.0, (m["mos"] - 1) / max(1e-6, base["mos"] - 1)))
    else:
        q = s
    c = max(0.0, 1 - max(0.0, m["cer"] - base["cer"]) / 0.3)
    return round(100 * (0.5 * s + 0.3 * q + 0.2 * c), 1)


def mean(rows: list[dict], key: str):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None
