"""読み上げの改善の実験: 何回か作って選ぶ・話す速さ。

  python -m studio.tts_lab <話者> [--n 4]   → bench/tts_lab_<話者>.json と表

  - 同じ文を n 個まとめて作り（まとめて作ると 1 個とほぼ同じ時間）、1 個だけ・2〜n 個から選んだ場合を比べる。
    選び方は本番と同じ「読み間違いが少ないもののうち、本人の声にいちばん似ているもの」
  - 話す速さ: 本人の素材（書き起こしのひらがな文字数 / 声の長さ）と読み上げの速さを比べる
"""

import argparse
import json
import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from studio import bench, engine, quality
from studio.common import SPEAKER, VOICES

SENTENCES = [
    "みなさん、こんにちは。今日もラジオを聴いてくれてありがとうございます。",
    "最近すっかり寒くなってきましたね。風邪には気をつけてください。",
    "このあいだ、久しぶりに友達とカフェに行ってきたんですよ。",
    "それがもう、すっごく美味しくて、思わず二つも食べちゃいました。",
    "今日は、みなさんからいただいたメッセージを紹介していきますね。",
    "えっ、本当ですか？それはびっくりですね。",
    "来週も、この時間にお会いしましょう。お相手は、あゆみんでした。",
    "週末は、ゆっくりお家で映画を観て過ごす予定です。",
]
CER_OK = 0.15  # これより読み間違いが多い候補は選ばない


def speech_sec(y, sr) -> float:
    import librosa

    iv = librosa.effects.split(y, top_db=35)
    return sum(e - s for s, e in iv) / sr if len(iv) else len(y) / sr


def pick(cands: list[dict]) -> dict:
    """本番と同じ選び方: 読み間違いが CER_OK 以下のうち似ている度が最大。全部だめなら読み間違いが最小。"""
    ok = [c for c in cands if c["cer"] <= CER_OK]
    return max(ok, key=lambda c: c["sim"]) if ok else min(cands, key=lambda c: c["cer"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("voice")
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    sid, vdir = a.voice, VOICES / a.voice
    ck = engine.tts_checkpoints(sid)[0]
    work = bench.BENCH / f"tts_lab_{sid}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    model = engine._load(("ft", sid, "tts", ck), vdir / "tts" / ck)
    files, meta = [], {}
    t_single = t_batch = 0.0
    for i, text in enumerate(SENTENCES):
        torch.manual_seed(i)
        t0 = time.time()
        model.generate_custom_voice(text=text, language="Auto", speaker=SPEAKER, max_new_tokens=engine._max_tokens(text))
        t_single += time.time() - t0
        t0 = time.time()
        wavs, sr = model.generate_custom_voice(text=[text] * a.n, language=["Auto"] * a.n, speaker=[SPEAKER] * a.n,
                                               max_new_tokens=engine._max_tokens(text))
        t_batch += time.time() - t0
        for k, w in enumerate(wavs):
            f = work / f"s{i}__k{k}.wav"
            sf.write(f, w, sr)
            files.append(f)
            meta[f.name] = {"i": i, "k": k, "sec": speech_sec(np.asarray(w, dtype=np.float32), sr)}
        print(f"[lab] {i + 1}/{len(SENTENCES)} 生成", flush=True)
    engine.unload()

    hyps = dict(zip((f.name for f in files), quality.transcribe(files)))
    quality.unload_asr()
    refs = sorted((vdir / "raw").glob("*.wav"))
    reals = random.Random(0).sample(sorted((vdir / "clips_hq").glob("*.wav")), 8)
    with tempfile.TemporaryDirectory() as tmp:
        for k, f in enumerate(reals):
            shutil.copy(f, Path(tmp) / f"real__{k}.wav")
        j = quality.judge(files + sorted(Path(tmp).glob("*.wav")), refs, print_cache=vdir / "voiceprint.npy")
    real = [v for k, v in j.items() if k.startswith("real__")]
    base = {"cer": 0.03, "sim": quality.mean(real, "sim"), "mos": quality.mean(real, "mos")}
    cands = {}
    for f in files:
        m = meta[f.name]
        c = {**j[f.name], "cer": quality.cer(SENTENCES[m["i"]], hyps[f.name]),
             "kps": len(quality.kana(SENTENCES[m["i"]])) / max(m["sec"], 0.1)}
        cands.setdefault(m["i"], []).append(c)
    rows = {}
    for n in range(1, a.n + 1):
        chosen = [pick(cs[:n]) for cs in cands.values()]
        r = {k: quality.mean(chosen, k) for k in ("sim", "mos", "cer", "kps")}
        r["total"] = quality.total(r, base)
        rows[f"{n} 個から選ぶ" if n > 1 else "1 個だけ（今のやり方）"] = r

    # 本人の話す速さ（素材の書き起こしのひらがな文字数 / 声の長さ）
    kps_real = []
    for line in (vdir / "clips.jsonl").read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        y, sr = sf.read(c["audio"], dtype="float32")
        kps_real.append(len(quality.kana(c["text"])) / max(speech_sec(y, sr), 0.1))
    out = {"base": base, "rows": rows, "kps_real": float(np.median(kps_real)),
           "time": {"single_sec": round(t_single, 1), "batch_sec": round(t_batch, 1), "n": a.n}}
    (bench.BENCH / f"tts_lab_{sid}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[lab] 本物: 似ている度 {base['sim']} / 自然さ {base['mos']}", flush=True)
    for label, r in rows.items():
        print(f"[lab] {label:14s} 総合 {r['total']:5.1f}  似ている度 {r['sim']:.3f}  自然さ {r['mos']:.2f}  "
              f"読み間違い {r['cer']:.0%}  速さ {r['kps']:.1f}文字/秒", flush=True)
    print(f"[lab] 本人の話す速さ（素材の中央値）: {out['kps_real']:.1f}文字/秒", flush=True)
    print(f"[lab] 生成時間: 1 個ずつ {t_single:.0f}秒 / {a.n} 個まとめて {t_batch:.0f}秒（{len(SENTENCES)} 文）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
