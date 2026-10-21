"""声変換の改善の実験: 品質（拡散ステップ数）・何回か変換して選ぶ・仕上げの組み合わせを、採点して比べる。

  python -m studio.vc_lab <話者>   → bench/vc_lab_<話者>.json と表

変換の元は精度検証の採点用の文（bench/test/vc_src、内蔵の声 Ono_Anna が読んだ 8 文。正解の文あり）。
採点は studio/quality.py（似ている度は話者の素材全体、自然さは UTMOSv2、読み間違いは Whisper）。
"""

import argparse
import json
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from studio import bench, engine, quality
from studio.common import VOICES
from studio.speakers import vc_model


def convert_dir(sid, src: Path, out: Path, steps: int):
    st = engine.vc_settings(sid)
    model = vc_model(sid)
    engine._vc_request({"segments": str(src), "target": str(engine._vc_ref(sid)), "out": str(out), "steps": steps,
                        "auto_pitch": True, "semitone": 0, "cfg": float(st.get("cfg", 0.7)),
                        "checkpoint": str(model[0]), "config": str(model[1])})


def polish(y, sr, kind: str):
    """強めの仕上げ（pedalboard）。"""
    from pedalboard import Compressor, HighpassFilter, LowpassFilter, Pedalboard, PeakFilter

    chain = {
        # 変換で出る高い帯域のシャリシャリを抑え、音量の揺れをならす
        "polish": [HighpassFilter(70), PeakFilter(7000, gain_db=-3, q=0.7), LowpassFilter(12000),
                   Compressor(threshold_db=-22, ratio=2.5, attack_ms=8, release_ms=120)],
    }[kind]
    return Pedalboard(chain)(y.astype(np.float32), sr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("voice")
    ap.add_argument("--n", type=int, default=3, help="何回か変換して選ぶときの回数")
    a = ap.parse_args()
    sid, vdir = a.voice, VOICES / a.voice
    test = json.loads((bench.TEST / "test.json").read_text(encoding="utf-8"))["vc"]
    texts = {t["name"]: t["text"] for t in test}
    refs = sorted((vdir / "raw").glob("*.wav"))
    work = bench.BENCH / f"vc_lab_{sid}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    # 1. 変換（ステップ数 3 通り。50 は n 回）
    runs = {}
    for steps, times in ((30, 1), (50, a.n), (100, 1)):
        for k in range(times):
            d = work / f"s{steps}_{k}"
            convert_dir(sid, bench.TEST / "vc_src", d, steps)
            runs[d.name] = d
            print(f"[lab] 変換 {d.name}", flush=True)
    engine.stop_vc_server()

    # 2. 仕上げ（50 ステップの 1 回目に対して）
    base = work / "s50_0"
    for kind in ("clean", "polish", "clean_polish"):
        d = work / f"s50_0_{kind}"
        d.mkdir()
        for f in sorted(base.glob("*.wav")):
            src = f
            if "clean" in kind:
                from studio.post import clean

                src = clean(f)
            y, sr = sf.read(src, dtype="float32")
            if "polish" in kind:
                y = polish(y, sr, "polish")
            sf.write(d / f.name, y, sr)
            if src != f:
                Path(src).unlink(missing_ok=True)
        runs[d.name] = d
        print(f"[lab] 仕上げ {kind}", flush=True)

    # 3. 採点（全部まとめて）。名前は「条件__文」
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for label, d in runs.items():
            for f in sorted(d.glob("*.wav")):
                dst = Path(tmp) / f"{label}__{f.name}"
                shutil.copy(f, dst)
                files.append(dst)
        reals = random.Random(0).sample(sorted((vdir / "clips_hq").glob("*.wav")), 8)
        for i, f in enumerate(reals):
            shutil.copy(f, Path(tmp) / f"real__{i}.wav")
        allf = files + [Path(tmp) / f"real__{i}.wav" for i in range(len(reals))]
        j = quality.judge(allf, refs)
        hyps = dict(zip(map(str, files), quality.transcribe(files)))
    quality.unload_asr()
    real = [v for k, v in j.items() if k.startswith("real__")]
    basem = {"cer": 0.03, "sim": quality.mean(real, "sim"), "mos": quality.mean(real, "mos")}
    rows = {}
    per = {}
    for f in files:
        label, name = f.name.split("__")
        v = {**j[f.name], "cer": quality.cer(texts[name], hyps[str(f)])}
        per.setdefault(label, {})[name] = v
    for label, d in per.items():
        m = {k: quality.mean(list(d.values()), k) for k in ("sim", "mos", "cer")}
        m["total"] = quality.total(m, basem)
        rows[label] = m
    # 何回か変換して、文ごとに一番良いものを選んだ場合
    pick = {}
    for name in texts:
        cands = [per[f"s50_{k}"][name] for k in range(a.n)]
        pick[name] = max(cands, key=lambda v: quality.total(v, basem))
    m = {k: quality.mean(list(pick.values()), k) for k in ("sim", "mos", "cer")}
    m["total"] = quality.total(m, basem)
    rows[f"s50_best_of_{a.n}"] = m
    out = {"base": basem, "rows": rows}
    (bench.BENCH / f"vc_lab_{sid}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[lab] 本物: 似ている度 {basem['sim']} / 自然さ {basem['mos']}", flush=True)
    for label, m in sorted(rows.items(), key=lambda kv: -kv[1]["total"]):
        print(f"[lab] {label:20s} 総合 {m['total']:5.1f}  似ている度 {m['sim']:.3f}  自然さ {m['mos']:.2f}  "
              f"読み間違い {m['cer']:.0%}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
