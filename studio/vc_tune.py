"""声変換の設定を話者ごとに自動で最適化する（参照音声の選び直し・声の寄せ具合）。

  python -m studio.vc_tune <話者>   → voices/<話者>/vc_settings.json

決まったテスト音声（assets/vc_eval_source.wav）を、参照音声の候補 × 寄せ具合（cfg）の組み合わせで変換し、
登録済みの声にいちばん似たものを選ぶ。参照音声の候補は、素材の中から「話者らしさ」の高いクリップで作る。
"""

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from studio import engine
from studio.common import ROOT, SEP_PY, VOICES, child_env, write_status
from studio.speakers import vc_model
from studio.vc_eval import EVAL_SOURCE

CFGS = [0.5, 0.7, 0.9]
N_REFS = 4


def purity(paths, refs, out_json, mode="files"):
    subprocess.run([str(SEP_PY), "-m", "studio.purity", mode, *map(str, paths), "--ref", *map(str, refs),
                    "-o", str(out_json)], cwd=ROOT, env=child_env(), check=True, capture_output=True)
    return json.loads(Path(out_json).read_text(encoding="utf-8"))


def make_refs(sid: str, tmp: Path) -> list[Path]:
    """参照音声の候補（約 12 秒ずつ）。話者らしさの高いクリップを 2〜3 本ずつつなぐ。"""
    vdir = VOICES / sid
    raws = list((vdir / "raw").glob("*.wav"))
    pj = vdir / "purity.json"
    sims = json.loads(pj.read_text(encoding="utf-8")) if pj.exists() else purity([vdir / "clips_hq"], raws, pj, "clips")
    good = []
    for name, sim in sorted(sims.items(), key=lambda kv: -kv[1]):
        f = vdir / "clips_hq" / name
        if f.exists() and 3 <= sf.info(f).duration <= 9:
            good.append(f)
    refs = []
    for k in range(N_REFS):
        pieces, total = [], 0.0
        for f in good[k::N_REFS]:  # 候補ごとに別のクリップを使う
            y, sr = sf.read(f, dtype="float32")
            pieces += [y if y.ndim == 1 else y.mean(axis=1), np.zeros(int(0.2 * sr), dtype=np.float32)]
            total += len(y) / sr
            if total >= 11:
                break
        if pieces:
            r = vdir / f"ref_cand{k + 1}.wav"
            sf.write(r, np.concatenate(pieces)[: int(14 * sr)], sr)
            refs.append(r)
    return [engine.vc_reference(vdir)] + refs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("voice")
    a = ap.parse_args()
    sid, vdir = a.voice, VOICES / a.voice
    model = vc_model(sid)
    if not model:
        raise SystemExit("声変換の学習済みモデルがありません")
    write_status(vdir, message="声変換の設定を最適化しています")
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        refs = make_refs(sid, tmp)
        y, sr = sf.read(EVAL_SOURCE, dtype="float32")
        seg = tmp / "seg"
        seg.mkdir()
        spans = engine.split_at_silence(y, sr)
        for i, (s0, e0) in enumerate(spans):
            sf.write(seg / f"{i:03d}.wav", y[s0:e0], sr)
        trials = []
        for r in refs:
            for cfg in CFGS:
                out = tmp / f"{r.stem}_{cfg}"
                engine._vc_request({"segments": str(seg), "target": str(r), "out": str(out), "steps": 30,
                                    "auto_pitch": True, "semitone": 0, "cfg": cfg,
                                    "checkpoint": str(model[0]), "config": str(model[1])})
                # 区間をつないで 1 本にして採点する
                joined = np.concatenate([sf.read(f, dtype="float32")[0] for f in sorted(out.glob("*.wav"))])
                f = tmp / f"trial_{r.stem}_{cfg}.wav"
                sf.write(f, joined, 44100)
                trials.append((f, r, cfg))
                print(f"[tune] {r.name} cfg={cfg}", flush=True)
        engine.stop_vc_server()
        sims = purity([t[0] for t in trials], list((vdir / "raw").glob("*.wav")), tmp / "sim.json")
    scored = sorted(((sims[f.name], r, cfg) for f, r, cfg in trials), key=lambda x: -x[0])
    base = next(s for s, r, cfg in scored if r.name == "ref_vc.wav" and cfg == 0.7)
    best_s, best_r, best_cfg = scored[0]
    st = {"ref": best_r.name, "cfg": best_cfg, "steps": 50, "score": round(best_s, 3), "baseline": round(base, 3),
          "tuned": time.strftime("%Y-%m-%d %H:%M")}
    (vdir / "vc_settings.json").write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    for s, r, cfg in scored:
        print(f"[tune] {s:.3f}  {r.name}  cfg={cfg}", flush=True)
    msg = f"声変換の設定を最適化しました（似ている度 {base:.2f} → {best_s:.2f}、{time.time() - t0:.0f}秒）"
    write_status(vdir, message=msg)
    print(f"[tune] {msg}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
