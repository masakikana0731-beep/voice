"""声変換モデルの採点と、前のモデルとの比べっこ。

決まったテスト音声（assets/vc_eval_source.wav）を変換し、登録済みの声にどれだけ似たかで採点する。
学習しすぎて前より悪くなることがあるので、学習のあと「前のモデル」と比べて良いほうを残す。

  python -m studio.vc_eval <話者> [--compare]   → 今のモデル（と前のモデル）の点数を表示
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from studio.common import ROOT, SEEDVC, SEP_PY, VC_PY, VOICES, child_env

EVAL_SOURCE = ROOT / "assets" / "vc_eval_source.wav"


def score(sid: str, model_dir: Path, with_mos: bool = False):
    """model_dir（ft_model.pth と設定）で EVAL_SOURCE を変換し、似ている度（0〜1）を返す。
    with_mos=True なら (似ている度, 自然さ 1〜5) を返す（自然さの採点に失敗したら None）。"""
    import soundfile as sf

    from studio.engine import split_at_silence, vc_reference

    ck = model_dir / "ft_model.pth"
    cfg = next(model_dir.glob("config_*.yml"))
    ref = vc_reference(VOICES / sid)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        y, sr = sf.read(EVAL_SOURCE, dtype="float32")
        seg = tmp / "seg"
        seg.mkdir()
        for i, (s0, e0) in enumerate(split_at_silence(y, sr)):
            sf.write(seg / f"{i:03d}.wav", y[s0:e0], sr)
        out = tmp / "out"
        subprocess.run([str(VC_PY), str(ROOT / "studio" / "vc_run.py"), str(ROOT / "studio" / "vc_batch.py"),
                        "--segments", str(seg), "--target", str(ref), "--out", str(out), "--diffusion-steps", "30",
                        "--checkpoint", str(ck), "--config", str(cfg)],
                       cwd=SEEDVC, env=child_env(), check=True, capture_output=True)
        files = sorted(out.glob("*.wav"))
        res = tmp / "sim.json"
        raws = [str(f) for f in (VOICES / sid / "raw").glob("*.wav")]
        subprocess.run([str(SEP_PY), "-m", "studio.purity", "files", *map(str, files), "--ref", *raws, "-o", str(res)],
                       cwd=ROOT, env=child_env(), check=True, capture_output=True)
        sims = json.loads(res.read_text(encoding="utf-8"))
        # 区間の長さで重みをつけた平均
        w = [sf.info(f).duration for f in files]
        sim = round(sum(sims[f.name] * d for f, d in zip(files, w)) / sum(w), 3)
        if not with_mos:
            return sim
        try:
            from studio.engine import mos

            m = mos(files)
            return sim, round(sum(m[f.name] * d for f, d in zip(files, w)) / sum(w), 3)
        except Exception:
            return sim, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("voice")
    ap.add_argument("--compare", action="store_true", help="vc_prev（前のモデル）とも比べる")
    a = ap.parse_args()
    vdir = VOICES / a.voice
    print(f"[vc_eval] 今のモデル（似ている度, 自然さ）: {score(a.voice, vdir / 'vc', True)}", flush=True)
    if a.compare and (vdir / "vc_prev" / "ft_model.pth").exists():
        print(f"[vc_eval] 前のモデル（似ている度, 自然さ）: {score(a.voice, vdir / 'vc_prev', True)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
