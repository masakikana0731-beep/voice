"""音声の採点（.venv-sep で動かす）: 似ている度（ECAPA）と自然さ（UTMOSv2）を一度に測る。

  python -m studio.judge <wav...> --ref <本物の声...> -o out.json [--no-mos]
  → {ファイル名: {"sim": 0〜1, "mos": 1〜5}}

精度検証（studio/bench.py）と、学習後の自動採点（studio/quality.py）から呼ぶ。
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    import librosa
    import numpy as np

    from studio.common import ROOT
    from studio.separate import embed, voice_print

    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--no-mos", action="store_true")
    a = ap.parse_args()

    ref = voice_print(a.ref)
    result = {}
    waves = {}
    for p in map(Path, a.paths):
        y, _ = librosa.load(str(p), sr=16000, mono=True)
        waves[p.name] = y
        segs = [y[i:i + 48000] for i in range(0, max(len(y) - 24000, 1), 48000)] or [y]
        v = embed(segs).mean(axis=0)
        result[p.name] = {"sim": round(float(v @ ref / (np.sum(v ** 2) ** 0.5)), 4)}
    if not a.no_mos:
        os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
        import utmosv2

        model = utmosv2.create_model(pretrained=True)
        for name, y in waves.items():
            result[name]["mos"] = round(float(np.ravel(model.predict(data=y, sr=16000, verbose=False))[0]), 4)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[judge] {len(result)} 件", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
