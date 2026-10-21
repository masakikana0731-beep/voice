"""音声の採点（.venv-sep で動かす）: 似ている度（ECAPA）と自然さ（UTMOSv2）を一度に測る。

  python -m studio.judge <wav...> --ref <本物の声...> -o out.json [--no-mos] [--list ファイル一覧.txt]
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
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--list", help="採点する wav の一覧（1 行 1 つ）。数が多くコマンドに書ききれないとき")
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--no-mos", action="store_true")
    ap.add_argument("--reps", type=int, default=5, help="自然さを何回測って平均するか（素材選びは 1 で速く）")
    ap.add_argument("--print-cache", help="本物の声の特徴の保存先（.npy）。本物の声のファイルより新しければ使い回す")
    a = ap.parse_args()

    paths = list(a.paths) + (Path(a.list).read_text(encoding="utf-8").splitlines() if a.list else [])
    cache = Path(a.print_cache) if a.print_cache else None
    newest = max(Path(r).stat().st_mtime for r in a.ref)
    if cache and cache.exists() and cache.stat().st_mtime > newest:
        ref = np.load(cache)  # 素材 30 分を毎回聞き直すと 10 秒以上かかるので使い回す
    else:
        ref = voice_print(a.ref)
        if cache:
            np.save(cache, ref)
    result = {}
    waves = {}
    for p in map(Path, filter(None, paths)):
        y, _ = librosa.load(str(p), sr=16000, mono=True)
        waves[p.name] = y
        segs = [y[i:i + 48000] for i in range(0, max(len(y) - 24000, 1), 48000)] or [y]
        v = embed(segs).mean(axis=0)
        result[p.name] = {"sim": round(float(v @ ref / (np.sum(v ** 2) ** 0.5)), 4)}
    if not a.no_mos:
        os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
        import utmosv2

        import torch

        model = utmosv2.create_model(pretrained=True)
        for name, y in waves.items():
            # 区間をランダムに切り出して採点するので、1 回だと同じ音声でも ±0.1 ほどぶれる（本物 3.12 → 2.98）。
            # 乱数を固定し、5 回の平均にする
            torch.manual_seed(0)
            np.random.seed(0)
            pred = model.predict(data=y, sr=16000, verbose=False, num_repetitions=a.reps)
            result[name]["mos"] = round(float(np.ravel(pred)[0]), 4)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[judge] {len(result)} 件", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
