"""自然さの自動採点（.venv-sep で動かす。UTMOSv2 を使う）。

  python -m studio.mos <wav...> -o mos.json   → {ファイル名: 自然さ}

自然さは 1〜5（人が聞いて付ける「自然さの点数」MOS を AI が予想したもの）。3 が普通、4 以上でかなり自然。
英語中心のデータで学習したモデルなので、日本語では「同じ声・同じ文の比べっこ」に使う（点数そのものより差を見る）。
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

    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    # モデルは models/ に置く（標準のキャッシュはシンボリックリンク権限がなく失敗するため）
    from studio.common import ROOT

    os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))
    os.environ.setdefault("TORCH_HOME", str(ROOT / "models" / "torch"))
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    import utmosv2

    model = utmosv2.create_model(pretrained=True)
    result = {}
    for p in map(Path, a.paths):
        y, _ = librosa.load(str(p), sr=16000, mono=True)
        result[p.name] = round(float(np.ravel(model.predict(data=y, sr=16000, verbose=False))[0]), 3)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[mos] {len(result)} 件", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
