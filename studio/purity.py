"""声の近さを測る（.venv-sep で動かす。話者埋め込み ECAPA を使う）。

  学習クリップの「混ざり」チェック:
    python -m studio.purity clips <クリップのフォルダ> --ref <登録済みの声...> -o purity.json
  生成した音声の近さ（読み上げモデルの採点用）:
    python -m studio.purity files <wav...> --ref <登録済みの声...> -o sim.json

出力は {ファイル名: 近さ}。同じ人 ≈0.6 以上、別人 ≈0.3 以下が目安。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from studio.separate import embed, voice_print  # noqa: E402


def main() -> int:
    import librosa

    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["clips", "files"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    ref = voice_print(a.ref)
    files = sorted(Path(a.paths[0]).glob("*.wav")) if a.mode == "clips" else [Path(p) for p in a.paths]
    result = {}
    for f in files:
        y, _ = librosa.load(str(f), sr=16000, mono=True)
        # 3 秒ずつに切って平均（短いクリップはそのまま）
        segs = [y[i:i + 48000] for i in range(0, max(len(y) - 24000, 1), 48000)] or [y]
        v = embed(segs).mean(axis=0)
        result[f.name] = round(float(v @ ref / (sum(v ** 2) ** 0.5)), 3)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[purity] {len(result)} 件", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
