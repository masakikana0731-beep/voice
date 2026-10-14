"""Seed-VC で複数の区間をまとめて変換する（モデルの読み込みは 1 回だけ）。.venv-vc で studio/vc_run.py 経由で動かす。

  vc_run.py studio/vc_batch.py --segments <区間 wav のフォルダ> --target <参照音声> --out <出力フォルダ> [inference.py と同じ引数]

区間 NNN.wav ごとに <出力フォルダ>/NNN.wav を書く。
Seed-VC 本体は 30 秒を超えると内部で区切り、継ぎ目が 0.2 秒ほどしか重ならないため音が途切れやすい。
呼び出し側で無音のところを区切りにして 15 秒以下ずつ渡すことで、継ぎ目を声の途中に作らない。
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")  # cwd = third_party/seed-vc
import inference  # noqa: E402

_cache = {}
_load = inference.load_models


def cached_load(args):
    if "models" not in _cache:
        _cache["models"] = _load(args)
    return _cache["models"]


inference.load_models = cached_load


def str2bool(v):
    return str(v).lower() in ("true", "1", "yes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--diffusion-steps", type=int, default=50)
    ap.add_argument("--length-adjust", type=float, default=1.0)
    ap.add_argument("--inference-cfg-rate", type=float, default=0.7)
    ap.add_argument("--f0-condition", type=str2bool, default=True)
    ap.add_argument("--auto-f0-adjust", type=str2bool, default=True)
    ap.add_argument("--semi-tone-shift", type=int, default=0)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--fp16", type=str2bool, default=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    segs = sorted(Path(a.segments).glob("*.wav"))
    for i, seg in enumerate(segs, 1):
        tmp = out / f"_{seg.stem}"
        args = argparse.Namespace(
            source=str(seg), target=a.target, output=str(tmp), diffusion_steps=a.diffusion_steps,
            length_adjust=a.length_adjust, inference_cfg_rate=a.inference_cfg_rate, f0_condition=a.f0_condition,
            auto_f0_adjust=a.auto_f0_adjust, semi_tone_shift=a.semi_tone_shift, checkpoint=a.checkpoint,
            config=a.config, fp16=a.fp16,
        )
        inference.main(args)
        produced = list(tmp.glob("vc_*.wav"))
        if not produced:
            raise RuntimeError(f"{seg.name} の変換結果がありません")
        shutil.move(str(produced[0]), out / seg.name)
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"[vc] {i}/{len(segs)}", flush=True)


if __name__ == "__main__":
    main()
