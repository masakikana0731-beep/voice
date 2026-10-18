"""声変換の常駐係（.venv-vc で studio/vc_run.py 経由で動かす。cwd = third_party/seed-vc）。

モデルを一度読み込んだら持ったまま、標準入力から 1 行 1 件の依頼（JSON）を受けて変換する。
変換のたびにモデルを読み込み直す（約 10 秒）のをなくすため。

依頼: {"segments": 区間wavのフォルダ, "target": 参照音声, "out": 出力フォルダ, "steps": 50,
       "auto_pitch": true, "semitone": 0, "checkpoint": パス|null, "config": パス|null}
返事: 標準出力に "@@DONE" か "@@ERR <理由>" の 1 行（ほかの出力は無視してよい）
"""

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

sys.path.insert(0, ".")
import inference  # noqa: E402

_models = {}
_load = inference.load_models


def cached_load(args):
    """モデルの組み合わせ（チェックポイント・設定）ごとに 1 回だけ読み込む。"""
    key = (args.checkpoint, args.config, args.f0_condition, args.fp16)
    if key not in _models:
        _models.clear()  # 別の話者に切り替えたら前のモデルは捨てる（GPU メモリ節約）
        _models[key] = _load(args)
    return _models[key]


inference.load_models = cached_load


def handle(req):
    if req.get("warm"):  # モデルを先に読み込むだけ（1 回目の変換の待ちをなくす）
        inference.load_models(argparse.Namespace(checkpoint=req.get("checkpoint"), config=req.get("config"),
                                                 f0_condition=True, fp16=True))
        return
    out = Path(req["out"])
    out.mkdir(parents=True, exist_ok=True)
    for seg in sorted(Path(req["segments"]).glob("*.wav")):
        tmp = out / f"_{seg.stem}"
        args = argparse.Namespace(
            source=str(seg), target=req["target"], output=str(tmp), diffusion_steps=int(req.get("steps", 50)),
            length_adjust=1.0, inference_cfg_rate=float(req.get("cfg", 0.7)), f0_condition=True,
            auto_f0_adjust=bool(req.get("auto_pitch", True)), semi_tone_shift=int(req.get("semitone", 0)),
            checkpoint=req.get("checkpoint"), config=req.get("config"), fp16=True,
        )
        inference.main(args)
        produced = list(tmp.glob("vc_*.wav"))
        if not produced:
            raise RuntimeError(f"{seg.name} の変換結果がありません")
        shutil.move(str(produced[0]), out / seg.name)
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("@@READY", flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            handle(json.loads(line))
            print("@@DONE", flush=True)
        except Exception as e:
            traceback.print_exc()
            print(f"@@ERR {e}".replace("\n", " "), flush=True)


if __name__ == "__main__":
    main()
