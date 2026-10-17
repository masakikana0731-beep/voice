"""学習の全工程を順に実行する（UI から裏で起動される。単体でも可）。

  python -m studio.train <声の名前> [--tts-epochs 10] [--vc-steps 3000] [--size 1.7B] [--skip tts|vc]

工程ごとに別プロセスで動かし、GPU メモリを毎回解放する。ログは voices/<名前>/train.log。
"""

import argparse
import os
import re

from studio.common import PY, ROOT, SEEDVC, VC_CONFIG, VC_PY, VOICES, run_logged, write_status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("voice")
    ap.add_argument("--size", default="1.7B")
    ap.add_argument("--tts-epochs", type=int, default=3)
    ap.add_argument("--vc-steps", type=int, default=3000)
    ap.add_argument("--vc-batch", type=int, default=1, help="12GB の GPU では 1（2 だとメモリがあふれて極端に遅くなる）")
    ap.add_argument("--rt-steps", type=int, default=2500, help="リアルタイム用モデルの学習ステップ数")
    ap.add_argument("--purity", type=float, default=0, help="（テスト機能）混ざりの除外のしきい値")
    ap.add_argument("--eval-sim", action="store_true", help="（テスト機能）採点に声の近さも使う")
    ap.add_argument("--skip", action="append", default=[], choices=["prep", "tts", "vc", "rt"])
    args = ap.parse_args()

    vdir = VOICES / args.voice
    log = vdir / "train.log"
    write_status(vdir, stage="start", message="学習を開始します", pid=os.getpid(), error=None, progress=0)

    def fail(what):
        write_status(vdir, stage="error", error=f"{what}で失敗しました。train.log を確認してください", pid=None)
        return 1

    if "prep" not in args.skip and run_logged([PY, "-m", "studio.prep", args.voice, "--purity", args.purity], log) != 0:
        return fail("データ準備")

    if "tts" not in args.skip:
        cmd = [PY, "-m", "studio.train_tts", args.voice, "--size", args.size, "--epochs", args.tts_epochs,
               "--keep", args.tts_epochs]
        if run_logged(cmd, log) != 0:
            return fail("文章読み上げの学習")
        # 学習しすぎで壊れることがあるので、各周を採点して一番よく読めるものを選ぶ
        if run_logged([PY, "-m", "studio.eval_tts", args.voice] + (["--sim"] if args.eval_sim else []), log) != 0:
            return fail("読み上げモデルの採点")

    if "vc" not in args.skip:
        steps = args.vc_steps

        def progress(line):
            m = re.search(r"step (\d+), loss: ([\d.]+)", line)
            if m:
                s = int(m.group(1))
                write_status(vdir, stage="vc", progress=round(s / steps, 3),
                             message=f"声変換の学習 step {s}/{steps} loss {float(m.group(2)):.3f}")

        write_status(vdir, stage="vc", message="声変換の学習を準備中", progress=0)
        cmd = [VC_PY, ROOT / "studio" / "vc_run.py", "train.py", "--config", VC_CONFIG, "--dataset-dir", vdir / "clips_hq",
               "--run-name", args.voice, "--batch-size", args.vc_batch, "--max-steps", steps, "--max-epochs", 100000,
               "--save-every", 500, "--num-workers", 0]
        if run_logged(cmd, log, cwd=SEEDVC, on_line=progress) != 0:
            return fail("声変換の学習")
        from studio.speakers import keep_vc_model

        keep_vc_model(args.voice)  # 話者フォルダ（voices/<名前>/vc）にもまとめて保存
        write_status(vdir, vc_checkpoint=str(vdir / "vc" / "ft_model.pth"))

    if "rt" not in args.skip:
        # リアルタイム変換（Voice Studio Live）用の軽いモデル。声変換のモデルは重くてリアルタイムに使えないため別に学習する
        from studio.speakers import RT_CONFIG, keep_rt_model

        rt_steps = args.rt_steps

        def rt_progress(line):
            m = re.search(r"step (\d+), loss: ([\d.]+)", line)
            if m:
                write_status(vdir, stage="rt", progress=round(int(m.group(1)) / rt_steps, 3),
                             message=f"リアルタイム用の学習 step {m.group(1)}/{rt_steps}")

        write_status(vdir, stage="rt", message="リアルタイム用の学習を準備中", progress=0)
        cmd = [VC_PY, ROOT / "studio" / "vc_run.py", "train.py", "--config", RT_CONFIG, "--dataset-dir", vdir / "clips_hq",
               "--run-name", f"{args.voice}_rt", "--batch-size", 2, "--max-steps", rt_steps, "--max-epochs", 100000,
               "--save-every", 500, "--num-workers", 0]
        if run_logged(cmd, log, cwd=SEEDVC, on_line=rt_progress) != 0:
            return fail("リアルタイム用の学習")
        keep_rt_model(args.voice)

    write_status(vdir, stage="done", message="すべての学習が完了しました", progress=1, pid=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
