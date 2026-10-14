"""学習した読み上げモデル（各周のチェックポイント）が正しく読めているかを採点する。

各チェックポイントで例文を読ませ、Whisper で書き起こして文字誤り率（CER）を出す。
学習しすぎるとモデルが壊れて雑音や同じ音の繰り返しになるので、それを見つけて一番良い周を選ぶ。

  python -m studio.eval_tts <声の名前>      → voices/<名前>/tts/eval.json と best.txt
"""

import argparse
import json
import re
import shutil

from studio.common import VOICES, write_status

SENTENCES = [
    "こんにちは。今日も一日よろしくお願いします。",
    "えー、本当ですか？すごくうれしいです。",
    "最近寒くなってきたので、風邪に気をつけてくださいね。",
]


def norm(t: str) -> str:
    t = re.sub(r"[\s、。，．,.!！?？「」『』…ー〜~]", "", t)
    return t.replace("本当", "ほんとう").replace("嬉しい", "うれしい")


def cer(ref: str, hyp: str) -> float:
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, b in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a != b))
    return d[-1] / max(len(r), 1)


def main() -> int:
    import librosa
    import soundfile as sf
    import torch
    from transformers import pipeline

    from studio import engine
    from tts import fetch_model

    ap = argparse.ArgumentParser()
    ap.add_argument("voice")
    ap.add_argument("--dir", default="tts", help="採点するフォルダ（voices/<名前>/ 以下）")
    ap.add_argument("--sim", action="store_true", help="（テスト機能）声の近さも採点に入れる")
    args = ap.parse_args()
    vdir = VOICES / args.voice
    ckpts = [p.name for p in sorted((vdir / args.dir).glob("epoch-*"), reverse=True)]
    if not ckpts:
        raise SystemExit("チェックポイントがありません")

    write_status(vdir, message="学習した読み上げモデルを採点中")
    wavs = {}
    for ck in ckpts:
        wavs[ck] = [engine.speak(args.voice, s, ck, sub=args.dir)[0] for s in SENTENCES]
        print(f"[eval] {ck} 生成完了", flush=True)
    engine.unload()

    asr = pipeline("automatic-speech-recognition", model=fetch_model("openai/whisper-large-v3-turbo"),
                   dtype=torch.float16, device="cuda:0")
    result = {}
    for ck, files in wavs.items():
        scores, texts = [], []
        for ref, f in zip(SENTENCES, files):
            y, sr = sf.read(f, dtype="float32")
            hyp = asr({"raw": librosa.resample(y, orig_sr=sr, target_sr=16000), "sampling_rate": 16000},
                      generate_kwargs={"language": "japanese"}, return_timestamps=True)["text"]
            scores.append(min(cer(ref, hyp), 1.0))
            texts.append(hyp)
        result[ck] = {"cer": round(sum(scores) / len(scores), 3), "texts": texts}
        print(f"[eval] {ck} 文字誤り率 {result[ck]['cer']:.2f} | {texts[0][:30]}", flush=True)

    if args.sim:
        # （テスト機能）生成した声が登録済みの声にどれだけ近いか。読めていても別人の声なら減点する
        import subprocess

        from studio.common import ROOT, SEP_PY, child_env

        files = [str(f) for fs in wavs.values() for f in fs]
        raws = [str(f) for f in (vdir / "raw").glob("*.wav")]
        sj = vdir / args.dir / "sim.json"
        subprocess.run([str(SEP_PY), "-m", "studio.purity", "files", *files, "--ref", *raws, "-o", str(sj)],
                       cwd=ROOT, env=child_env(), check=True)
        sims = json.loads(sj.read_text(encoding="utf-8"))
        from pathlib import Path

        for ck, fs in wavs.items():
            sim = sum(sims[Path(f).name] for f in fs) / len(fs)
            result[ck]["sim"] = round(sim, 3)
            # 誤り率に、近さの不足分（0.75 を下回った分）を足して総合点にする（小さいほど良い）
            result[ck]["score"] = round(result[ck]["cer"] + max(0.0, 0.75 - sim), 3)
            print(f"[eval] {ck} 声の近さ {sim:.2f} / 総合 {result[ck]['score']:.2f}", flush=True)

    (vdir / args.dir / "eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    # 誤り率が同じくらいなら新しい（よく学習した）ほうを選ぶ
    if args.sim:
        best = min(result, key=lambda c: (result[c]["score"], ckpts.index(c)))
    else:
        best = min(result, key=lambda c: (round(result[c]["cer"], 1), ckpts.index(c)))  # ckpts は新しい順
    if result[best]["cer"] > 0.5:
        # どれも読めていない → 使わない（画面では参照音声からのクローンで読み上げる）
        for ck in ckpts:
            shutil.rmtree(vdir / args.dir / ck, ignore_errors=True)
        (vdir / args.dir / "best.txt").unlink(missing_ok=True)
        write_status(vdir, tts_best=None, message="読み上げモデルの学習に失敗しました（参照音声からのクローンで読み上げます）")
        print("[eval] 使えるモデルがありません。すべて削除しました", flush=True)
        return 0
    (vdir / args.dir / "best.txt").write_text(best, encoding="utf-8")
    write_status(vdir, tts_best=best, tts_best_cer=result[best]["cer"],
                 message=f"一番よく読めるモデル: {best}（文字誤り率 {result[best]['cer']:.0%}）")
    print(f"[eval] 一番良いモデル: {best}", flush=True)
    for ck in ckpts:  # 壊れたモデルは消して取り違えを防ぐ
        if result[ck]["cer"] > 0.5 and ck != best:
            shutil.rmtree(vdir / args.dir / ck, ignore_errors=True)
            print(f"[eval] {ck} は読めていないので削除", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
