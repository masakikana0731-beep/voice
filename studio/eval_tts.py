"""学習した読み上げモデル（各周のチェックポイント）が正しく読めているかを採点する。

各チェックポイントで例文を読ませ、読み間違い（Whisper の書き起こし）・似ている度・自然さを測り、
本物の録音を 100 とした総合点（studio/quality.py）で一番良い周を選ぶ。
学習しすぎるとモデルが壊れて雑音や同じ音の繰り返しになるので、読み間違いが 5 割を超える周は使わない。

  python -m studio.eval_tts <声の名前>      → voices/<名前>/tts/eval.json と best.txt
"""

import argparse
import json
import shutil

from studio.common import VOICES, write_status

SENTENCES = [
    "こんにちは。今日も一日よろしくお願いします。",
    "えー、本当ですか？すごくうれしいです。",
    "最近寒くなってきたので、風邪に気をつけてくださいね。",
]


def main() -> int:
    from studio import engine

    ap = argparse.ArgumentParser()
    ap.add_argument("voice")
    ap.add_argument("--dir", default="tts", help="採点するフォルダ（voices/<名前>/ 以下）")
    ap.add_argument("--sim", action="store_true", help="（互換用。いまは常に似ている度・自然さも測る）")
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

    from studio import quality

    files = [f for fs in wavs.values() for f in fs]
    hyps = dict(zip(map(str, files), quality.transcribe(files)))
    quality.unload_asr()
    result = {}
    for ck, fs in wavs.items():
        texts = [hyps[str(f)] for f in fs]
        result[ck] = {"cer": round(sum(quality.cer(r, h) for r, h in zip(SENTENCES, texts)) / len(fs), 3),
                      "texts": texts}
        print(f"[eval] {ck} 文字誤り率 {result[ck]['cer']:.2f} | {texts[0][:30]}", flush=True)

    # 似ている度と自然さも測り、本物の録音（素材のクリップ）を 100 とした総合点で選ぶ（studio/quality.py）。
    # 読み間違いだけで選ぶと、読めていても声が離れた周を選んでしまう（ayumin: 近さ 0.44 の周があった）
    import random
    import shutil as _sh
    import tempfile
    from pathlib import Path

    use_total = True
    try:
        refs = sorted((vdir / "raw").glob("*.wav")) or sorted((vdir / "clips_hq").glob("*.wav"))
        reals = sorted((vdir / "clips_hq").glob("*.wav"))
        reals = random.Random(0).sample(reals, min(8, len(reals)))
        with tempfile.TemporaryDirectory() as tmp:
            named = {}
            for ck, fs in wavs.items():  # 名前が重ならないよう「周__番号」にそろえる
                for i, f in enumerate(fs):
                    dst = Path(tmp) / f"{ck}__{i}.wav"
                    _sh.copy(f, dst)
                    named[dst.name] = ck
            for i, f in enumerate(reals):
                _sh.copy(f, Path(tmp) / f"real__{i}.wav")
            j = quality.judge(sorted(Path(tmp).glob("*.wav")), refs)
        real = [v for k, v in j.items() if k.startswith("real__")]
        base = {"cer": 0.03, "sim": quality.mean(real, "sim"), "mos": quality.mean(real, "mos")}
        for ck in wavs:
            vals = [v for k, v in j.items() if named.get(k) == ck]
            result[ck].update(sim=quality.mean(vals, "sim"), mos=quality.mean(vals, "mos"))
            result[ck]["total"] = quality.total(result[ck], base)
            print(f"[eval] {ck} 似ている度 {result[ck]['sim']} / 自然さ {result[ck]['mos']} / 総合 {result[ck]['total']}"
                  f"（本物: 似ている度 {base['sim']} / 自然さ {base['mos']}）", flush=True)
    except Exception as e:  # 採点役が動かなければ、読み間違いだけで選ぶ
        print(f"[eval] 似ている度・自然さを測れませんでした（読み間違いだけで選びます）: {e}", flush=True)
        use_total = False

    (vdir / args.dir / "eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    # 誤り率が同じくらいなら新しい（よく学習した）ほうを選ぶ
    usable = [c for c in result if result[c]["cer"] <= 0.5]
    if use_total and usable:
        best = max(usable, key=lambda c: (result[c]["total"], -ckpts.index(c)))  # 同点なら新しい周
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
