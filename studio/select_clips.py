"""素材の自動選別: 学習の前に、相手の声が混ざったクリップや雑音・こもりの強いクリップを外す。

  python -m studio.select_clips <話者>   （単体で動かすと、外す候補を表示するだけ）

クリップごとに「似ている度」（話者の素材全体の声との近さ）と「自然さ」（AI の音質の採点）を測り
（studio/judge.py。自然さは 1 回だけ測って速くする。24 分の素材で 3〜4 分）、
  - 似ている度 SIM_MIN 未満 → 別の声が混ざっている
  - 自然さ MOS_MIN 未満     → 雑音・こもりが強い
を外す。外しすぎないよう、外すのは全体の長さの MAX_DROP まで（悪い順）。
あゆみん（ラジオ素材）では、似ている度の中央値 0.79・下位 5% が 0.62、自然さの中央値 1.9・下位 10% が 1.4 だった。
"""

import argparse
import json
from pathlib import Path

import soundfile as sf

from studio.common import VOICES

SIM_MIN = 0.62
MOS_MIN = 1.35
MAX_DROP = 0.25


def score_clips(vdir: Path) -> dict:
    from studio import quality

    clips = sorted((vdir / "clips_hq").glob("*.wav"))
    refs = sorted((vdir / "raw").glob("*.wav")) or clips
    j = quality.judge(clips, refs, mos=True, reps=1)
    (vdir / "clip_scores.json").write_text(json.dumps(j, ensure_ascii=False, indent=1), encoding="utf-8")
    return j


def choose(vdir: Path, scores: dict) -> dict:
    """外すクリップ {名前: 理由}。"""
    bad = []
    for name, v in scores.items():
        why = []
        if v["sim"] < SIM_MIN:
            why.append(f"別の声が混ざっている（似ている度 {v['sim']:.2f}）")
        if v.get("mos") is not None and v["mos"] < MOS_MIN:
            why.append(f"雑音・こもりが強い（自然さ {v['mos']:.2f}）")
        if why:
            # 悪さ: 基準からどれだけ下か（似ている度は 0.1、自然さは 0.3 を同じ重みに）
            badness = max(0.0, SIM_MIN - v["sim"]) / 0.1 + max(0.0, MOS_MIN - (v.get("mos") or MOS_MIN)) / 0.3
            bad.append((badness, name, "・".join(why)))
    total = sum(sf.info(vdir / "clips_hq" / n).duration for n in scores)
    out, dropped = {}, 0.0
    for _, name, why in sorted(bad, reverse=True):
        d = sf.info(vdir / "clips_hq" / name).duration
        if dropped + d > MAX_DROP * total:
            break
        out[name] = why
        dropped += d
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("voice")
    a = ap.parse_args()
    vdir = VOICES / a.voice
    drop = choose(vdir, score_clips(vdir))
    total = len(list((vdir / "clips_hq").glob("*.wav")))
    print(f"[select] {total} クリップ中 {len(drop)} を外す候補", flush=True)
    for n, why in drop.items():
        print(f"  {n}: {why}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
