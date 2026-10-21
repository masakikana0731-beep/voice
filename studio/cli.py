"""コマンドで声を作る（Claude のチャットから使う。スキル voice-convert が呼ぶ）。

  python -m studio.cli list
  python -m studio.cli convert --speaker ayumin --input 録音.m4a [--preset 部屋] [--semitone 0]
  python -m studio.cli speak   --speaker ayumin --text "こんにちは" [--emotion 嬉しい]
  python -m studio.cli sing    --speaker ayumin --input 曲.mp3 [--semitone 0] [--no-separate]

チャットからの自動作成は、本人の同意を記録した話者だけ使える。結果は mp3 で、最後の行に「OUTPUT=パス」を出す。
"""

import argparse
import subprocess
import sys

from studio import engine, post
from studio import speakers as spk
from studio.common import ffmpeg


def to_mp3(wav) -> str:
    from pathlib import Path

    mp3 = Path(wav).with_suffix(".mp3")
    subprocess.run([ffmpeg(), "-v", "error", "-y", "-i", str(wav), "-b:a", "160k", str(mp3)], check=True)
    return str(mp3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "convert", "speak", "sing"])
    ap.add_argument("--speaker")
    ap.add_argument("--input")
    ap.add_argument("--text")
    ap.add_argument("--emotion", default="")
    ap.add_argument("--preset", default="なし", choices=post.PRESETS)
    ap.add_argument("--lips", type=int, default=30)
    ap.add_argument("--semitone", type=int, default=0)
    ap.add_argument("--no-separate", action="store_true")
    ap.add_argument("--clean", action="store_true", help="（互換用。いまは最初から AI クリーンアップをかける）")
    ap.add_argument("--no-clean", action="store_true", help="AI クリーンアップをかけない")
    a = ap.parse_args()

    if a.cmd == "list":
        for sid in spk.list_ids():
            i = spk.info(sid)
            print(f"{sid}\t{i['name']}\t同意:{'あり' if spk.consent(sid) else 'なし'}\t声変換:{'○' if i['vc'] else '×'}")
        return 0

    sid = a.speaker
    if not sid or sid not in spk.list_ids():
        print(f"ERROR=話者「{sid}」が見つかりません（list で確認）")
        return 2
    try:
        spk.require_consent(sid, "チャットでの自動作成")
        if a.cmd == "convert":
            out, info = engine.convert(sid, a.input, semitone=a.semitone)
        elif a.cmd == "speak":
            from studio import natural

            nat = natural.settings(sid)  # 画面の読み上げで最後に使った自然さの設定（間・息継ぎなど）
            if a.emotion:
                out, info = engine.speak_emotion(sid, a.text, a.emotion, nat)
            else:
                ck = (engine.tts_checkpoints(sid) or [None])[0]
                out, info = engine.speak(sid, a.text, ck, nat=nat, best_of=engine.TTS_BEST_OF)
                out, info2 = engine.convert(sid, str(out))  # 読み上げのあと声変換で声を近づける
                info += f" → 声変換（{info2}）"
            out = engine.finish_natural(sid, out, nat)
        else:
            out, info = engine.sing(sid, a.input, separate=not a.no_separate, semitone=a.semitone)
        if a.cmd != "sing":
            out = post.apply(out, a.preset, lips=a.lips, ai_clean=not a.no_clean)
    except PermissionError as e:
        print(f"ERROR={e}")
        return 3
    finally:
        engine.stop_vc_server()
    print(f"INFO={info}")
    print(f"OUTPUT={to_mp3(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
