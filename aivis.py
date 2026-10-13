"""AivisSpeech / VOICEVOX エンジンの HTTP API を叩く簡易クライアント（標準ライブラリのみ）。

事前にアプリ（AivisSpeech か VOICEVOX）を起動しておくと、エンジンがローカルで待ち受けます。
  AivisSpeech: http://127.0.0.1:10101
  VOICEVOX   : http://127.0.0.1:50021   （--engine voicevox）

  python aivis.py --list                        # 話者とスタイル ID 一覧
  python aivis.py "こんにちは" --style <スタイルID> -o out.wav
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request

ENGINES = {"aivis": "http://127.0.0.1:10101", "voicevox": "http://127.0.0.1:50021"}


def call(base: str, path: str, method: str = "GET", params=None, body=None) -> bytes:
    url = base + path + ("?" + urllib.parse.urlencode(params) if params else "")
    data = json.dumps(body).encode() if body is not None else (b"" if method == "POST" else None)
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main() -> int:
    p = argparse.ArgumentParser(description="AivisSpeech / VOICEVOX ローカル API クライアント")
    p.add_argument("text", nargs="?", help="読み上げるテキスト")
    p.add_argument("--engine", choices=ENGINES.keys(), default="aivis")
    p.add_argument("--style", type=int, default=None, help="スタイル ID（省略時は先頭の話者）")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("-o", "--out", default="out.wav")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    base = ENGINES[args.engine]
    try:
        speakers = json.loads(call(base, "/speakers"))
    except OSError as e:
        print(f"[error] {base} に接続できません。アプリを起動してください: {e}", file=sys.stderr)
        return 1

    if args.list or not args.text:
        for s in speakers:
            for st in s["styles"]:
                print(f"{st['id']:>12}  {s['name']} / {st['name']}")
        return 0

    style = args.style if args.style is not None else speakers[0]["styles"][0]["id"]
    query = json.loads(call(base, "/audio_query", "POST", {"text": args.text, "speaker": style}))
    query["speedScale"] = args.speed
    wav = call(base, "/synthesis", "POST", {"speaker": style}, body=query)
    with open(args.out, "wb") as f:
        f.write(wav)
    print(f"[done] {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
