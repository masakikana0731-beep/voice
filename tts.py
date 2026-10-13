"""Qwen3-TTS をローカルで動かす簡易 CLI。

使い方の例:
  # 内蔵話者で読み上げ（日本語ネイティブ話者 Ono_Anna）
  python tts.py speak "こんにちは、ローカル音声合成のテストです。"

  # 感情・話し方を指示
  python tts.py speak "やった、ついに完成したよ！" --instruct "とても嬉しそうに"

  # 声の説明文から新しい声を作る（1.7B のみ）
  python tts.py design "本日のニュースをお伝えします。" --instruct "落ち着いた30代男性のアナウンサー"

  # 参照音声（数秒〜十数秒）から声をクローン
  python tts.py clone "クローンした声で話します。" --ref-audio samples/me.wav --ref-text "参照音声で話している内容"
"""

import argparse
import sys
import time
from pathlib import Path

MODELS = {
    "speak": "Qwen/Qwen3-TTS-12Hz-{size}-CustomVoice",
    "design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "clone": "Qwen/Qwen3-TTS-12Hz-{size}-Base",
}


def pick_device(requested: str):
    import torch

    if requested != "auto":
        device = requested
    elif torch.cuda.is_available():
        device = "cuda:0"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if device.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        # CPU / Apple Silicon は float32 が無難（bf16 は環境によって遅い・不安定）
        dtype = torch.float32
    return device, dtype


def fetch_model(model_id: str) -> str:
    """models/<名前> にダウンロードしてそのパスを返す。

    HF 標準キャッシュはシンボリックリンクを作るため、権限のない Windows では失敗する。
    local_dir に実ファイルで置けばリンク不要で、途中で止まっても再実行で続きから落とせる。
    """
    from huggingface_hub import snapshot_download

    local = Path(__file__).parent / "models" / model_id.split("/")[-1]
    for attempt in range(1, 11):
        try:
            return snapshot_download(model_id, local_dir=local)
        except OSError as e:  # 回線が途中で切れても、落とした分は残るので再試行で続きから取れる
            if attempt == 10:
                raise
            print(f"[warn] ダウンロード中断（{attempt}回目）。再試行します: {e}", file=sys.stderr)
            time.sleep(5)


def load_model(model_id: str, device: str, dtype):
    from qwen_tts import Qwen3TTSModel

    model_id = fetch_model(model_id)
    kwargs = {"device_map": device, "dtype": dtype}
    if device.startswith("cuda"):
        try:
            import flash_attn  # noqa: F401

            kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            kwargs["attn_implementation"] = "sdpa"
    return Qwen3TTSModel.from_pretrained(model_id, **kwargs)


def main() -> int:
    p = argparse.ArgumentParser(description="Qwen3-TTS ローカル音声合成")
    p.add_argument("mode", choices=MODELS.keys(), help="speak=内蔵話者 / design=声を文章で指定 / clone=声のクローン")
    p.add_argument("text", help="読み上げるテキスト（ファイルを渡す場合は @path/to/file.txt）")
    p.add_argument("-o", "--out", default=None, help="出力 wav（省略時 outputs/<時刻>.wav）")
    p.add_argument("--size", choices=["0.6B", "1.7B"], default="1.7B", help="モデルサイズ（VRAM が少なければ 0.6B）")
    p.add_argument("--language", default="Japanese", help="Japanese / English / Auto など")
    p.add_argument("--speaker", default="Ono_Anna", help="speak モードの話者名（--list-speakers で一覧）")
    p.add_argument("--instruct", default=None, help="話し方の指示（speak/design）")
    p.add_argument("--ref-audio", default=None, help="clone モードの参照音声")
    p.add_argument("--ref-text", default=None, help="参照音声の書き起こし（省略すると品質が下がる）")
    p.add_argument("--device", default="auto", help="auto / cuda:0 / mps / cpu")
    p.add_argument("--list-speakers", action="store_true", help="話者と対応言語を表示して終了")
    args = p.parse_args()

    text = Path(args.text[1:]).read_text(encoding="utf-8") if args.text.startswith("@") else args.text

    model_id = MODELS[args.mode].format(size=args.size)
    device, dtype = pick_device(args.device)
    print(f"[info] model={model_id} device={device} dtype={dtype}", file=sys.stderr)

    t0 = time.time()
    model = load_model(model_id, device, dtype)
    print(f"[info] loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    if args.list_speakers:
        print("speakers :", model.get_supported_speakers())
        print("languages:", model.get_supported_languages())
        return 0

    t0 = time.time()
    if args.mode == "speak":
        wavs, sr = model.generate_custom_voice(
            text=text, language=args.language, speaker=args.speaker, instruct=args.instruct
        )
    elif args.mode == "design":
        if not args.instruct:
            p.error("design モードでは --instruct（声の説明）が必要です")
        wavs, sr = model.generate_voice_design(text=text, language=args.language, instruct=args.instruct)
    else:
        if not args.ref_audio:
            p.error("clone モードでは --ref-audio が必要です")
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=args.language,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            x_vector_only_mode=args.ref_text is None,
        )

    import soundfile as sf

    out = Path(args.out or f"outputs/{time.strftime('%Y%m%d-%H%M%S')}.wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, wavs[0], sr)
    dur = len(wavs[0]) / sr
    el = time.time() - t0
    print(f"[done] {out}  ({dur:.1f}s 音声 / 生成 {el:.1f}s, RTF={el / max(dur, 1e-6):.2f})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
