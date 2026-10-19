"""Qwen3-TTS を 1 人の声でファインチューニングする。

公式 third_party/Qwen3-TTS/finetuning/sft_12hz.py をもとに、Windows + 12GB GPU 向けに変更:
  - 【不具合修正】テキスト埋め込みに text_projection を通す（公式は通しておらず、推論とずれて壊れる）
  - 【不具合修正】損失のラベルが二重にずれていたのを直す（公式＋transformers の自動シフトで 2 つ先を学習していた）
  - 【不具合修正】細かい音（サブ）の予測を、推論と同じ位置の隠れ状態で学習する（公式は 1 つずれていて、
    学習率を上げると崩れた。2026-10-19 の精度検証で発見）
  - flash-attn なし（sdpa）、文字コードは UTF-8 固定
  - VRAM 節約: テキスト埋め込みを凍結、8bit AdamW、勾配チェックポイント
  - チェックポイントはベースモデルの重みをコピーせず、最新 N 個だけ残す
  - 進捗を status.json に書く
  - 精度検証（studio/bench.py）用の切り替え: 学習率の変え方（一定 / ウォームアップ＋コサイン）、
    話者の特徴を全クリップの平均にする、サブ損失の重み、まとめる量（勾配累積）

既定値は studio/train_defaults.json（精度検証で一番良かった設定。なければ下の値）。

  python -m studio.train_tts <声の名前> --epochs 10
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from studio.common import QWEN_FT, SPEAKER, VOICES, write_status

sys.path.insert(0, str(QWEN_FT))


def encode_codes(vdir, device):
    """clips.jsonl に音声コード（audio_codes）を付けて clips_codes.jsonl を作る。"""
    from qwen_tts import Qwen3TTSTokenizer

    from tts import fetch_model

    src, dst = vdir / "clips.jsonl", vdir / "clips_codes.jsonl"
    if dst.exists() and dst.stat().st_mtime > src.stat().st_mtime:
        return dst
    write_status(vdir, stage="tts", message="音声をコードに変換中")
    tok = Qwen3TTSTokenizer.from_pretrained(fetch_model("Qwen/Qwen3-TTS-Tokenizer-12Hz"), device_map=device)
    lines = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    for k in range(0, len(lines), 8):
        batch = lines[k:k + 8]
        res = tok.encode([b["audio"] for b in batch])
        for b, code in zip(batch, res.audio_codes):
            b["audio_codes"] = code.cpu().tolist()
        print(f"[tts] コード化 {min(k + 8, len(lines))}/{len(lines)}", flush=True)
    dst.write_text("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lines), encoding="utf-8")
    del tok
    torch.cuda.empty_cache()
    return dst


def save_checkpoint(model, base_path, out_dir, spk_embedding):
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    # 重み以外（設定・トークナイザ・speech_tokenizer）をコピー
    # 本体の重みは下で書き出すので、直下の model.safetensors だけ除く（speech_tokenizer/ の重みは必要）
    def skip(d, names):
        top = Path(d) == base_path
        return [n for n in names if n in (".cache", ".gitattributes") or n.endswith(".md")
                or (top and n == "model.safetensors")]

    shutil.copytree(base_path, out_dir, dirs_exist_ok=True, ignore=skip)
    cfg = json.loads((base_path / "config.json").read_text(encoding="utf-8"))
    cfg["tts_model_type"] = "custom_voice"
    talker = cfg.setdefault("talker_config", {})
    talker["spk_id"] = {SPEAKER: 3000}
    talker["spk_is_dialect"] = {SPEAKER: False}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    state = {k: v.detach().to("cpu") for k, v in model.state_dict().items() if not k.startswith("speaker_encoder")}
    w = state["talker.model.codec_embedding.weight"]
    w[3000] = spk_embedding[0].detach().to(w.dtype).cpu()
    save_file(state, str(out_dir / "model.safetensors"))


DEFAULTS_FILE = Path(__file__).resolve().parent / "train_defaults.json"
# 2026-10-15 時点の既定（ayumin で決めたもの）。精度検証の結果があればそちらを使う
BUILTIN = {"lr": 2e-6, "epochs": 3, "grad_accum": 4, "schedule": "const", "spk_avg": 0, "sub_weight": 0.3}


def tts_defaults() -> dict:
    try:
        saved = json.loads(DEFAULTS_FILE.read_text(encoding="utf-8")).get("tts", {})
    except (OSError, ValueError):
        saved = {}
    return {**BUILTIN, **{k: v for k, v in saved.items() if k in BUILTIN}}


@torch.no_grad()
def mean_speaker_embedding(model, dataset, data, n: int = 64):
    """話者の特徴を、クリップ（最大 n 本）それぞれから取って平均する。参照音声 1 本より声の特徴が安定する。"""
    import librosa

    embs = []
    step = max(1, len(data) // n)
    for item in data[::step][:n]:
        wav, _ = librosa.load(item["audio"], sr=24000, mono=True)
        mel = dataset.extract_mels(audio=wav.astype("float32"), sr=24000)
        embs.append(model.speaker_encoder(mel.to(model.device).to(model.dtype)).float())
    return torch.cat(embs).mean(dim=0, keepdim=True)


def main() -> int:
    from accelerate import Accelerator
    from dataset import TTSDataset
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    from transformers import AutoConfig

    from tts import fetch_model

    d = tts_defaults()
    ap = argparse.ArgumentParser()
    ap.add_argument("voice")
    ap.add_argument("--size", choices=["0.6B", "1.7B"], default="1.7B")
    ap.add_argument("--epochs", type=int, default=d["epochs"])
    ap.add_argument("--lr", type=float, default=d["lr"])
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=d["grad_accum"], help="何バッチ分まとめて 1 回更新するか")
    ap.add_argument("--schedule", choices=["const", "cosine"], default=d["schedule"],
                    help="学習率の変え方。cosine = 最初の 1 割で徐々に上げ、あとは徐々に下げる")
    ap.add_argument("--spk-avg", type=int, default=d["spk_avg"],
                    help="1 なら、話者の特徴を参照音声 1 本ではなく全クリップ（最大 64 本）の平均にする")
    ap.add_argument("--sub-weight", type=float, default=d["sub_weight"], help="細かい音（サブ）の損失の重み")
    ap.add_argument("--sub-align", choices=["next", "own"], default="next",
                    help="サブの予測に使う隠れ状態の位置。next = 推論と同じ（修正版）、own = 公式と同じ（比較用）")
    ap.add_argument("--keep", type=int, default=2, help="残すチェックポイント数")
    ap.add_argument("--optim", choices=["adamw8bit", "adamw"], default="adamw8bit")
    ap.add_argument("--no-freeze", action="store_true", help="テキスト埋め込みも学習する（公式と同じ）")
    ap.add_argument("--no-gc", action="store_true", help="勾配チェックポイントを使わない（公式と同じ）")
    ap.add_argument("--out", default="tts", help="保存先フォルダ名（voices/<名前>/ 以下）")
    args = ap.parse_args()

    vdir = VOICES / args.voice
    out_root = vdir / args.out
    train_jsonl = encode_codes(vdir, "cuda:0")

    base = Path(fetch_model(f"Qwen/Qwen3-TTS-12Hz-{args.size}-Base"))

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum, mixed_precision="bf16")
    qwen = Qwen3TTSModel.from_pretrained(str(base), torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    config = AutoConfig.from_pretrained(str(base))

    data = [json.loads(l) for l in train_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    dataset = TTSDataset(data, qwen.processor, config)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    model = qwen.model
    # 大きいテキスト埋め込み（語彙 15 万）と話者エンコーダは学習しない → メモリ節約
    if not args.no_freeze:
        for name, p in model.named_parameters():
            if name.startswith("speaker_encoder") or "text_embedding" in name:
                p.requires_grad_(False)
    if not args.no_gc:
        try:
            model.talker.gradient_checkpointing_enable()
        except Exception as e:  # 対応していなければそのまま
            print(f"[tts] gradient checkpointing 無効: {e}", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optim == "adamw8bit":
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    print(f"[tts] size={args.size} optim={args.optim} freeze={not args.no_freeze} gc={not args.no_gc} lr={args.lr} "
          f"schedule={args.schedule} grad_accum={args.grad_accum} spk_avg={args.spk_avg} sub_weight={args.sub_weight} "
          f"sub_align={args.sub_align}",
          flush=True)

    fixed_spk = None
    if args.spk_avg:
        model.to("cuda:0")
        fixed_spk = mean_speaker_embedding(model, dataset, data).to(torch.bfloat16)
        print(f"[tts] 話者の特徴を {min(64, len(data))} クリップの平均にしました", flush=True)

    total_steps = args.epochs * len(loader)
    scheduler = None
    if args.schedule == "cosine":
        from transformers import get_cosine_schedule_with_warmup

        updates = max(1, total_steps // args.grad_accum)
        scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, updates // 10), updates)

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)
    model.train()
    spk_embedding = fixed_spk
    t0 = time.time()
    done = 0

    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                codec_ids = batch["codec_ids"]
                text_mask = batch["text_embedding_mask"]
                codec_emb_mask = batch["codec_embedding_mask"]
                codec_mask = batch["codec_mask"]

                if fixed_spk is not None:
                    speaker_embedding = fixed_spk.to(model.dtype).expand(input_ids.shape[0], -1)
                else:
                    speaker_embedding = model.speaker_encoder(
                        batch["ref_mels"].to(model.device).to(model.dtype)).detach()
                if spk_embedding is None:
                    spk_embedding = speaker_embedding

                # 公式 sft_12hz.py は text_projection を通していないが、推論（generate）では必ず通す。
                # 学習と推論で入力がずれてモデルが壊れる（話し終わらない・雑音）ので、ここで通す
                emb = model.talker.text_projection(model.talker.model.text_embedding(input_ids[:, :, 0])) * text_mask
                codec_emb = model.talker.model.codec_embedding(input_ids[:, :, 1]) * codec_emb_mask
                codec_emb[:, 6, :] = speaker_embedding
                emb = emb + codec_emb
                for i in range(1, 16):
                    e = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    emb = emb + e * codec_mask.unsqueeze(-1)

                # 【不具合修正】公式はラベルを 1 つずらして model に渡すが、transformers の損失関数
                # （ForCausalLMLoss）も内部でさらに 1 つずらすため「2 つ先」を学習してしまい、モデルが壊れる。
                # ずらし済みのラベルで、ずらさない交差エントロピーを自分で計算する
                out = model.talker(
                    inputs_embeds=emb[:, :-1, :], attention_mask=batch["attention_mask"][:, :-1],
                    output_hidden_states=True,
                )
                labels = batch["codec_0_labels"][:, 1:]
                talker_loss = F.cross_entropy(out.logits.float().reshape(-1, out.logits.shape[-1]),
                                              labels.reshape(-1), ignore_index=-100)
                # 【不具合修正 2026-10-19・精度検証で発見】細かい音（サブ）の予測に使う隠れ状態の位置。
                # 推論では「位置 p の隠れ状態（次のフレームの基本の音を予測したもの）」で次のフレームの細かい音を作る。
                # 公式は codec_mask[:, :-1]（自分の入力フレームと組）で、入力に答えが入っているため写すことを覚えてしまい、
                # 学習率を上げると読み上げが崩れた（1e-5 で 1 周目から読めない）。codec_mask[:, 1:] で 1 つ前の位置と組にする
                align = codec_mask[:, 1:] if args.sub_align == "next" else codec_mask[:, :-1]
                hidden = out.hidden_states[0][-1][align]
                sub_codes = codec_ids[codec_mask]
                sub_logits, _ = model.talker.forward_sub_talker_finetune(sub_codes, hidden)  # 返る損失はずれているので使わない
                sub_loss = F.cross_entropy(sub_logits.float().reshape(-1, sub_logits.shape[-1]),
                                           sub_codes[:, 1:].reshape(-1))
                loss = talker_loss + args.sub_weight * sub_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            done += 1
            if step % 10 == 0:
                eta = (time.time() - t0) / done * (total_steps - done)
                msg = f"文章読み上げの学習 {epoch + 1}/{args.epochs} 周目 step {step}/{len(loader)} loss {loss.item():.3f}"
                print(f"[tts] {msg}", flush=True)
                write_status(vdir, stage="tts", message=msg, progress=round(done / total_steps, 3),
                             eta_min=round(eta / 60))

        if accelerator.is_main_process:
            ckpt = out_root / f"epoch-{epoch + 1:03d}"
            save_checkpoint(accelerator.unwrap_model(model), base, ckpt, spk_embedding)
            for old in sorted(out_root.glob("epoch-*"))[:-args.keep]:
                shutil.rmtree(old, ignore_errors=True)
            write_status(vdir, tts_checkpoint=str(ckpt))
            print(f"[tts] 保存 {ckpt}", flush=True)

    write_status(vdir, stage="tts_done", message="文章読み上げの学習が完了")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
