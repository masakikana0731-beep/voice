# voice — ローカル音声合成環境

2026年10月時点のおすすめ構成を、PC にそのまま入れられる形でまとめています。

| 用途 | おすすめ | 必要スペック | ライセンス |
| --- | --- | --- | --- |
| 日本語を手軽に高品質で読み上げたい | **AivisSpeech**（GUI アプリ） | CPU でも可 | アプリは LGPL、音声モデルごとに規約あり |
| 声のクローン・声質を文章で指定・感情指示 | **Qwen3-TTS**（Python） | NVIDIA GPU 推奨（VRAM 8GB〜） | Apache-2.0 |

- 迷ったら：**GPU なし → AivisSpeech**、**NVIDIA GPU あり → 両方**。
- 定番の VOICEVOX も同じ API で使えます（`aivis.py --engine voicevox`）。

## 0. Voice Studio（声を学習して使う UI）

**`start-studio.bat` をダブルクリック** → ブラウザで http://127.0.0.1:7860 が開きます。

| タブ | すること | 中身 |
| --- | --- | --- |
| ⓪ 動画から声を抽出 | 動画・音声ファイルか YouTube の URL → 声だけにして話者ごとに分け、残す人を選んで学習データへ | yt-dlp / audio-separator（MelBand Roformer で分離・残響除去・雑音除去）/ ECAPA で話者分け |
| ① 声を学習 | なりたい声の録音（合計30分〜1時間、最低10分）を入れて「学習スタート」 | 自動で区切り・書き起こし → 2つのモデルを学習 |
| ② 文章から読み上げ | 文章を入れると学習した声で読む | Qwen3-TTS 1.7B をファインチューニング |
| ③ 自分の声で抑揚指定 | 自分で読んだ録音を学習した声に変換。抑揚・間はそのまま | Seed-VC（F0 条件つき 44kHz）をファインチューニング |

- 学習は裏で動くので、UI を閉じても続きます（「止める」で中断）。状態は 5 秒ごとに更新。
- 学習前でも ②③ は「参照音声からのクローン」として試せます（① で録音を入れ、データ準備が終わった後）。
- データ・モデルは `voices/<名前>/`、声変換の学習結果は `third_party/seed-vc/runs/<名前>/`、出力は `outputs/<名前>/`。
- 初回セットアップ: `setup\setup.ps1` → `setup\setup-studio.ps1`。

## 1. AivisSpeech（いちばん簡単）

1. https://aivis-project.com/ からインストーラをダウンロードして起動
2. そのまま GUI で読み上げ可能。スクリプトから使う場合はアプリを起動したまま:

```bash
python aivis.py --list                       # 話者・スタイルID一覧
python aivis.py "こんにちは" --style <ID> -o out.wav
```

`aivis.py` は標準ライブラリのみなので、Python があればそのまま動きます。

## 2. Qwen3-TTS（クローン・声デザイン）

前提: Git と（Windows なら）PowerShell。Python は `uv` が自動で 3.12 を入れます。

```bash
git clone <このリポジトリ> voice && cd voice

# Windows
powershell -ExecutionPolicy Bypass -File setup\setup.ps1
# macOS / Linux / WSL
bash setup/setup.sh
```

初回実行時にモデル（0.6B で約 2.4GB。1.7B はさらに大きく、未計測）を Hugging Face から自動ダウンロードします。

```bash
# 内蔵の日本語話者（Ono_Anna）で読み上げ
python tts.py speak "こんにちは、ローカル音声合成のテストです。"

# 話し方を指示
python tts.py speak "やった、ついに完成したよ！" --instruct "とても嬉しそうに"

# 声を文章で設計（1.7B のみ）
python tts.py design "本日のニュースをお伝えします。" --instruct "落ち着いた30代男性のアナウンサー"

# 声のクローン（参照音声 3〜15 秒程度 + その書き起こし）
python tts.py clone "クローンした声で話します。" --ref-audio samples/me.wav --ref-text "参照音声で話している内容"

# 長文はファイルで
python tts.py speak @script.txt -o outputs/narration.wav

# Web UI
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --port 8000   # → http://127.0.0.1:8000
```

主なオプション: `--size 0.6B`（軽量版）、`--device cpu|cuda:0|mps`、`--list-speakers`。
出力は既定で `outputs/` に保存されます。

### メモ

- 起動時の `SoX could not be found` / `flash-attn is not installed` 警告は無視して問題ありません（12Hz モデルでは SoX 不使用）。
- CPU のみだと 0.6B でも実時間の 4〜7 倍程度かかります（クラウドの CPU 環境で計測）。実用は GPU 推奨。
- 他人の声のクローンは本人の同意を得たうえで使ってください。
