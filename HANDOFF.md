# 引き継ぎメモ（voice プロジェクト）

新しい作業場所（Claude セッション）で最初にこのファイルを読ませてください。

## 1. やりたいこと（ゴール）

1. **自宅のWindows PCでローカル音声合成（TTS）を動かす**
   - 日本語の読み上げ、声のクローン、声質の指定、感情の指示
2. **PC上の Claude をスマホから遠隔操作する（Remote Control）**
   - スマホから「音声を作って」と頼めば、PC上で作業が進む状態にする
3. ユーザーは手順が苦手なので、**できるだけ自動化・ワンクリック化**する

## 2. 決めたこと

| 用途 | 採用 | 理由 |
| --- | --- | --- |
| 手軽な日本語読み上げ | AivisSpeech（GUIアプリ、https://aivis-project.com/） | CPUでも動き、導入が簡単 |
| クローン・声質の指定・感情 | Qwen3-TTS（Apache-2.0、2026年1月公開、`pip install qwen-tts`） | 高機能で商用利用も可 |
| 遠隔操作 | Claude Code の Remote Control（`claude rc` / `/rc`） | スマホの Claude アプリから操作できる |

- 想定：Windows。GPU は未確認（NVIDIA があれば Qwen3-TTS が実用的。なければ AivisSpeech が中心）

## 3. できているもの（GitHub: masakikana0731-beep/voice、ブランチ `ccr-96507d6e-dwl2v2`）

| ファイル | 内容 |
| --- | --- |
| `README.md` | 使い方の説明 |
| `tts.py` | Qwen3-TTS の CLI。`speak`（内蔵話者 Ono_Anna）/ `design`（文章で声を指定）/ `clone`（声のクローン）。GPU を自動で判定する |
| `aivis.py` | AivisSpeech（ポート 10101）/ VOICEVOX（ポート 50021）の API クライアント。標準ライブラリのみで動く |
| `setup/setup.ps1` | Windows 用の環境構築。uv で Python 3.12 を入れ、GPU に合わせた PyTorch と qwen-tts をインストールする |
| `setup/setup.sh` | Mac / Linux 用の同じスクリプト |
| `setup/start-remote.bat` | **ダブルクリックで** Git と Claude Code をインストール → リポジトリを clone（`%USERPROFILE%\voice`）→ `claude rc` を実行 |
| `requirements.txt` | qwen-tts==0.1.1, soundfile |

## 4. 確認できたこと・できていないこと

- ✅ クラウドの Linux（CPU）で、Qwen3-TTS 0.6B の `speak` と `clone` が動作した
  - 生成時間は音声の長さの 4〜7 倍。モデルは約 2.4GB
- ❌ Windows では未実行：`setup.ps1` と `start-remote.bat`
- ❌ GPU での速度、1.7B モデル、`design` モードは未確認
- ❌ `aivis.py` は未実行（AivisSpeech のポート 10101 は記憶ベース）
- ❌ 未ログインの状態で `claude rc` が動くかは不明。bat では失敗時に `claude` を起動し、ログイン後に `/rc` を入力してもらう作りにしている

## 5. つまずいた点（ややこしくなった原因）

- 前のセッションは**クラウド上**で動いていたため、ユーザーのPCを直接操作できなかった
- ユーザーはチャットに `claude rc` と打てば PC で動くと思っていた。実際は **PC の PowerShell で実行する必要がある**
- → 新しい作業は、**PC 上で起動した Claude Code（Desktop アプリ、または `claude rc`）で行う**のが前提

## 6. 次にやること

1. PC で `setup/start-remote.bat` をダブルクリックする
   - GitHub の該当ブランチから `setup/start-remote.bat` をダウンロードすれば、clone は不要
2. claude.ai にログインし、表示された QR コードをスマホで読み取る
3. PC 上の Claude に次を依頼する
   - GPU の有無を確認（`nvidia-smi`）
   - `setup\setup.ps1` を実行
   - `python tts.py speak "こんにちは"` で動作を確認
4. 必要に応じて AivisSpeech をインストールし、`python aivis.py --list` で確認する
5. 作業ブランチを main に取り込む（必要であれば PR を作る）

## 7. ユーザーの好み

- 回答は簡潔に。長くなる場合は、最後に要点を箇条書きで付ける
- 確認できていない内容は、そう明示し、推測の根拠を説明する
- 操作手順はなるべく自動化し、ワンクリックにする
