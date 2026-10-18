"""Voice Studio Live（PC 専用の拡張画面）: リアルタイム声変換を通話アプリのマイクに流す。

  python -m studio.live     →  http://127.0.0.1:7880（この PC だけ。スマホ・ネットからは開けない）

- 話者データは Voice Studio と同じ voices/ をそのまま使う
- 「本人の同意」の記録がある話者だけ使える（記録は話者ごとに meta.json の consent に保存）
- 変換した声は仮想マイク（VB-CABLE）に流し、通話アプリのマイクに「CABLE Output」を選んで使う
- 変換そのものは Seed-VC 付属のリアルタイム変換ツール（real-time-gui.py）で行う。
  この画面は、同意の確認・話者と機器の選択・設定の書き込み・起動だけを受け持つ
"""

import json
import subprocess
import time
from pathlib import Path

import gradio as gr

from studio import speakers as spk
from studio.common import ROOT, SEEDVC, VOICES, child_env

PORT = 7880
RT_CONFIG = SEEDVC / "configs" / "inuse" / "config.json"
VB_CABLE_URL = "https://vb-audio.com/Cable/"
HOSTAPI = "MME"

NOTICE = """
## ⚠ 使う前に必ずお読みください

このツールは、話している声をリアルタイムで別の人の声に変えて、**通話アプリなどに流す**ためのものです。

- **本人の同意がない声は使用できません。**
  実在の人の声を、本人の同意なく使うことは禁止です。相手にそれと知らせずに使うと、
  **なりすまし・詐欺・名誉毀損**などの法的な問題になるおそれがあり、通話アプリの利用規約にも反します。
- 使えるのは、**本人から「リアルタイム変換・通話で使ってよい」という同意を得た声**だけです。
  同意は話者ごとに記録し、記録のない話者は選べません。
- 同意の記録がない話者は **「試聴のみ」**: ヘッドホン・スピーカーで自分で聞いて品質を確かめるだけに使えます。
  仮想マイク・ループバックなど、他のアプリに渡せる出力には流せません。
- 自分の声の加工や、相手も声が変わっていると分かっている場面で使ってください。
- 使った日時は話者ごとに記録されます（`voices/<話者>/live_log.txt`）。
"""


# ---------------------------------------------------------------- 同意の記録

consent = spk.consent  # 同意の記録は Voice Studio と共通（voices/<話者>/meta.json）


def speaker_rows():
    rows = []
    for sid in spk.list_ids():
        c = consent(sid)
        rows.append([spk.meta(sid)["name"], "✅ 同意あり" if c else "⛔ 同意の記録なし（使えません）",
                     f"{c['person']} / {c['date']} / {c['method']}" if c else ""])
    return rows


# 他のアプリにマイクとして渡せてしまう出力（仮想マイク・ループバック）。同意のない声はここに流せない
ROUTING_WORDS = ("CABLE", "Loop", "loop", "Virtual", "VoiceMeeter", "Voicemeeter", "VB-Audio", "ステレオ ミキサー", "Stereo Mix")


def is_routing(device: str) -> bool:
    return any(w in (device or "") for w in ROUTING_WORDS)


def usable_choices():
    """同意の記録がある話者は「通話OK」、ない話者は「試聴のみ」（自分で聞くだけ）として選べる。"""
    return [(spk.meta(s)["name"] + ("（通話OK）" if consent(s) else "（試聴のみ・同意なし）"), s) for s in spk.list_ids()]


def all_choices():
    return [(spk.meta(s)["name"], s) for s in spk.list_ids()]


def save_consent(sid, person, date, method, note, sure):
    if not sid:
        raise gr.Error("話者を選んでください")
    if not sure:
        raise gr.Error("「本人から同意を得ています」にチェックを入れてください")
    try:
        spk.set_consent(sid, person, date, method, note)
    except ValueError as e:
        raise gr.Error(str(e))
    gr.Info(f"「{spk.meta(sid)['name']}」の同意を記録しました")
    return speaker_rows(), gr.update(choices=usable_choices()), False


def revoke_consent(sid):
    if not sid:
        raise gr.Error("話者を選んでください")
    spk.clear_consent(sid)
    gr.Info(f"「{spk.meta(sid)['name']}」の同意の記録を取り消しました（もう使えません）")
    return speaker_rows(), gr.update(choices=usable_choices(), value=None)


# ---------------------------------------------------------------- 機器

def devices():
    """MME の入力（マイク）・出力の一覧。リアルタイム変換ツールと同じ名前で返す。"""
    import sounddevice as sd

    sd._terminate()
    sd._initialize()
    ha = next(h for h in sd.query_hostapis() if h["name"] == HOSTAPI)
    devs = sd.query_devices()
    ins = [devs[i]["name"] for i in ha["devices"] if devs[i]["max_input_channels"] > 0]
    outs = [devs[i]["name"] for i in ha["devices"] if devs[i]["max_output_channels"] > 0]
    return ins, outs


def device_updates():
    ins, outs = devices()
    cable = next((o for o in outs if "CABLE Input" in o), None)
    mic = next((i for i in ins if "CABLE" not in i and "マッパー" not in i), ins[0] if ins else None)
    if cable:
        msg = ("✅ 仮想マイク（VB-CABLE）が見つかりました。\n通話アプリのマイクに **「CABLE Output (VB-Audio Virtual Cable)」** を選ぶと、"
               "変換した声が相手に届きます。")
    else:
        msg = (f"⚠ 仮想マイク（VB-CABLE）が入っていません。通話アプリに流すには [VB-CABLE（無料）]({VB_CABLE_URL}) を"
               "**ご自身でインストール**して PC を再起動し、「機器を読み直す」を押してください。"
               "（入れるまでは、出力をヘッドホンにして自分で聞く確認だけできます）")
    return gr.update(choices=ins, value=mic), gr.update(choices=outs, value=cable or (outs[0] if outs else None)), msg


# ---------------------------------------------------------------- 起動

_proc = {"p": None}


def running() -> bool:
    return _proc["p"] is not None and _proc["p"].poll() is None


# 設定の組み合わせ。遅れ ≒ 処理の区切り + 後ろを見る長さ + つなぎ目。変換にかかる時間が「区切り」を超えると途切れる
PRESETS = {
    "低遅延": dict(block_time=0.2, crossfade_length=0.04, extra_time_ce=2.0, extra_time=0.5, extra_time_right=0.02,
                diffusion_steps=6, max_prompt_length=3.0),
    "標準": dict(block_time=0.3, crossfade_length=0.06, extra_time_ce=2.5, extra_time=0.8, extra_time_right=0.05,
               diffusion_steps=10, max_prompt_length=4.0),
    "なめらか・リアル": dict(block_time=0.5, crossfade_length=0.1, extra_time_ce=2.5, extra_time=1.2, extra_time_right=0.1,
                       diffusion_steps=15, max_prompt_length=5.0),
}
PRESET_NOTES = {
    "低遅延": "遅れ 約0.4秒。会話向き。少しぶつ切れ感が出ることがあります",
    "標準": "遅れ 約0.6秒。バランス型",
    "なめらか・リアル": "遅れ 約0.9秒。つなぎ目がなめらかで自然。ゲーム実況・配信向き",
}


def start(agreed, sid, mic, out, preset, gate_db=-45):
    if not agreed:
        raise gr.Error("最初の注意を読んで、同意してください")
    if not sid:
        raise gr.Error("話者を選んでください")
    listen_only = not consent(sid)
    if listen_only and is_routing(out):
        # 同意の記録がない声は、通話アプリなどに渡せる出力（仮想マイク等）には流さない。
        # 自分で聞いて品質を確かめる「試聴のみ」なら使える
        raise gr.Error("この話者は本人の同意の記録がないため「試聴のみ」です。出力をヘッドホンかスピーカーにしてください"
                       "（仮想マイク・ループバックには流せません）")
    if running():
        raise gr.Error("すでに起動しています")
    vdir = VOICES / sid
    from studio.engine import _vc_ref

    ref = _vc_ref(sid)  # 最適化で選んだ参照音声（なければ標準のもの）
    RT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "reference_audio_path": str(ref), "sg_hostapi": HOSTAPI, "sg_wasapi_exclusive": False,
        "sg_input_device": mic, "sg_output_device": out, "sr_type": "sr_model", "inference_cfg_rate": 0.7,
        **PRESETS.get(preset, PRESETS["標準"]),
    }
    RT_CONFIG.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    # ノイズゲート（studio/rt_ja.py が変換ツールに組み込む）。-70 はオフ
    extra = {"noise_gate_db": None if gate_db <= -70 else float(gate_db)}
    (RT_CONFIG.parent / "live_extra.json").write_text(json.dumps(extra), encoding="utf-8")
    with open(vdir / "live_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} 開始{'（試聴のみ）' if listen_only else ''} "
                f"設定={preset} 入力={mic} 出力={out}\n")
    cmd = [str(ROOT / ".venv-vc" / "Scripts" / "python.exe"), str(ROOT / "studio" / "vc_run.py"), "real-time-gui.py"]
    model = spk.rt_model(sid)
    if model:  # この話者の声で学習したリアルタイム用モデル（ないときは汎用モデル）
        cmd += ["--checkpoint-path", str(model[0]), "--config-path", str(model[1])]
    _proc["p"] = subprocess.Popen(cmd, cwd=SEEDVC, env=child_env(), creationflags=subprocess.CREATE_NEW_CONSOLE)
    note = ("\n\n🎧 **試聴のみ**: 同意の記録がない話者なので、ヘッドホン・スピーカーで自分で聞く確認だけにしてください。"
            "（変換ツールの画面で出力を仮想マイクに変えて、通話などに使うのは禁止です）") if listen_only else ""
    return ("▶ リアルタイム変換ツールを開きました。開いた画面の **「Start」** を押すと変換が始まります。"
            "（初回はモデルの準備に数分かかります。設定はこの画面で選んだものが入っています）" + note)


def stop():
    if running():
        subprocess.run(["taskkill", "/PID", str(_proc["p"].pid), "/T", "/F"], capture_output=True)
        _proc["p"] = None
        return "■ 止めました"
    return "起動していません"


# ---------------------------------------------------------------- 画面

with gr.Blocks(title="Voice Studio Live") as demo:
    gr.Markdown("# 🔴 Voice Studio Live\n話しながら、その場で声を変える（PC 専用）。")
    with gr.Column(elem_classes="notice") as gate:
        gr.Markdown(NOTICE)
        agree = gr.Checkbox(False, label="上の内容を理解し、本人の同意を得た声だけを使います")
        agree_btn = gr.Button("同意して始める", variant="primary", elem_classes="big")
    agreed = gr.State(False)

    with gr.Column(visible=False) as main:
        # ---- STEP1
        gr.Markdown("### STEP1　話者を選ぶ", elem_classes="step")
        with gr.Column(elem_classes="card"):
            sid = gr.Dropdown(usable_choices(), label="話者", value=None)
            mode_md = gr.Markdown("話者を選んでください")
            with gr.Accordion("📝 同意の記録（記録すると通話にも使えるようになります）", open=False):
                table = gr.Dataframe(speaker_rows(), headers=["話者", "同意", "記録（誰から / 日付 / 方法）"],
                                     interactive=False, wrap=True)
                gr.Markdown("本人から「この声をリアルタイム変換・通話で使ってよい」と同意を得たときだけ記録してください。",
                            elem_classes="muted")
                c_sid = gr.Dropdown(all_choices(), label="話者")
                with gr.Row():
                    c_person = gr.Textbox(label="同意した人（本人の名前）")
                    c_date = gr.Textbox(label="同意を得た日", placeholder="2026-10-15")
                    c_method = gr.Dropdown(["書面", "メール・メッセージ", "口頭（記録あり）", "その他"], label="方法")
                c_note = gr.Textbox(label="メモ（同意の範囲・保存場所など）")
                c_sure = gr.Checkbox(False, label="この人本人から、上の内容で同意を得ています")
                with gr.Row():
                    c_save = gr.Button("同意を記録する", variant="primary")
                    c_revoke = gr.Button("同意の記録を取り消す", variant="stop")

        # ---- STEP2
        gr.Markdown("### STEP2　マイクと出力を選ぶ", elem_classes="step")
        with gr.Column(elem_classes="card"):
            with gr.Row():
                mic = gr.Dropdown(label="🎤 入力（自分のマイク）")
                out = gr.Dropdown(label="🔈 出力（試聴ならヘッドホン / 通話なら CABLE Input）")
            dev_msg = gr.Markdown(elem_classes="muted")
            reload_dev = gr.Button("機器を読み直す", size="sm")
            preset = gr.Radio(list(PRESETS), value="標準", label="設定（ぶつ切れ感が気になるときは「なめらか・リアル」）")
            preset_md = gr.Markdown(PRESET_NOTES["標準"], elem_classes="muted")
            gate = gr.Slider(-70, -25, value=-45, step=1,
                             label="ノイズゲート（dB）: この音量より小さい音は無音にする。雑音で変な声が出るなら上げる（-70 でオフ）")
            with gr.Accordion("🎧 通話しながら、変換後の声を自分でも聞くには", open=False):
                gr.Markdown("出力を CABLE Input にしているときは、Windows の設定で聞けます（PC の設定の変更なので、ご自身で行ってください）。\n\n"
                            "1. 設定 → システム → サウンド → 「サウンドの詳細設定」\n"
                            "2. 「録音」タブ → **CABLE Output** を右クリック → プロパティ → 「聴く」タブ\n"
                            "3. **「このデバイスを聴く」にチェック** → 再生するデバイスにヘッドホンを選んで OK\n\n"
                            "やめるときは同じチェックを外してください。", elem_classes="muted")

        # ---- STEP3
        gr.Markdown("### STEP3　スタート", elem_classes="step")
        with gr.Column(elem_classes="card"):
            with gr.Row():
                start_btn = gr.Button("▶ リアルタイム変換を開く", variant="primary", elem_classes="big", scale=3)
                stop_btn = gr.Button("■ 止める", variant="stop", elem_classes="big", scale=1)
            state_msg = gr.Markdown()
            gr.Markdown("開いた小さな画面（Seed-VC）で **Start** を押すと変換が始まります。"
                        "通話に使うとき（同意のある話者のみ）は、通話アプリのマイクを **CABLE Output** にしてください。",
                        elem_classes="muted")

    def _agree(ok):
        if not ok:
            raise gr.Error("チェックを入れてください")
        return True, gr.update(visible=False), gr.update(visible=True), *device_updates()

    def _mode(s):
        if not s:
            return "話者を選んでください"
        model = "🎯 この声で学習したリアルタイム用モデルを使います" if spk.rt_model(s) else \
            "ℹ リアルタイム用モデルは未学習（汎用モデルで動きます。声の似方は控えめ）"
        if consent(s):
            return "📞 **通話OK**（本人の同意が記録されています）\n\n" + model
        return "🎧 **試聴のみ**（同意の記録がないため、出力はヘッドホン・スピーカーだけ）\n\n" + model

    agree_btn.click(_agree, agree, [agreed, gate, main, mic, out, dev_msg])
    sid.change(_mode, sid, mode_md)
    reload_dev.click(device_updates, None, [mic, out, dev_msg])
    c_save.click(save_consent, [c_sid, c_person, c_date, c_method, c_note, c_sure], [table, sid, c_sure])
    c_revoke.click(revoke_consent, c_sid, [table, sid])
    start_btn.click(start, [agreed, sid, mic, out, preset, gate], state_msg)
    preset.change(lambda p: PRESET_NOTES[p], preset, preset_md)
    stop_btn.click(stop, None, state_msg)


if __name__ == "__main__":
    import sys

    from studio import style

    # この PC の中だけで待ち受ける（Tailscale の中継もしない）
    demo.queue().launch(server_name="127.0.0.1", server_port=PORT, inbrowser="--no-browser" not in sys.argv,
                        allowed_paths=[str(VOICES.resolve())], css=style.CSS, theme=style.theme("red"))
