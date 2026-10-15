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
- 自分の声の加工や、相手も声が変わっていると分かっている場面で使ってください。
- 使った日時は話者ごとに記録されます（`voices/<話者>/live_log.txt`）。
"""


# ---------------------------------------------------------------- 同意の記録

def consent(sid: str) -> dict | None:
    return spk.meta(sid).get("consent")


def speaker_rows():
    rows = []
    for sid in spk.list_ids():
        c = consent(sid)
        rows.append([spk.meta(sid)["name"], "✅ 同意あり" if c else "⛔ 同意の記録なし（使えません）",
                     f"{c['person']} / {c['date']} / {c['method']}" if c else ""])
    return rows


def usable_choices():
    return [(spk.meta(s)["name"], s) for s in spk.list_ids() if consent(s)]


def all_choices():
    return [(spk.meta(s)["name"], s) for s in spk.list_ids()]


def save_consent(sid, person, date, method, note, sure):
    if not sid:
        raise gr.Error("話者を選んでください")
    if not sure:
        raise gr.Error("「本人から同意を得ています」にチェックを入れてください")
    if not person.strip() or not date.strip() or not method:
        raise gr.Error("同意した人・日付・方法を入れてください")
    spk.save_meta(sid, consent={"person": person.strip(), "date": date.strip(), "method": method,
                                "note": note.strip(), "recorded": time.strftime("%Y-%m-%d %H:%M")})
    gr.Info(f"「{spk.meta(sid)['name']}」の同意を記録しました")
    return speaker_rows(), gr.update(choices=usable_choices()), False


def revoke_consent(sid):
    if not sid:
        raise gr.Error("話者を選んでください")
    m = spk.meta(sid)
    if "consent" in m:
        del m["consent"]
        (VOICES / sid / "meta.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    gr.Info(f"「{m['name']}」の同意の記録を取り消しました（もう使えません）")
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


def start(agreed, sid, mic, out, steps, block):
    if not agreed:
        raise gr.Error("最初の注意を読んで、同意してください")
    if not sid or not consent(sid):
        raise gr.Error("本人の同意が記録された話者だけ使えます")
    if running():
        raise gr.Error("すでに起動しています")
    vdir = VOICES / sid
    from studio.engine import vc_reference

    ref = vc_reference(vdir)
    RT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "reference_audio_path": str(ref), "sg_hostapi": HOSTAPI, "sg_wasapi_exclusive": False,
        "sg_input_device": mic, "sg_output_device": out, "sr_type": "sr_model",
        "diffusion_steps": int(steps), "inference_cfg_rate": 0.7, "max_prompt_length": 3.0,
        "block_time": float(block), "crossfade_length": 0.04, "extra_time_ce": 2.5, "extra_time": 0.5,
        "extra_time_right": 0.02,
    }
    RT_CONFIG.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    with open(vdir / "live_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} 開始 入力={mic} 出力={out}\n")
    _proc["p"] = subprocess.Popen(
        [str(ROOT / ".venv-vc" / "Scripts" / "python.exe"), str(ROOT / "studio" / "vc_run.py"), "real-time-gui.py"],
        cwd=SEEDVC, env=child_env(), creationflags=subprocess.CREATE_NEW_CONSOLE)
    return ("▶ リアルタイム変換ツールを開きました。開いた画面の **「Start」** を押すと変換が始まります。"
            "（初回はモデルの準備に数分かかります。設定はこの画面で選んだものが入っています）")


def stop():
    if running():
        subprocess.run(["taskkill", "/PID", str(_proc["p"].pid), "/T", "/F"], capture_output=True)
        _proc["p"] = None
        return "■ 止めました"
    return "起動していません"


# ---------------------------------------------------------------- 画面

with gr.Blocks(title="Voice Studio Live") as demo:
    gr.Markdown("# 🔴 Voice Studio Live（PC 専用）\nリアルタイム声変換を、通話アプリなどのマイクに流します。")
    with gr.Group() as gate:
        gr.Markdown(NOTICE)
        agree = gr.Checkbox(False, label="上の内容を理解し、本人の同意を得た声だけを使います")
        agree_btn = gr.Button("同意して始める", variant="primary")
    agreed = gr.State(False)

    with gr.Column(visible=False) as main:
        gr.Markdown("### 1. 話者（同意の記録がある人だけ使えます）")
        table = gr.Dataframe(speaker_rows(), headers=["話者", "同意", "記録（誰から / 日付 / 方法）"],
                             interactive=False, wrap=True)
        with gr.Accordion("📝 同意を記録する・取り消す", open=False):
            gr.Markdown("本人から「この声をリアルタイム変換・通話で使ってよい」と同意を得たときだけ記録してください。")
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
        sid = gr.Dropdown(usable_choices(), label="使う話者", value=None)

        gr.Markdown("### 2. 機器")
        dev_msg = gr.Markdown()
        with gr.Row():
            mic = gr.Dropdown(label="入力（自分のマイク）")
            out = gr.Dropdown(label="出力（通話に流すなら CABLE Input）")
        reload_dev = gr.Button("機器を読み直す")
        with gr.Accordion("⚙ 詳しい設定", open=False):
            steps = gr.Slider(4, 30, value=10, step=1, label="品質（多いほど自然・遅れが増える）")
            block = gr.Slider(0.15, 1.0, value=0.3, step=0.05, label="処理の区切り（秒）。短いほど遅れが少ないが途切れやすい")

        gr.Markdown("### 3. 開始")
        with gr.Row():
            start_btn = gr.Button("▶ リアルタイム変換を開く", variant="primary")
            stop_btn = gr.Button("■ 止める", variant="stop")
        state_msg = gr.Markdown()
        gr.Markdown("**通話アプリ側の設定**（例: PC 版 LINE）: 設定 → 通話 → マイク を **CABLE Output (VB-Audio Virtual Cable)** にする。"
                    "スピーカーはいつもどおり。終わったらマイクを元に戻してください。")

    def _agree(ok):
        if not ok:
            raise gr.Error("チェックを入れてください")
        return True, gr.update(visible=False), gr.update(visible=True), *device_updates()

    agree_btn.click(_agree, agree, [agreed, gate, main, mic, out, dev_msg])
    reload_dev.click(device_updates, None, [mic, out, dev_msg])
    c_save.click(save_consent, [c_sid, c_person, c_date, c_method, c_note, c_sure], [table, sid, c_sure])
    c_revoke.click(revoke_consent, c_sid, [table, sid])
    start_btn.click(start, [agreed, sid, mic, out, steps, block], state_msg)
    stop_btn.click(stop, None, state_msg)


if __name__ == "__main__":
    import sys

    # この PC の中だけで待ち受ける（Tailscale の中継もしない）
    demo.queue().launch(server_name="127.0.0.1", server_port=PORT, inbrowser="--no-browser" not in sys.argv,
                        allowed_paths=[str(VOICES.resolve())])
