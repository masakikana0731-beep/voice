"""Voice Studio（ブラウザで使う UI）。

  python -m studio.app        →  http://127.0.0.1:7860 が開く

  ① 声を学習       なりたい声の録音を入れて、長時間学習させる
  ② 文章から読み上げ 学習した声で文章を読む
  ③ 自分の声で抑揚指定 自分で読んだ録音を、学習した声に変換（抑揚・間はそのまま）
"""

import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

import gradio as gr

from studio import engine
from studio.common import AUDIO_EXT, PY, ROOT, child_env, list_voices, read_status, safe_name, voice_dir, write_status


def _alive(pid) -> bool:
    if not pid:
        return False
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    k.GetExitCodeProcess(h, ctypes.byref(code))
    k.CloseHandle(h)
    return code.value == 259  # STILL_ACTIVE


def raw_summary(voice):
    if not voice:
        return "声を選ぶか、新しく作ってください"
    files = [p for p in (voice_dir(voice) / "raw").iterdir() if p.suffix.lower() in AUDIO_EXT]
    mb = sum(p.stat().st_size for p in files) / 1e6
    names = "、".join(p.name for p in files[:8]) + (" ほか" if len(files) > 8 else "")
    return f"録音ファイル: {len(files)} 個（{mb:.0f}MB）\n{names}"


def create_voice(name):
    if not name.strip():
        raise gr.Error("名前を入れてください")
    v = safe_name(name)
    voice_dir(v)
    return gr.update(choices=list_voices(), value=v), ""


def add_files(voice, files):
    if not voice:
        raise gr.Error("先に声を選ぶか作ってください")
    for f in files or []:
        shutil.copy(f, voice_dir(voice) / "raw" / Path(f).name)
    return raw_summary(voice), None


def start_training(voice, size, tts_epochs, vc_steps, do_tts, do_vc):
    if not voice:
        raise gr.Error("先に声を選んでください")
    vdir = voice_dir(voice)
    if _alive(read_status(vdir).get("pid")):
        raise gr.Error("すでに学習中です")
    if not any(p.suffix.lower() in AUDIO_EXT for p in (vdir / "raw").iterdir()):
        raise gr.Error("録音ファイルを追加してください")
    cmd = [str(PY), "-m", "studio.train", voice, "--size", size,
           "--tts-epochs", str(int(tts_epochs)), "--vc-steps", str(int(vc_steps))]
    if not do_tts:
        cmd += ["--skip", "tts"]
    if not do_vc:
        cmd += ["--skip", "vc"]
    engine.unload()  # 読み上げ用モデルが GPU を占有していると学習が極端に遅くなる
    # UI を閉じても学習が続くよう、独立したプロセスとして起動
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
    with open(vdir / "train.log", "a", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, cwd=ROOT, env=child_env(), stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    write_status(vdir, stage="start", message="学習を開始しました", pid=p.pid, error=None, progress=0)
    return engine.status_text(voice)


def stop_training(voice):
    vdir = voice_dir(voice)
    pid = read_status(vdir).get("pid")
    if _alive(pid):
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        write_status(vdir, stage="stopped", message="学習を止めました（保存済みのモデルは使えます）", pid=None)
    return engine.status_text(voice)


def refresh(voice):
    if not voice:
        return "", ""
    log = voice_dir(voice) / "train.log"
    tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]) if log.exists() else ""
    return engine.status_text(voice), tail


def ckpt_choices(voice):
    names = engine.tts_checkpoints(voice) if voice else []
    return gr.update(choices=names + ["（未学習のクローン）"], value=names[0] if names else "（未学習のクローン）")


def do_speak(voice, text, ckpt, instruct):
    if not voice or not text.strip():
        raise gr.Error("声と文章を入れてください")
    if any(_alive(read_status(voice_dir(v)).get("pid")) for v in list_voices()):
        gr.Warning("学習中です。読み上げも学習も遅くなります")
    try:
        out, info = engine.speak(voice, text, None if ckpt.startswith("（") else ckpt, instruct)
    except Exception as e:
        raise gr.Error(str(e))
    return str(out), info


def do_convert(voice, audio, steps, auto_pitch, semitone):
    if not voice or not audio:
        raise gr.Error("声を選び、録音してください")
    try:
        out, info = engine.convert(voice, audio, int(steps), auto_pitch, int(semitone))
    except Exception as e:
        raise gr.Error(str(e))
    return str(out), info


with gr.Blocks(title="Voice Studio") as demo:
    gr.Markdown("# Voice Studio\nなりたい声を学習して、**文章から**、または**自分の声の抑揚で**しゃべらせます。")
    with gr.Row():
        voice = gr.Dropdown(list_voices(), value=(list_voices() or [None])[0], label="使う声", scale=2)
        new_name = gr.Textbox(label="新しい声の名前", placeholder="例: narrator", scale=2)
        new_btn = gr.Button("作成", scale=1)

    with gr.Tab("① 声を学習"):
        gr.Markdown("なりたい声の録音を入れてください。**合計30分〜1時間**あると良いです（最低10分）。"
                    "雑音・BGMなし、1人だけの声。長い録音のままでOK（自動で区切って書き起こします）。")
        files = gr.File(file_count="multiple", file_types=list(AUDIO_EXT), label="録音ファイル（wav / mp3 / m4a など）")
        add_btn = gr.Button("録音を追加")
        raw_info = gr.Textbox(label="追加済み", lines=2, interactive=False)
        with gr.Accordion("学習の設定", open=False):
            size = gr.Radio(["1.7B", "0.6B"], value="1.7B", label="読み上げモデルの大きさ（メモリ不足なら 0.6B）")
            tts_epochs = gr.Slider(1, 50, value=10, step=1, label="文章読み上げの学習回数（周）。多いほど長時間・クセを覚える")
            vc_steps = gr.Slider(500, 20000, value=3000, step=500, label="声変換の学習ステップ数")
            do_tts = gr.Checkbox(True, label="文章読み上げ（②）を学習する")
            do_vc = gr.Checkbox(True, label="声変換（③）を学習する")
        with gr.Row():
            start_btn = gr.Button("学習スタート", variant="primary")
            stop_btn = gr.Button("止める", variant="stop")
        status = gr.Textbox(label="状態", lines=4, interactive=False)
        log = gr.Textbox(label="ログ（最後の15行）", lines=8, interactive=False)

    with gr.Tab("② 文章から読み上げ"):
        text = gr.Textbox(label="読ませる文章", lines=4)
        with gr.Row():
            ckpt = gr.Dropdown(label="モデル（新しい順）", scale=2)
            reload_ckpt = gr.Button("一覧を更新", scale=1)
        instruct = gr.Textbox(label="話し方の指示（任意。学習済みモデルでは効かないことがあります）", placeholder="例: 嬉しそうに")
        speak_btn = gr.Button("読み上げる", variant="primary")
        speak_out = gr.Audio(label="結果", type="filepath")
        speak_info = gr.Markdown()

    with gr.Tab("③ 自分の声で抑揚指定"):
        gr.Markdown("自分で、つけたい抑揚・間・感情のとおりに読んで録音してください。声だけが学習した声に変わります。")
        src = gr.Audio(sources=["microphone", "upload"], type="filepath", label="自分の録音")
        with gr.Row():
            auto_pitch = gr.Checkbox(True, label="声の高さを学習した声に合わせる（男女差があるときはオン）")
            semitone = gr.Slider(-12, 12, value=0, step=1, label="高さの微調整（半音）")
            steps = gr.Slider(10, 100, value=30, step=5, label="品質（多いほど高品質・遅い）")
        conv_btn = gr.Button("変換する", variant="primary")
        conv_out = gr.Audio(label="結果", type="filepath")
        conv_info = gr.Markdown()

    new_btn.click(create_voice, new_name, [voice, new_name])
    voice.change(raw_summary, voice, raw_info).then(refresh, voice, [status, log]).then(ckpt_choices, voice, ckpt)
    add_btn.click(add_files, [voice, files], [raw_info, files])
    start_btn.click(start_training, [voice, size, tts_epochs, vc_steps, do_tts, do_vc], status)
    stop_btn.click(stop_training, voice, status)
    gr.Timer(5).tick(refresh, voice, [status, log])
    reload_ckpt.click(ckpt_choices, voice, ckpt)
    speak_btn.click(do_speak, [voice, text, ckpt, instruct], [speak_out, speak_info])
    conv_btn.click(do_convert, [voice, src, steps, auto_pitch, semitone], [conv_out, conv_info])
    demo.load(raw_summary, voice, raw_info).then(refresh, voice, [status, log]).then(ckpt_choices, voice, ckpt)


if __name__ == "__main__":
    demo.queue().launch(inbrowser="--no-browser" not in sys.argv, allowed_paths=[str(ROOT / "outputs")])
