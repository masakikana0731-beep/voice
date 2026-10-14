"""Voice Studio（ブラウザで使う UI）。

  python -m studio.app           →  https://<この PC の IP>:7860（スマホからも可。パスワード付き）
  python -m studio.app --local   →  http://127.0.0.1:7860（この PC だけ）

  ⓪ 動画から声を抽出 動画・音声から BGM・効果音・残響・雑音を消して声だけにする
  ① 声を学習       なりたい声の録音を入れて、長時間学習させる
  ② 文章から読み上げ 学習した声で文章を読む
  ③ 自分の声で抑揚指定 自分で読んだ録音を、学習した声に変換（抑揚・間はそのまま）
"""

import ctypes
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import gradio as gr

from studio import engine, post
from studio.common import ffmpeg, AUDIO_EXT, PY, ROOT, SEP_PY, VIDEO_EXT, child_env, list_voices, read_status, safe_name, voice_dir, write_status


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


FEATURES = ROOT / ".studio" / "features.json"


def test_mode() -> bool:
    try:
        return bool(json.loads(FEATURES.read_text(encoding="utf-8")).get("test_mode"))
    except (OSError, ValueError):
        return False


def set_test_mode(on):
    FEATURES.parent.mkdir(exist_ok=True)
    FEATURES.write_text(json.dumps({"test_mode": bool(on)}), encoding="utf-8")
    return [gr.update(visible=bool(on))] * 4


def disk_usage():
    sep = sum(f.stat().st_size for f in SEP_DIR.rglob("*") if f.is_file()) if SEP_DIR.exists() else 0
    allout = sum(f.stat().st_size for f in OUTPUTS.rglob("*") if f.is_file())
    return f"outputs 全体: {allout / 1e9:.1f}GB（うち抽出結果: {sep / 1e9:.1f}GB）"


def cleanup_separated(days):
    """（テスト機能）古い抽出結果の大きいファイル（全員分の声 *_voice.wav）を消す。話者別のファイルは残す。"""
    limit = time.time() - float(days) * 86400
    freed = 0
    for f in SEP_DIR.glob("*/*_voice.wav"):
        if f.stat().st_mtime < limit:
            freed += f.stat().st_size
            f.unlink()
    gr.Info(f"{freed / 1e9:.2f}GB 空けました")
    return disk_usage()


def start_realtime():
    """（テスト機能）リアルタイム声変換の画面を PC 上に開く。"""
    subprocess.Popen(["cmd", "/c", "start", "", str(ROOT / "realtime-vc.bat")], cwd=ROOT)
    gr.Info("PC の画面にリアルタイム声変換が開きます（初回はモデルの準備に数分かかります）")


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


def start_training(voice, size, tts_epochs, vc_steps, do_tts, do_vc, use_purity=False, purity=0.55, eval_sim=False):
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
    if test_mode() and use_purity:
        cmd += ["--purity", str(purity)]
    if test_mode() and eval_sim:
        cmd += ["--eval-sim"]
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


MAX_SPK = 5
SAME_PERSON = 0.6  # 登録済みの声との近さがこれ以上なら同じ人とみなす


OUTPUTS = (ROOT / "outputs").resolve()
SEP_DIR = ROOT / "outputs" / "separated"
DETACH = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW


def _sep_status(job):
    try:
        return json.loads((SEP_DIR / job / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # status.json のない古い結果は、出力があれば完了扱い
        return {"state": "done", "message": "完了"} if list((SEP_DIR / job).glob("*_voice.wav")) else {}


def sep_jobs():
    """過去の抽出（新しい順）。表示名は「日時  入力（状態）」。"""
    items = []
    for d in sorted((p for p in SEP_DIR.glob("*") if p.is_dir()), reverse=True)[:30]:
        inp = d / "inputs.txt"
        first = inp.read_text(encoding="utf-8").splitlines()[0] if inp.exists() else ""
        name = Path(first).name if first and not first.startswith("http") else first.split("//")[-1][:40]
        state = {"running": "処理中", "error": "失敗", "done": "完了"}.get(_sep_status(d.name).get("state"), "?")
        when = f"{d.name[4:6]}/{d.name[6:8]} {d.name[9:11]}:{d.name[11:13]}"
        items.append((f"{when}  {name}（{state}）", d.name))
    return items


def start_separate(voice, files, urls, dereverb, denoise, speakers):
    inputs = list(files or []) + [u.strip() for u in (urls or "").splitlines() if u.strip()]
    if not inputs:
        raise gr.Error("動画・音声ファイルか、YouTube の URL を入れてください")
    engine.unload()
    job = time.strftime("%Y%m%d-%H%M%S")
    out_dir = SEP_DIR / job
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = {"分けない": "off", "自動": "auto"}.get(speakers, speakers.rstrip("人"))
    cmd = [str(SEP_PY), "-m", "studio.separate", *inputs, "-o", str(out_dir), "--speakers", mode]
    if not dereverb:
        cmd.append("--no-dereverb")
    if not denoise:
        cmd.append("--no-denoise")
    # 「使う声」に登録済みの録音があれば、それに近い人を自動で選べるよう比べる
    if voice and mode != "off":
        refs = sorted((voice_dir(voice) / "raw").glob("*"), key=lambda f: f.stat().st_mtime)[-3:]
        refs = [str(f) for f in refs if f.suffix.lower() in AUDIO_EXT]
        if refs:
            cmd += ["--match", *refs]
    # 画面を閉じても・スマホがスリープしても処理が続くよう、独立したプロセスで動かす
    with open(out_dir / "sep.log", "w", encoding="utf-8") as log:
        subprocess.Popen(cmd, cwd=ROOT, env=child_env(), stdout=log, stderr=subprocess.STDOUT, creationflags=DETACH)
    gr.Info("抽出を始めました。画面を閉じても処理は続きます")
    return gr.update(choices=sep_jobs(), value=job)


def render_job(job):
    """抽出結果を画面に出す。戻り値: [メッセージ, ファイル一覧, パス, 表示中の job] + 話者枠。"""
    hidden = [gr.update(value=None, visible=False), gr.update(value=False, visible=False)] * MAX_SPK
    if not job:
        return ["", None, [], None] + hidden
    st = _sep_status(job)
    if not st:
        return ["状態が分かりません（古い形式の処理か、途中で止まった可能性があります）", None, [], None] + hidden
    if st.get("state") != "done":
        msg = {"running": "⏳ 処理中: ", "error": "⚠ "}.get(st.get("state"), "") + st.get("message", "")
        return [msg + f"（{st.get('updated', '')}）", None, [], None] + hidden
    out_dir = SEP_DIR / job
    outs = sorted(str(f) for f in out_dir.glob("*_voice.wav"))
    spk_json = out_dir / "speakers" / "speakers.json"
    spk = json.loads(spk_json.read_text(encoding="utf-8")) if spk_json.exists() else []
    # 話者ごとの枠（最大 5）。話者分けしないときは抽出結果をそのまま並べる
    slots = (spk or [{"file": f, "seconds": None} for f in outs])[:MAX_SPK]
    updates = []
    for i in range(MAX_SPK):
        if i < len(slots):
            sec, sim = slots[i]["seconds"], slots[i].get("similarity")
            label = f"話者{i + 1}（{sec:.0f}秒）" if sec is not None else Path(slots[i]["file"]).name
            if sim is not None:
                label += f" 登録済みの声との近さ {sim:.2f}" + (" ★" if sim >= SAME_PERSON else "")
            # 登録済みの声と比べたなら、近い人（同一人物とみなせる人）にだけチェック。比べていなければ 1 人目
            checked = sim >= SAME_PERSON if sim is not None else i == 0
            updates += [gr.update(value=slots[i]["file"], label=label, visible=True),
                        gr.update(value=checked, label=f"{label} を残す", visible=True)]
        else:
            updates += [gr.update(value=None, visible=False), gr.update(value=False, visible=False)]
    msg = "✅ 抽出完了。" + (f"{len(spk)} 人に分けました。残したい人にチェックしてください。" if spk else "")
    if any(s.get("similarity", 0) >= SAME_PERSON for s in spk):
        msg += "（★ = 登録済みの声と同じ人とみられるので、自動でチェックしました）"
    return [msg, outs + [s["file"] for s in spk], [s["file"] for s in slots], job] + updates


def tick_job(job, shown):
    """5 秒ごと: 処理中は進み具合だけ更新。完了した瞬間に一度だけ結果を出す（チェックを消さないため）。"""
    if not job or job == shown:
        return [gr.skip()] * (4 + 2 * MAX_SPK)
    st = _sep_status(job)
    if st.get("state") == "done":
        return render_job(job)
    return [render_job(job)[0]] + [gr.skip()] * (3 + 2 * MAX_SPK)


def add_separated(voice, paths, *checks):
    if not voice:
        raise gr.Error("先に上の「使う声」を選ぶか作ってください")
    chosen = [f for f, c in zip(paths or [], checks) if c]
    if not chosen:
        raise gr.Error("残す人にチェックを入れてください")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for f in chosen:
        shutil.copy(f, voice_dir(voice) / "raw" / f"{stamp}_{Path(f).name}")
    gr.Info(f"「{voice}」の学習データに {len(chosen)} 本追加しました")
    return raw_summary(voice)


def to_mp3(wav) -> str:
    """mp3 を作り、保存用ボタンの HTML を返す。

    普通のリンクだとスマホは再生アプリ（YouTube Music など）で開いてしまうので、
    「保存してください」という指定（Content-Disposition: attachment）付きで配る /dl を使う。
    """
    from urllib.parse import quote

    mp3 = Path(wav).with_suffix(".mp3")
    subprocess.run([ffmpeg(), "-v", "error", "-y", "-i", str(wav), "-b:a", "160k", str(mp3)], check=True)
    rel = mp3.resolve().relative_to(OUTPUTS).as_posix()
    return (f'<a href="dl?f={quote(rel)}" download="{mp3.name}" '
            'style="display:block;text-align:center;padding:14px;border-radius:10px;'
            'background:#e8590c;color:#fff;font-weight:bold;text-decoration:none;font-size:17px">'
            f'⬇ スマホに保存する（{mp3.name}）</a>')


def finish(raw, preset, reverb, echo, ambience, lips):
    """仕上げ（リップノイズ・響き・環境）をかけて、試聴用・保存用を返す。"""
    if not raw:
        raise gr.Error("先に音声を作ってください")
    fx = post.apply(raw, preset, reverb, echo, ambience, lips)
    return str(fx), to_mp3(fx)


def do_speak(voice, text, ckpt, instruct, chain_vc, *fx):
    if not voice or not text.strip():
        raise gr.Error("声と文章を入れてください")
    if any(_alive(read_status(voice_dir(v)).get("pid")) for v in list_voices()):
        gr.Warning("学習中です。読み上げも学習も遅くなります")
    try:
        out, info = engine.speak(voice, text, None if ckpt.startswith("（") else ckpt, instruct)
        if test_mode() and chain_vc:
            # （テスト機能）読み上げた音声に声変換をもう一度かけて、声を近づける
            out, info2 = engine.convert(voice, str(out))
            info += f" → 声変換で仕上げ（{info2}）"
    except Exception as e:
        raise gr.Error(str(e))
    return (*finish(str(out), *fx), info, str(out))


def do_convert(voice, audio, steps, auto_pitch, semitone, *fx):
    if not voice or not audio:
        raise gr.Error("声を選び、録音してください")
    try:
        out, info = engine.convert(voice, audio, int(steps), auto_pitch, int(semitone))
    except Exception as e:
        raise gr.Error(str(e))
    return (*finish(str(out), *fx), info, str(out))


def finish_controls():
    """仕上げの設定欄（②③共通）。"""
    with gr.Accordion("仕上げ（リップノイズ・響き・環境）", open=False):
        preset = gr.Dropdown(post.PRESETS, value="なし", label="環境（プリセット）")
        with gr.Row():
            lips = gr.Slider(0, 100, value=30, step=5, label="リップノイズ（話し始めの口の音）")
            reverb = gr.Slider(0, 100, value=0, step=5, label="リバーブ（響き）")
        with gr.Row():
            echo = gr.Slider(0, 100, value=0, step=5, label="エコー（やまびこ）")
            ambience = gr.Slider(0, 100, value=0, step=5, label="環境音（空気感）")
        redo = gr.Button("仕上げだけやり直す（変換し直さずに、この設定をかけ直す）")
    return [preset, reverb, echo, ambience, lips], redo


CSS = ".tab-wrapper button { padding: 0 8px !important; }"

with gr.Blocks(title="Voice Studio") as demo:
    gr.Markdown("# Voice Studio\nなりたい声を学習して、**文章から**、または**自分の声の抑揚で**しゃべらせます。")
    with gr.Row():
        voice = gr.Dropdown(list_voices(), value=(list_voices() or [None])[0], label="使う声", scale=2)
        new_name = gr.Textbox(label="新しい声の名前", placeholder="例: narrator", scale=2)
        new_btn = gr.Button("作成", scale=1)
    test_sw = gr.Checkbox(test_mode(), label="🧪 テストモード（試験中の機能を表示する）")

    with gr.Tab("⓪抽出"):
        gr.Markdown("### ⓪ 動画から声を抽出\n動画・音声ファイル、または **YouTube の URL** から、BGM・効果音・残響・雑音を消して声だけを取り出し、"
                    "**話している人ごとに分けます**。聞き比べて、残したい人だけを学習データに追加できます。"
                    "上の「使う声」に録音を登録済みなら、**同じ人に自動でチェック**が付きます。")
        with gr.Row():
            sep_files = gr.File(file_count="multiple", file_types=list(VIDEO_EXT + AUDIO_EXT), label="動画・音声ファイル")
            sep_urls = gr.Textbox(label="YouTube などの URL（1行に1つ。複数可）", lines=4,
                                  placeholder="https://www.youtube.com/watch?v=...")
        with gr.Row():
            speakers = gr.Dropdown(["自動", "1人", "2人", "3人", "4人", "5人", "分けない"], value="自動",
                                   label="話している人数（自動でずれたら指定）")
            dereverb = gr.Checkbox(True, label="残響（エコー）も消す")
            denoise = gr.Checkbox(True, label="雑音も消す")
        sep_btn = gr.Button("声を抽出する", variant="primary")
        sep_job = gr.Dropdown(sep_jobs(), label="抽出結果（新しい順。前の結果もここから開けます）")
        sep_info = gr.Markdown()
        spk_audio, spk_check = [], []
        for i in range(MAX_SPK):
            with gr.Row():
                spk_audio.append(gr.Audio(type="filepath", visible=False, scale=4))
                spk_check.append(gr.Checkbox(False, visible=False, scale=1))
        sep_paths = gr.State([])
        sep_shown = gr.State(None)
        sep_add = gr.Button("チェックした人を「使う声」の学習データに追加", variant="primary")
        sep_out = gr.File(label="抽出したファイル（ダウンロード可）", file_count="multiple")
        with gr.Group(visible=test_mode()) as t0:
            gr.Markdown("**🧪 ディスクの整理**（抽出結果の大きいファイルを消す。話者別のファイルは残す）")
            with gr.Row():
                clean_days = gr.Slider(0, 30, value=7, step=1, label="何日より前のものを消すか")
                clean_btn = gr.Button("整理する")
            disk_info = gr.Markdown(disk_usage())

    with gr.Tab("①学習"):
        gr.Markdown("### ① 声を学習\nなりたい声の録音を入れてください。**合計30分〜1時間**あると良いです（最低10分）。"
                    "雑音・BGMなし、1人だけの声。長い録音のままでOK（自動で区切って書き起こします）。")
        files = gr.File(file_count="multiple", file_types=list(AUDIO_EXT), label="録音ファイル（wav / mp3 / m4a など）")
        add_btn = gr.Button("録音を追加")
        raw_info = gr.Textbox(label="追加済み", lines=2, interactive=False)
        with gr.Accordion("学習の設定", open=False):
            size = gr.Radio(["1.7B", "0.6B"], value="1.7B", label="読み上げモデルの大きさ（メモリ不足なら 0.6B）")
            tts_epochs = gr.Slider(1, 20, value=3, step=1, label="文章読み上げの学習回数（周）。多すぎると壊れるので、各周を自動で採点して一番良いものを使います")
            vc_steps = gr.Slider(500, 20000, value=3000, step=500, label="声変換の学習ステップ数")
            do_tts = gr.Checkbox(True, label="文章読み上げ（②）を学習する")
            do_vc = gr.Checkbox(True, label="声変換（③）を学習する")
        with gr.Group(visible=test_mode()) as t1:
            gr.Markdown("**🧪 テスト機能**")
            with gr.Row():
                use_purity = gr.Checkbox(False, label="学習データの混ざり（相手の声など）を除外")
                purity = gr.Slider(0.3, 0.8, value=0.55, step=0.05, label="この近さ未満のクリップを除外")
            eval_sim = gr.Checkbox(False, label="読み上げモデルの採点に「声の近さ」も使う")
        with gr.Row():
            start_btn = gr.Button("学習スタート", variant="primary")
            stop_btn = gr.Button("止める", variant="stop")
        status = gr.Textbox(label="状態", lines=4, interactive=False)
        log = gr.Textbox(label="ログ（最後の15行）", lines=8, interactive=False)

    with gr.Tab("②読み上げ"):
        gr.Markdown("### ② 文章から読み上げ")
        text = gr.Textbox(label="読ませる文章", lines=4)
        with gr.Row():
            ckpt = gr.Dropdown(label="モデル（一番上が採点で一番良かったもの）", scale=2)
            reload_ckpt = gr.Button("一覧を更新", scale=1)
        instruct = gr.Textbox(label="話し方の指示（任意。学習済みモデルでは効かないことがあります）", placeholder="例: 嬉しそうに")
        with gr.Group(visible=test_mode()) as t2:
            chain_vc = gr.Checkbox(False, label="🧪 声変換で仕上げる（読み上げ後に③をかけて声を近づける。時間は2倍ほど）")
        speak_fx, speak_redo = finish_controls()
        speak_btn = gr.Button("読み上げる", variant="primary")
        speak_out = gr.Audio(label="結果", type="filepath")
        speak_file = gr.HTML()
        speak_info = gr.Markdown()
        speak_raw = gr.State(None)

    with gr.Tab("③声変換"):
        gr.Markdown("### ③ 自分の声で抑揚指定\n自分で、つけたい抑揚・間・感情のとおりに読んで録音してください。声だけが学習した声に変わります。")
        src = gr.Audio(sources=["microphone", "upload"], type="filepath", label="自分の録音")
        with gr.Row():
            auto_pitch = gr.Checkbox(True, label="声の高さを学習した声に合わせる（男女差があるときはオン）")
            semitone = gr.Slider(-12, 12, value=0, step=1, label="高さの微調整（半音）")
            steps = gr.Slider(10, 100, value=50, step=5, label="品質（多いほど高品質・遅い）")
        with gr.Group(visible=test_mode()) as t3:
            gr.Markdown("**🧪 リアルタイム声変換**（PC のマイクに話すと、その場で声が変わる。PC の画面で操作。"
                        "Reference Audio に `voices/<名前>/ref_vc.wav` を選んでください）")
            rt_btn = gr.Button("PC でリアルタイム声変換を開く")
        conv_fx, conv_redo = finish_controls()
        conv_btn = gr.Button("変換する", variant="primary")
        conv_out = gr.Audio(label="結果", type="filepath")
        conv_file = gr.HTML()
        conv_info = gr.Markdown()
        conv_raw = gr.State(None)

    spk_outputs = [x for pair in zip(spk_audio, spk_check) for x in pair]
    job_outputs = [sep_info, sep_out, sep_paths, sep_shown] + spk_outputs
    sep_btn.click(start_separate, [voice, sep_files, sep_urls, dereverb, denoise, speakers], sep_job)
    sep_job.change(render_job, sep_job, job_outputs)
    gr.Timer(5).tick(tick_job, [sep_job, sep_shown], job_outputs)
    sep_add.click(add_separated, [voice, sep_paths] + spk_check, raw_info)
    new_btn.click(create_voice, new_name, [voice, new_name])
    voice.change(raw_summary, voice, raw_info).then(refresh, voice, [status, log]).then(ckpt_choices, voice, ckpt)
    add_btn.click(add_files, [voice, files], [raw_info, files])
    start_btn.click(start_training, [voice, size, tts_epochs, vc_steps, do_tts, do_vc, use_purity, purity, eval_sim],
                    status)
    test_sw.change(set_test_mode, test_sw, [t0, t1, t2, t3])
    clean_btn.click(cleanup_separated, clean_days, disk_info)
    rt_btn.click(start_realtime)
    stop_btn.click(stop_training, voice, status)
    gr.Timer(5).tick(refresh, voice, [status, log])
    reload_ckpt.click(ckpt_choices, voice, ckpt)
    speak_btn.click(do_speak, [voice, text, ckpt, instruct, chain_vc] + speak_fx, [speak_out, speak_file, speak_info, speak_raw])
    speak_redo.click(finish, [speak_raw] + speak_fx, [speak_out, speak_file])
    conv_btn.click(do_convert, [voice, src, steps, auto_pitch, semitone] + conv_fx,
                   [conv_out, conv_file, conv_info, conv_raw])
    conv_redo.click(finish, [conv_raw] + conv_fx, [conv_out, conv_file])
    demo.load(lambda: gr.update(choices=sep_jobs(), value=(sep_jobs() or [(None, None)])[0][1]), None, sep_job)
    demo.load(raw_summary, voice, raw_info).then(refresh, voice, [status, log]).then(ckpt_choices, voice, ckpt)


if __name__ == "__main__":
    opts = dict(allowed_paths=[str(ROOT / "outputs")], css=CSS)
    # この PC の中（127.0.0.1）だけで待ち受ける。スマホからは Tailscale の中継（tailscale serve）経由で開く。
    # Tailscale に入れるのは自分のアカウントの端末だけなので、パスワードは付けない
    url = "http://127.0.0.1:7860"
    if "--local" not in sys.argv:
        from studio import remote

        url = remote.show() or url
    if "--no-browser" not in sys.argv:
        import threading
        import webbrowser

        threading.Timer(4, webbrowser.open, [url]).start()
    demo.queue().launch(server_name="127.0.0.1", prevent_thread_lock=True, **opts)

    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    def download(f: str):
        """outputs/ の中のファイルを「保存」として配る（外のファイルは出さない）。"""
        path = (OUTPUTS / f).resolve()
        if not path.is_file() or OUTPUTS not in path.parents:
            raise HTTPException(404)
        return FileResponse(path, media_type="application/octet-stream", filename=path.name)

    demo.app.add_api_route("/dl", download, methods=["GET"])
    demo.block_thread()
