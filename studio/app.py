"""Voice Studio（ブラウザで使う UI）。

  python -m studio.app           →  Tailscale の URL（スマホからも可）
  python -m studio.app --local   →  http://127.0.0.1:7860（この PC だけ）

ふだん: 話者（声のプリセット）を選ぶ → 🎙 声変換 か 📝 文章読み上げ。
新しい話者を作る・育てるときだけ、STEP1 素材 → STEP2 声を選ぶ → STEP3 保存 → STEP4 学習 が出る。
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
from studio import speakers as spk
from studio.common import AUDIO_EXT, PY, ROOT, SEP_PY, VIDEO_EXT, VOICES, child_env, ffmpeg, read_status, voice_dir, write_status

OUTPUTS = (ROOT / "outputs").resolve()
SEP_DIR = ROOT / "outputs" / "separated"
DETACH = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
MAX_SPK = 5
SAME_PERSON = 0.6  # 登録済みの声との近さがこれ以上なら同じ人とみなす


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


# ---------------------------------------------------------------- テストモード・全体設定

FEATURES = ROOT / ".studio" / "features.json"
N_TEST_GROUPS = 4


def test_mode() -> bool:
    try:
        return bool(json.loads(FEATURES.read_text(encoding="utf-8")).get("test_mode"))
    except (OSError, ValueError):
        return False


def set_test_mode(on):
    FEATURES.parent.mkdir(exist_ok=True)
    FEATURES.write_text(json.dumps({"test_mode": bool(on)}), encoding="utf-8")
    return [gr.update(visible=bool(on))] * N_TEST_GROUPS


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


# ---------------------------------------------------------------- 話者の一覧・選択

def gallery_items():
    items = []
    for sid in spk.list_ids():
        i = spk.info(sid)
        mark = "⏳ 学習中" if i["training"] else ("✓ 使える" if i["vc"] else "未学習")
        items.append((i["icon"], f"{i['name']}（{mark}）"))
    items.append((str(spk.plus_icon()), "＋ 新しい話者を追加"))
    return items


def speaker_title(sid):
    if not sid:
        return "### 上から話者を選んでください"
    i = spk.info(sid)
    vc = "✓" if i["vc"] else "未学習（簡易モードで変換）"
    tts = "✓" if i["tts"] else "未学習（簡易モードで読み上げ）"
    mins = f" / 素材 {i['minutes']}分" if i["minutes"] else ""
    return f"### 🎤 {i['name']}\n声変換: {vc}　文章読み上げ: {tts}{mins}"


def ckpt_choices(sid):
    names = engine.tts_checkpoints(sid) if sid else []
    return gr.update(choices=names + ["（未学習のクローン）"], value=names[0] if names else "（未学習のクローン）")


def show_use(sid):
    """話者を選んだ（ふだんの画面）。"""
    i = spk.info(sid) if sid else None
    return [sid, None, gr.update(visible=bool(sid)), gr.update(visible=False), speaker_title(sid),
            ckpt_choices(sid), gr.update(value=i["name"] if i else ""), gr.update(value=None)]


def show_add(target):
    """新しい話者を追加（target=None）／この話者を育てる（target=話者）の画面。"""
    if target:
        title = f"## 🌱 {spk.meta(target)['name']} を育てる（素材を追加・再学習）"
        save_label = f"STEP3: チェックした声を「{spk.meta(target)['name']}」の素材に追加"
    else:
        title = "## ＋ 新しい話者を追加"
        save_label = "STEP3: この声で話者を作る"
    return [target, gr.update(visible=False), gr.update(visible=True), title,
            gr.update(visible=target is None), gr.update(value=save_label), raw_summary(target)]


def on_gallery(evt: gr.SelectData):
    """話者のカードを押した。出力: [cur, target, use_col, add_col, add_title, new_fields, save_btn, raw_info,
    title, ckpt, set_name]"""
    ids = spk.list_ids()
    if evt.index >= len(ids):  # 「＋ 新しい話者を追加」
        a = show_add(None)
        return [gr.skip()] + a + [gr.skip()] * 3
    sid = ids[evt.index]
    u = show_use(sid)
    return [sid, None, u[2], u[3], gr.skip(), gr.skip(), gr.skip(), raw_summary(sid), u[4], u[5], u[6]]


# ---------------------------------------------------------------- 話者の設定

def save_settings(sid, name, icon_file):
    if not sid:
        raise gr.Error("話者を選んでください")
    if name.strip():
        spk.save_meta(sid, name=name.strip())
    if icon_file:
        spk.set_icon(sid, icon_file)
    gr.Info("保存しました")
    return gr.update(value=gallery_items()), speaker_title(sid), None


def delete_speaker(sid, sure):
    if not sid:
        raise gr.Error("話者を選んでください")
    if not sure:
        raise gr.Error("「本当に削除する」にチェックを入れてください")
    name = spk.meta(sid)["name"]
    spk.delete(sid)
    gr.Info(f"「{name}」を削除しました（voices/_trash に残っているので戻せます）")
    ids = spk.list_ids()
    return [gr.update(value=gallery_items())] + show_use(ids[0] if ids else None) + [False]


# ---------------------------------------------------------------- STEP1〜2: 抽出

def raw_summary(sid):
    if not sid:
        return "（まだ素材はありません）"
    files = [p for p in (voice_dir(sid) / "raw").iterdir() if p.suffix.lower() in AUDIO_EXT]
    mb = sum(p.stat().st_size for p in files) / 1e6
    mins = read_status(VOICES / sid).get("minutes")
    return f"いまの素材: {len(files)} ファイル（{mb:.0f}MB）" + (f" / 前回の学習で使った長さ {mins}分" if mins else "")


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


def start_separate(target, files, urls, dereverb, denoise, speakers):
    inputs = list(files or []) + [u.strip() for u in (urls or "").splitlines() if u.strip()]
    if not inputs:
        raise gr.Error("YouTube の URL か、動画・音声ファイルを入れてください")
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
    # 育てる話者が決まっていれば、その声に近い人を自動で選べるよう比べる
    if target and mode != "off":
        refs = sorted((voice_dir(target) / "raw").glob("*"), key=lambda f: f.stat().st_mtime)[-3:]
        refs = [str(f) for f in refs if f.suffix.lower() in AUDIO_EXT]
        if refs:
            cmd += ["--match", *refs]
    # 画面を閉じても・スマホがスリープしても処理が続くよう、独立したプロセスで動かす
    with open(out_dir / "sep.log", "w", encoding="utf-8") as log:
        subprocess.Popen(cmd, cwd=ROOT, env=child_env(), stdout=log, stderr=subprocess.STDOUT, creationflags=DETACH)
    gr.Info("抽出を始めました。画面を閉じても処理は続きます")
    return gr.update(choices=sep_jobs(), value=job)


def render_job(job):
    """抽出結果を画面に出す。戻り値: [メッセージ, パス, 表示中の job] + 話者枠。"""
    hidden = [gr.update(value=None, visible=False), gr.update(value=False, visible=False)] * MAX_SPK
    if not job:
        return ["", [], None] + hidden
    st = _sep_status(job)
    if not st:
        return ["状態が分かりません（古い形式の処理か、途中で止まった可能性があります）", [], None] + hidden
    if st.get("state") != "done":
        msg = {"running": "⏳ 処理中: ", "error": "⚠ "}.get(st.get("state"), "") + st.get("message", "")
        return [msg + f"（{st.get('updated', '')}）", [], None] + hidden
    out_dir = SEP_DIR / job
    outs = sorted(str(f) for f in out_dir.glob("*_voice.wav"))
    spk_json = out_dir / "speakers" / "speakers.json"
    found = json.loads(spk_json.read_text(encoding="utf-8")) if spk_json.exists() else []
    # 話者ごとの枠（最大 5）。話者分けしないときは抽出結果をそのまま並べる
    slots = (found or [{"file": f, "seconds": None} for f in outs])[:MAX_SPK]
    updates = []
    for i in range(MAX_SPK):
        if i < len(slots):
            sec, sim = slots[i]["seconds"], slots[i].get("similarity")
            label = f"声{i + 1}（{sec:.0f}秒）" if sec is not None else Path(slots[i]["file"]).name
            if sim is not None:
                label += f" 似ている度 {sim:.2f}" + (" ★" if sim >= SAME_PERSON else "")
            # 育てる話者と比べたなら、同じ人とみなせる声にだけチェック。比べていなければ 1 人目
            checked = sim >= SAME_PERSON if sim is not None else i == 0
            updates += [gr.update(value=slots[i]["file"], label=label, visible=True),
                        gr.update(value=checked, label="この声を使う", visible=True)]
        else:
            updates += [gr.update(value=None, visible=False), gr.update(value=False, visible=False)]
    msg = "✅ 抽出完了。" + (f"{len(found)} 人に分けました。**使う声にチェック**して STEP3 へ。" if found else "")
    if any(s.get("similarity", 0) >= SAME_PERSON for s in found):
        msg += "（★ = 育てている話者と同じ人とみられるので、自動でチェックしました）"
    return [msg, [s["file"] for s in slots], job] + updates


def tick_job(job, shown):
    """5 秒ごと: 処理中は進み具合だけ更新。完了した瞬間に一度だけ結果を出す（チェックを消さないため）。"""
    if not job or job == shown:
        return [gr.skip()] * (3 + 2 * MAX_SPK)
    if _sep_status(job).get("state") == "done":
        return render_job(job)
    return [render_job(job)[0]] + [gr.skip()] * (2 + 2 * MAX_SPK)


# ---------------------------------------------------------------- STEP3: 話者として保存

def save_material(target, name, icon_file, paths, uploads, *checks):
    """チェックした声（と、そのまま使う録音）を話者の素材に入れる。新しい話者ならここで作る。"""
    chosen = [f for f, c in zip(paths or [], checks) if c] + list(uploads or [])
    if not chosen:
        raise gr.Error("STEP2 で使う声にチェックを入れるか、録音ファイルを入れてください")
    if not target:
        if not (name or "").strip():
            raise gr.Error("話者の名前を入れてください")
        target = spk.create(name, icon_file)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for f in chosen:
        shutil.copy(f, voice_dir(target) / "raw" / f"{stamp}_{Path(f).name}")
    gr.Info(f"「{spk.meta(target)['name']}」の素材に {len(chosen)} 本入れました。STEP4 で学習できます")
    add = show_add(target)
    return [target, target, gr.update(value=gallery_items()), add[3], add[4], add[5], add[6], None, speaker_title(target)]


# ---------------------------------------------------------------- STEP4: 学習

def start_training(sid, size, tts_epochs, vc_steps, do_tts, do_vc, use_purity=False, purity=0.55, eval_sim=False):
    if not sid:
        raise gr.Error("先に STEP3 で話者を保存してください")
    vdir = voice_dir(sid)
    if _alive(read_status(vdir).get("pid")):
        raise gr.Error("すでに学習中です")
    if not any(p.suffix.lower() in AUDIO_EXT for p in (vdir / "raw").iterdir()):
        raise gr.Error("素材がありません（STEP1〜3）")
    cmd = [str(PY), "-m", "studio.train", sid, "--size", size,
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
    with open(vdir / "train.log", "a", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, cwd=ROOT, env=child_env(), stdout=log, stderr=subprocess.STDOUT, creationflags=DETACH)
    write_status(vdir, stage="start", message="学習を開始しました", pid=p.pid, error=None, progress=0)
    return engine.status_text(sid)


def stop_training(sid):
    if not sid:
        return ""
    vdir = voice_dir(sid)
    pid = read_status(vdir).get("pid")
    if _alive(pid):
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        write_status(vdir, stage="stopped", message="学習を止めました（保存済みのモデルは使えます）", pid=None)
    return engine.status_text(sid)


def refresh(sid):
    if not sid:
        return "", ""
    log = voice_dir(sid) / "train.log"
    tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]) if log.exists() else ""
    return engine.status_text(sid), tail


# ---------------------------------------------------------------- 声変換・読み上げ

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


def _busy_warning():
    if any(_alive(read_status(VOICES / s).get("pid")) for s in spk.list_ids()):
        gr.Warning("学習中です。変換も学習も遅くなります")


def do_convert(sid, audio, steps, auto_pitch, semitone, *fx):
    if not sid:
        raise gr.Error("話者を選んでください")
    if not audio:
        raise gr.Error("録音してください（またはファイルを入れてください）")
    _busy_warning()
    try:
        out, info = engine.convert(sid, audio, int(steps), auto_pitch, int(semitone))
    except Exception as e:
        raise gr.Error(str(e))
    return (*finish(str(out), *fx), info, str(out))


def do_speak(sid, text, ckpt, instruct, chain_vc, *fx):
    if not sid:
        raise gr.Error("話者を選んでください")
    if not text.strip():
        raise gr.Error("読ませる文章を入れてください")
    _busy_warning()
    try:
        out, info = engine.speak(sid, text, None if (ckpt or "（").startswith("（") else ckpt, instruct)
        if test_mode() and chain_vc:
            # （テスト機能）読み上げた音声に声変換をもう一度かけて、声を近づける
            out, info2 = engine.convert(sid, str(out))
            info += f" → 声変換で仕上げ（{info2}）"
    except Exception as e:
        raise gr.Error(str(e))
    return (*finish(str(out), *fx), info, str(out))


def finish_controls():
    """仕上げの設定欄（声変換・読み上げ共通）。"""
    with gr.Accordion("🎛 仕上げ（リップノイズ・響き・環境）", open=False):
        preset = gr.Dropdown(post.PRESETS, value="なし", label="環境（プリセット）")
        with gr.Row():
            lips = gr.Slider(0, 100, value=30, step=5, label="リップノイズ（話し始めの口の音）")
            reverb = gr.Slider(0, 100, value=0, step=5, label="リバーブ（響き）")
        with gr.Row():
            echo = gr.Slider(0, 100, value=0, step=5, label="エコー（やまびこ）")
            ambience = gr.Slider(0, 100, value=0, step=5, label="環境音（空気感）")
        redo = gr.Button("仕上げだけやり直す（変換し直さずに、この設定をかけ直す）")
    return [preset, reverb, echo, ambience, lips], redo


# ---------------------------------------------------------------- 画面

CSS = """
.tab-wrapper button { padding: 0 10px !important; }
.step { border-left: 4px solid #e8590c; padding-left: 10px; margin-top: 14px; }
#speakers .caption-label { font-size: 15px !important; white-space: normal !important; }
"""

_first = (spk.list_ids() or [None])[0]

with gr.Blocks(title="Voice Studio") as demo:
    gr.Markdown("# 🎤 Voice Studio\n**話者を選ぶ** → **🎙 声変換** か **📝 文章読み上げ**。新しい話者は「＋」から作れます。")
    gallery = gr.Gallery(gallery_items(), label="話者（タップして選ぶ）", columns=4, height=230, allow_preview=False,
                         object_fit="cover", elem_id="speakers")
    cur = gr.State(_first)        # 使う話者
    target = gr.State(None)       # 追加・育てる対象（None = 新しい話者）

    # ---- ふだんの画面
    with gr.Column(visible=bool(_first)) as use_col:
        title = gr.Markdown(speaker_title(_first))
        with gr.Tabs():
            with gr.Tab("🎙 声変換"):
                gr.Markdown("つけたい**抑揚・間・感情のとおりに自分で読んで録音** → 声だけがこの話者の声になります。")
                src = gr.Audio(sources=["microphone", "upload"], type="filepath", label="自分の録音（マイク or ファイル）")
                conv_btn = gr.Button("🎙 変換する", variant="primary", size="lg")
                conv_out = gr.Audio(label="結果", type="filepath")
                conv_file = gr.HTML()
                conv_info = gr.Markdown()
                conv_raw = gr.State(None)
                conv_fx, conv_redo = finish_controls()
                with gr.Accordion("⚙ 詳しい設定", open=False):
                    auto_pitch = gr.Checkbox(True, label="声の高さを話者に合わせる（男女差があるときはオン）")
                    semitone = gr.Slider(-12, 12, value=0, step=1, label="高さの微調整（半音）")
                    steps = gr.Slider(10, 100, value=50, step=5, label="品質（多いほど高品質・遅い）")
                with gr.Group(visible=test_mode()) as t3:
                    gr.Markdown("**🧪 リアルタイム声変換**（PC のマイクに話すと、その場で声が変わる。PC の画面で操作。"
                                "Reference Audio に `voices/<話者>/ref_vc.wav` を選んでください）")
                    rt_btn = gr.Button("PC でリアルタイム声変換を開く")
            with gr.Tab("📝 文章読み上げ"):
                text = gr.Textbox(label="読ませる文章", lines=4)
                speak_btn = gr.Button("📝 読み上げる", variant="primary", size="lg")
                speak_out = gr.Audio(label="結果", type="filepath")
                speak_file = gr.HTML()
                speak_info = gr.Markdown()
                speak_raw = gr.State(None)
                speak_fx, speak_redo = finish_controls()
                with gr.Accordion("⚙ 詳しい設定", open=False):
                    with gr.Row():
                        ckpt = gr.Dropdown(label="モデル（一番上が採点で一番良かったもの）", scale=3)
                        reload_ckpt = gr.Button("一覧を更新", scale=1)
                    instruct = gr.Textbox(label="話し方の指示（学習済みモデルでは効かないことがあります）",
                                          placeholder="例: 嬉しそうに")
                with gr.Group(visible=test_mode()) as t2:
                    chain_vc = gr.Checkbox(False, label="🧪 声変換で仕上げる（読み上げ後に声変換をかけて声を近づける。時間は2倍ほど）")
        grow_btn = gr.Button("🌱 この話者を育てる（素材を追加して再学習）")
        with gr.Accordion("⚙ この話者の設定（名前・アイコン・削除）", open=False):
            with gr.Row():
                set_name = gr.Textbox(label="名前", scale=2)
                set_icon = gr.Image(label="アイコン（画像を入れると変更）", type="filepath", height=160, scale=1)
            set_btn = gr.Button("保存")
            with gr.Row():
                del_sure = gr.Checkbox(False, label="本当に削除する")
                del_btn = gr.Button("この話者を削除", variant="stop")

    # ---- 追加・育てる画面（STEP1〜4）
    with gr.Column(visible=not _first) as add_col:
        with gr.Row():
            add_title = gr.Markdown("## ＋ 新しい話者を追加")
            close_btn = gr.Button("✕ 閉じる", scale=0, min_width=90)

        gr.Markdown("### STEP1　素材を入れる", elem_classes="step")
        gr.Markdown("**YouTube の URL** か動画を入れると、BGM・効果音・残響・雑音を消して声だけを取り出し、話している人ごとに分けます。"
                    "合計 **30分〜1時間** あると上手に学習できます（最低10分）。")
        sep_urls = gr.Textbox(label="YouTube などの URL（1行に1つ。複数可）", lines=3, placeholder="https://www.youtube.com/watch?v=...")
        sep_files = gr.File(file_count="multiple", file_types=list(VIDEO_EXT + AUDIO_EXT), label="または動画・音声ファイル")
        with gr.Accordion("⚙ 抽出の設定", open=False):
            speakers = gr.Dropdown(["自動", "1人", "2人", "3人", "4人", "5人", "分けない"], value="自動",
                                   label="話している人数（自動でずれたら指定）")
            dereverb = gr.Checkbox(True, label="残響（エコー）も消す")
            denoise = gr.Checkbox(True, label="雑音も消す")
        sep_btn = gr.Button("声を取り出す", variant="primary")
        with gr.Accordion("📁 きれいな録音（1人の声だけ・BGMなし）を、そのまま素材にする", open=False):
            uploads = gr.File(file_count="multiple", file_types=list(AUDIO_EXT), label="録音ファイル（STEP3 でいっしょに保存されます）")

        gr.Markdown("### STEP2　使う声を選ぶ", elem_classes="step")
        sep_job = gr.Dropdown(sep_jobs(), label="取り出した結果（新しい順。前の結果もここから開けます）")
        sep_info = gr.Markdown()
        spk_audio, spk_check = [], []
        for _ in range(MAX_SPK):
            with gr.Row():
                spk_audio.append(gr.Audio(type="filepath", visible=False, scale=4))
                spk_check.append(gr.Checkbox(False, visible=False, scale=1))
        sep_paths = gr.State([])
        sep_shown = gr.State(None)

        gr.Markdown("### STEP3　話者として保存", elem_classes="step")
        with gr.Row(visible=not _first) as new_fields:
            new_name = gr.Textbox(label="話者の名前", placeholder="例: あゆみん", scale=2)
            new_icon = gr.Image(label="アイコン（任意。あとからでも変えられます）", type="filepath", height=160, scale=1)
        save_btn = gr.Button("STEP3: この声で話者を作る", variant="primary")
        raw_info = gr.Markdown(raw_summary(None))

        gr.Markdown("### STEP4　学習する", elem_classes="step")
        gr.Markdown("押したら放っておいてOK（画面を閉じても続きます）。目安: 素材25分で1時間ほど。")
        with gr.Accordion("⚙ 学習の設定", open=False):
            size = gr.Radio(["1.7B", "0.6B"], value="1.7B", label="読み上げモデルの大きさ（メモリ不足なら 0.6B）")
            tts_epochs = gr.Slider(1, 20, value=3, step=1, label="文章読み上げの学習回数（周）。各周を自動で採点して一番良いものを使います")
            vc_steps = gr.Slider(500, 20000, value=3000, step=500, label="声変換の学習ステップ数")
            do_tts = gr.Checkbox(True, label="文章読み上げを学習する")
            do_vc = gr.Checkbox(True, label="声変換を学習する")
        with gr.Group(visible=test_mode()) as t1:
            gr.Markdown("**🧪 テスト機能**")
            with gr.Row():
                use_purity = gr.Checkbox(False, label="素材の混ざり（相手の声など）を除外")
                purity = gr.Slider(0.3, 0.8, value=0.55, step=0.05, label="この似ている度未満のクリップを除外")
            eval_sim = gr.Checkbox(False, label="読み上げモデルの採点に「声の近さ」も使う")
        with gr.Row():
            start_btn = gr.Button("STEP4: 学習スタート", variant="primary")
            stop_btn = gr.Button("止める", variant="stop")
        status = gr.Textbox(label="学習の状態", lines=4, interactive=False)
        log = gr.Textbox(label="ログ（最後の12行）", lines=6, interactive=False)

    with gr.Accordion("⚙ 全体の設定", open=False):
        test_sw = gr.Checkbox(test_mode(), label="🧪 テストモード（試験中の機能を表示する）")
        with gr.Group(visible=test_mode()) as t0:
            gr.Markdown("**🧪 ディスクの整理**（取り出した結果の大きいファイルを消す。人ごとに分けたファイルは残す）")
            with gr.Row():
                clean_days = gr.Slider(0, 30, value=7, step=1, label="何日より前のものを消すか")
                clean_btn = gr.Button("整理する")
            disk_info = gr.Markdown(disk_usage())

    # ---- 動き
    use_outputs = [cur, target, use_col, add_col, title, ckpt, set_name, set_icon]
    add_outputs = [target, use_col, add_col, add_title, new_fields, save_btn, raw_info]
    gallery.select(on_gallery, None, [cur, target, use_col, add_col, add_title, new_fields, save_btn, raw_info,
                                      title, ckpt, set_name])
    grow_btn.click(show_add, cur, add_outputs)
    close_btn.click(show_use, cur, use_outputs)
    set_btn.click(save_settings, [cur, set_name, set_icon], [gallery, title, set_icon])
    del_btn.click(delete_speaker, [cur, del_sure], [gallery] + use_outputs + [del_sure])

    spk_outputs = [x for pair in zip(spk_audio, spk_check) for x in pair]
    job_outputs = [sep_info, sep_paths, sep_shown] + spk_outputs
    sep_btn.click(start_separate, [target, sep_files, sep_urls, dereverb, denoise, speakers], sep_job)
    sep_job.change(render_job, sep_job, job_outputs)
    gr.Timer(5).tick(tick_job, [sep_job, sep_shown], job_outputs)
    save_btn.click(save_material, [target, new_name, new_icon, sep_paths, uploads] + spk_check,
                   [target, cur, gallery, add_title, new_fields, save_btn, raw_info, uploads, title])

    start_btn.click(start_training, [target, size, tts_epochs, vc_steps, do_tts, do_vc, use_purity, purity, eval_sim],
                    status)
    stop_btn.click(stop_training, target, status)
    gr.Timer(5).tick(refresh, target, [status, log])

    conv_btn.click(do_convert, [cur, src, steps, auto_pitch, semitone] + conv_fx,
                   [conv_out, conv_file, conv_info, conv_raw])
    conv_redo.click(finish, [conv_raw] + conv_fx, [conv_out, conv_file])
    speak_btn.click(do_speak, [cur, text, ckpt, instruct, chain_vc] + speak_fx,
                    [speak_out, speak_file, speak_info, speak_raw])
    speak_redo.click(finish, [speak_raw] + speak_fx, [speak_out, speak_file])
    reload_ckpt.click(ckpt_choices, cur, ckpt)

    test_sw.change(set_test_mode, test_sw, [t0, t1, t2, t3])
    clean_btn.click(cleanup_separated, clean_days, disk_info)
    rt_btn.click(start_realtime)

    def _on_load(sid):
        i = spk.info(sid) if sid else None
        return gr.update(value=gallery_items()), ckpt_choices(sid), gr.update(choices=sep_jobs()), \
            (i["name"] if i else ""), speaker_title(sid)

    demo.load(_on_load, cur, [gallery, ckpt, sep_job, set_name, title])


if __name__ == "__main__":
    # アイコン（voices/）と結果（outputs/）を画面に出せるようにする
    opts = dict(allowed_paths=[str(OUTPUTS), str(VOICES.resolve())], css=CSS)
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
