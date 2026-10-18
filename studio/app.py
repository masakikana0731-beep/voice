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

from studio import engine, natural, post, style
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
N_TEST_GROUPS = 1


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
    """古い抽出結果の大きいファイル（全員分の声 *_voice.wav）を消す。話者別のファイルは残す。"""
    limit = time.time() - float(days) * 86400
    freed = 0
    for f in SEP_DIR.glob("*/*_voice.wav"):
        if f.stat().st_mtime < limit:
            freed += f.stat().st_size
            f.unlink()
    gr.Info(f"{freed / 1e9:.2f}GB 空けました")
    return disk_usage()


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
    vc = "✓" if i["vc"] else "未学習（簡易モード）"
    tts = "✓" if i["tts"] else "未学習（簡易モード）"
    mins = f"　素材 {i['minutes']}分" if i["minutes"] else ""
    agree = "✅ 同意あり" if spk.consent(sid) else "⛔ 同意の記録なし"
    return f"### 🎤 {i['name']}\n声変換 {vc}　読み上げ {tts}{mins}　{agree}"


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

def _size(files) -> str:
    b = sum(Path(f).stat().st_size for f in files if Path(f).exists())
    return f"{b / 1e6:.0f}MB" if b < 1e9 else f"{b / 1e9:.1f}GB"


def cleanup_offer(job, paths, checks):
    """保存したあと、使わなかった声のデータを消すかを聞く欄の中身。消すものがなければ隠す。"""
    if not job or not (SEP_DIR / job).exists():
        return gr.update(visible=False), "", None
    unused = [f for f, c in zip(paths or [], checks) if not c and Path(f).exists()]
    mixed = [str(f) for f in (SEP_DIR / job).glob("*_voice.wav")]
    everything = [str(f) for f in (SEP_DIR / job).rglob("*") if f.is_file()]
    if not unused and not mixed:
        return gr.update(visible=False), "", None
    names = "、".join(f"声{paths.index(f) + 1}" for f in unused) or "なし"
    md = (f"### 🗑 使わなかったデータを消しますか？\n"
          f"選んだ声は話者の素材にコピー済みなので、消してもなくなりません。\n\n"
          f"- **使わなかった声だけ削除**: {names} ＋ 全員分がまざった音声（{_size(unused + mixed)} 空きます）\n"
          f"- **取り出した結果をまるごと削除**: この動画から取り出したデータ全部（{_size(everything)} 空きます）\n"
          f"- **残す**: あとで別の人の声も使いたいとき")
    return gr.update(visible=True), md, {"job": job, "unused": unused, "mixed": mixed}


def cleanup_unused(state):
    """使わなかった声と、全員分の音声を消す。結果一覧（speakers.json）からも外す。"""
    if not state:
        return gr.update(visible=False), gr.skip()
    for f in state["unused"] + state["mixed"]:
        Path(f).unlink(missing_ok=True)
    sj = SEP_DIR / state["job"] / "speakers" / "speakers.json"
    if sj.exists():
        keep = [x for x in json.loads(sj.read_text(encoding="utf-8")) if Path(x["file"]).exists()]
        sj.write_text(json.dumps(keep, ensure_ascii=False, indent=2), encoding="utf-8")
    gr.Info(f"使わなかったデータを削除しました（{len(state['unused']) + len(state['mixed'])} ファイル）")
    return gr.update(visible=False), gr.update(choices=sep_jobs())


def cleanup_all(state):
    if not state:
        return gr.update(visible=False), gr.skip()
    shutil.rmtree(SEP_DIR / state["job"], ignore_errors=True)
    gr.Info("取り出した結果をまるごと削除しました")
    return gr.update(visible=False), gr.update(choices=sep_jobs(), value=None)


def save_material(target, name, icon_file, job, paths, uploads, *checks):
    """チェックした声（と、そのまま使う録音）を話者の素材に入れる。新しい話者ならここで作る。
    そのあと、使わなかった声のデータを消すかを聞く。"""
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
    return [target, target, gr.update(value=gallery_items()), add[3], add[4], add[5], add[6], None, speaker_title(target),
            *cleanup_offer(job, paths, checks)]


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
        cmd += ["--skip", "vc", "--skip", "rt"]  # リアルタイム用も声変換の一種なので一緒に飛ばす
    if spk.vc_model(sid):
        cmd += ["--vc-continue"]  # 学習済みなら、その続きから（素材を足した再学習・延長）
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


# ---------------------------------------------------------------- 声変換・読み上げ・歌声

def save_button(mp3: Path) -> str:
    """保存用ボタンの HTML。普通のリンクだとスマホは再生アプリで開いてしまうので、
    「保存してください」という指定（Content-Disposition: attachment）付きで配る /dl を使う。"""
    from urllib.parse import quote

    rel = Path(mp3).resolve().relative_to(OUTPUTS).as_posix()
    return (f'<a href="dl?f={quote(rel)}" download="{Path(mp3).name}" '
            'style="display:block;text-align:center;padding:12px;margin:4px 0;border-radius:10px;'
            'background:#e8590c;color:#fff;font-weight:bold;text-decoration:none;font-size:16px">'
            f'⬇ 保存する（{Path(mp3).name}）</a>')


def to_mp3(wav) -> str:
    """mp3 を作り、保存用ボタンの HTML を返す。"""
    mp3 = Path(wav).with_suffix(".mp3")
    subprocess.run([ffmpeg(), "-v", "error", "-y", "-i", str(wav), "-b:a", "160k", str(mp3)], check=True)
    return save_button(mp3)


def finish(raw, preset, reverb, echo, ambience, lips, ai_clean=False):
    """仕上げ（AI クリーンアップ・リップノイズ・響き・環境）をかけて、試聴用・保存用を返す。"""
    if not raw:
        raise gr.Error("先に音声を作ってください")
    fx = post.apply(raw, preset, reverb, echo, ambience, lips, ai_clean=ai_clean)
    return str(fx), to_mp3(fx)


def _busy_warning():
    if any(_alive(read_status(VOICES / s).get("pid")) for s in spk.list_ids()):
        gr.Warning("学習中です。変換も学習も遅くなります")


def _need_consent(sid, what):
    if not spk.consent(sid):
        raise gr.Error(f"「{what}」は、本人の同意を記録した話者だけ使えます（⚙ この話者の設定 → 同意の記録）")


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


def do_batch(sid, files, *fx):
    """まとめて変換。1 本ずつ変換して仕上げ、保存ボタンを並べて返す。"""
    if not sid or not files:
        raise gr.Error("話者を選び、録音ファイルを入れてください")
    _busy_warning()
    buttons, t0 = [], time.time()
    for f in files:
        try:
            out, _ = engine.convert(sid, f)
            fx, _ = finish(str(out), *fx)
            buttons.append(to_mp3(fx))
        except Exception as e:
            buttons.append(f"<p>⚠ {Path(f).name}: {e}</p>")
    return "".join(buttons) + f"<p>{len(files)} 本（{time.time() - t0:.0f}秒）</p>"


def _nat(pauses, speed, breath, filler, creak) -> dict:
    return {"pauses": bool(pauses), "speed": round(float(speed), 2), "breath": int(breath), "filler": int(filler),
            "creak": int(creak)}


def _nat_label(nat) -> str:
    parts = (["間"] if nat["pauses"] else []) + ([f"速さ{nat['speed']:.2f}倍"] if nat["speed"] != 1.0 else [])
    parts += [f"{natural.CONSENT_KEYS[k]}{nat[k]}" for k in natural.CONSENT_KEYS if nat[k]]
    return "・".join(parts) or "なし"


def do_speak(sid, text, emotion, chain_vc, ckpt, instruct, pauses, speed, breath, filler, creak, *fx):
    if not sid:
        raise gr.Error("話者を選んでください")
    if not text.strip():
        raise gr.Error("読ませる文章を入れてください")
    nat = _nat(pauses, speed, breath, filler, creak)
    for name in natural.needs_consent(nat):
        _need_consent(sid, name)
    natural.save_settings(sid, **nat)  # 次に開いたとき・チャットでの作成でも同じ設定を使う
    _busy_warning()
    try:
        if emotion and emotion != "なし":
            _need_consent(sid, "感情つき読み上げ")
            out, info = engine.speak_emotion(sid, text, emotion, nat)
        else:
            out, info = engine.speak(sid, text, None if (ckpt or "（").startswith("（") else ckpt, instruct, nat=nat)
            if chain_vc and spk.vc_model(sid):
                # 読み上げた音声に声変換をもう一度かけて、声を話者に近づける（似ている度 0.64 → 0.77 の実測）
                out, info2 = engine.convert(sid, str(out), name="tts_vc")
                info += f" → 声変換で仕上げ（{info2}）"
        out = engine.finish_natural(sid, out, nat)  # 息継ぎ・文末の脱力は声変換のあとに入れる
        info += f" / 自然さ: {_nat_label(nat)}"
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(str(e))
    return (*finish(str(out), *fx), info, str(out))


# ---------------------------------------------------------------- 自然さ（設定・息の音・AI の採点）

def nat_values(sid):
    st = natural.settings(sid) if sid else natural.DEFAULTS
    return st["pauses"], st["speed"], st["breath"], st["filler"], st["creak"]


def breath_view(sid):
    n, preview = natural.breath_info(sid) if sid else (0, None)
    md = (f"息の音: **{n} 個**（下で試聴できます。変な音が混ざっていたら教えてください）" if n
          else "息の音: まだ取り出していません（右のボタンで取り出せます。1分ほど）")
    return md, (str(preview) if preview else None)


def do_extract_breaths(sid):
    if not sid:
        raise gr.Error("話者を選んでください")
    try:
        n = natural.extract_breaths(sid)
    except Exception as e:
        raise gr.Error(f"息の音を取り出せませんでした: {e}")
    gr.Info(f"息の音を {n} 個取り出しました")
    return breath_view(sid)


def do_mos(sid, path):
    """結果の自然さを AI で採点。比べやすいよう、本物の素材の点数（初回だけ測って保存）も出す。"""
    if not path:
        raise gr.Error("先に音声を作ってください")
    ref_file = VOICES / sid / "ref_hq.wav" if sid else None
    cache = VOICES / sid / "mos_ref.json" if sid else None
    ref = json.loads(cache.read_text(encoding="utf-8")).get("mos") if cache and cache.exists() else None
    files = [path] + ([ref_file] if ref is None and ref_file and ref_file.exists() else [])
    try:
        res = engine.mos(files)
    except Exception as e:
        raise gr.Error(f"採点できませんでした: {e}")
    if ref is None and ref_file and ref_file.name in res:
        ref = res[ref_file.name]
        cache.write_text(json.dumps({"mos": ref}), encoding="utf-8")
    v = res[Path(path).name]
    return (f"📏 自然さ（AI の採点）: **{v:.2f}** / 5" + (f"　（本物の素材: {ref:.2f}）" if ref is not None else "")
            + "　※英語中心で学習した AI なので、点数そのものより同じ文での比べっこに使ってください")


# ---------------------------------------------------------------- 聞き比べ（どちらが自然か）

AB_SAMPLES = [
    "今日はとってもいい天気ですね。こんな日は、どこかに出かけたくなります。",
    "最近ハマっているのは、朝のお散歩です。ちょっと早起きするだけで、一日が長く感じるんですよ。",
    "このあいだ、久しぶりに映画を観に行きました。最後のシーンで、思わず泣いちゃいました。",
    "みなさんは、寝る前に何をしていますか？わたしは、温かいお茶を飲むのが好きです。",
    "それでは、今日はこのへんで。また来週も、聴いてくださいね。",
]
AB_NAMES = {"pauses": "間を自然にする", "speed": "話す速さ", "breath": "息継ぎ", "filler": "話し言葉", "creak": "文末の脱力"}


def _ab_file(sid):
    return VOICES / sid / "ab.json"


def _ab_votes(sid) -> list:
    try:
        return json.loads(_ab_file(sid).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def _ab_value(dim, v) -> str:
    if dim == "pauses":
        return "オン" if v else "オフ"
    if dim == "speed":
        return f"{float(v):.2f}倍"
    return f"{v}" if v else "オフ"


def ab_make(sid, text, ckpt, chain_vc, pauses, speed, breath, filler, creak):
    """今の設定（基準）と、設定を 1 つだけ変えたものを作り、A/B をくじで決めて並べる。"""
    import random

    if not sid:
        raise gr.Error("話者を選んでください")
    base = _nat(pauses, speed, breath, filler, creak)
    for name in natural.needs_consent(base):
        _need_consent(sid, name)
    dims = ["pauses", "speed"]
    if spk.consent(sid):  # 息継ぎ・話し言葉・文末の脱力は同意の記録がある話者だけ
        dims += ["filler", "creak"] + (["breath"] if natural.breath_info(sid)[0] else [])
    # まだ比べた回数が少ない項目から選ぶ
    counts = {d: sum(1 for v in _ab_votes(sid) if v["dim"] == d) for d in dims}
    dim = random.choice([d for d in dims if counts[d] == min(counts.values())])
    other = dict(base)
    if dim == "pauses":
        other["pauses"] = not base["pauses"]
    elif dim == "speed":
        cands = [round(base["speed"] + d, 2) for d in (-0.1, 0.1) if 0.85 <= base["speed"] + d <= 1.25]
        other["speed"] = random.choice(cands)
    else:
        other[dim] = 0 if base[dim] else 50
    pair = [base, other]
    random.shuffle(pair)
    for f in (OUTPUTS / sid).glob("ab*.wav"):  # 前回の聞き比べの音は消す
        f.unlink(missing_ok=True)
    text = (text or "").strip() or random.choice(AB_SAMPLES)
    _busy_warning()
    try:
        a, b = engine.speak_variants(sid, text, None if (ckpt or "（").startswith("（") else ckpt, pair,
                                     chain_vc=bool(chain_vc))
    except PermissionError as e:
        raise gr.Error(str(e))
    except Exception as e:
        raise gr.Error(f"作れませんでした: {e}")
    state = {"dim": dim, "a": pair[0][dim], "b": pair[1][dim], "files": [str(a), str(b)], "text": text}
    return str(a), str(b), "A と B を聞いて、**自然なほう**を選んでください（何が違うかは選んだあとに出ます）", state


def ab_vote(sid, state, choice):
    if not sid or not state or state.get("voted"):
        raise gr.Error("先に「2つ作る」を押してください")
    votes = _ab_votes(sid)
    votes.append({"time": time.strftime("%Y-%m-%d %H:%M"), "dim": state["dim"], "a": state["a"], "b": state["b"],
                  "choice": choice, "text": state["text"]})
    _ab_file(sid).write_text(json.dumps(votes, ensure_ascii=False, indent=1), encoding="utf-8")
    name, d = AB_NAMES[state["dim"]], state["dim"]
    picked = {"a": "A", "b": "B", "same": "どちらとも言えない"}[choice]
    md = (f"違い: **{name}**　A = {_ab_value(d, state['a'])} / B = {_ab_value(d, state['b'])}"
          f"　→ あなたの答え: **{picked}**")
    return md, {**state, "voted": True}, ab_tally(sid)


def ab_ai(state):
    if not state:
        raise gr.Error("先に「2つ作る」を押してください")
    try:
        r = engine.mos(state["files"])
    except Exception as e:
        raise gr.Error(f"採点できませんでした: {e}")
    a, b = (r[Path(f).name] for f in state["files"])
    return f"🤖 AI の採点: A **{a:.2f}** / B **{b:.2f}**（5 点満点。AI の好みと人の耳は違うことがあります）"


def _ab_stats(sid) -> dict:
    """{項目: {値: [選ばれた回数, 出た回数]}}。「どちらとも言えない」は両方に 0.5。"""
    stats = {}
    for v in _ab_votes(sid):
        st = stats.setdefault(v["dim"], {})
        for side in ("a", "b"):
            row = st.setdefault(json.dumps(v[side]), [0.0, 0])
            row[1] += 1
            row[0] += 1.0 if v["choice"] == side else (0.5 if v["choice"] == "same" else 0.0)
    return stats


def ab_tally(sid):
    stats = _ab_stats(sid) if sid else {}
    if not stats:
        return "まだ聞き比べの結果はありません"
    lines = ["| 項目 | 設定 | 選ばれた割合 |", "| --- | --- | --- |"]
    for dim, st in stats.items():
        for key, (w, n) in sorted(st.items()):
            lines.append(f"| {AB_NAMES[dim]} | {_ab_value(dim, json.loads(key))} | {w / n:.0%}（{n} 回中）|")
    return "\n".join(lines)


def ab_apply(sid):
    """3 回以上比べて、6 割以上選ばれた設定を、読み上げの初期設定にする。"""
    if not sid:
        raise gr.Error("話者を選んでください")
    changed = {}
    for dim, st in _ab_stats(sid).items():
        best = max(((json.loads(k), w / n) for k, (w, n) in st.items() if n >= 3), key=lambda x: x[1], default=None)
        if best and best[1] >= 0.6:
            changed[dim] = best[0]
    if not changed:
        gr.Info("まだ決め手になる結果がありません（各項目 3 回以上・6 割以上で反映します）")
        return nat_values(sid)
    for name in natural.needs_consent(changed):
        _need_consent(sid, name)
    natural.save_settings(sid, **changed)
    gr.Info("読み上げの初期設定にしました: " + "、".join(f"{AB_NAMES[k]} {_ab_value(k, v)}" for k, v in changed.items()))
    return nat_values(sid)


def do_sing(sid, song, separate, semitone, accomp_db):
    if not sid or not song:
        raise gr.Error("話者を選び、曲（または自分の歌の録音）を入れてください")
    _need_consent(sid, "歌声変換")
    _busy_warning()
    try:
        out, info = engine.sing(sid, song, bool(separate), int(semitone), float(accomp_db))
    except Exception as e:
        raise gr.Error(str(e))
    return str(out), to_mp3(out), info


def history_choices(sid):
    files = engine.history(sid) if sid else []
    kinds = {"vc": "🎙 声変換", "tts": "📝 読み上げ", "emo": "😊 感情", "sing": "🎵 歌", "tts_vc": "📝 読み上げ"}

    def label(f):
        kind = next((v for k, v in kinds.items() if f.stem.startswith(k + "_")), "🔊")
        t = time.strftime("%m/%d %H:%M", time.localtime(f.stat().st_mtime))
        return f"{t}  {kind}  {f.stem[-20:]}"

    return gr.update(choices=[(label(f), str(f)) for f in files], value=str(files[0]) if files else None)


def history_pick(path):
    if not path or not Path(path).exists():
        return None, ""
    return path, save_button(Path(path))


def dict_rows():
    return engine.load_dict() or [["", ""]]


def dict_save(df):
    rows = df.values.tolist() if hasattr(df, "values") else df
    engine.save_dict(rows)
    gr.Info(f"読み方辞書を保存しました（{len(engine.load_dict())} 件）")
    return engine.load_dict() or [["", ""]]


# ---------------------------------------------------------------- この話者の設定（同意・最適化）

def consent_md(sid):
    c = spk.consent(sid) if sid else None
    if c:
        return f"✅ **同意の記録あり**：{c['person']} / {c['date']} / {c['method']}（記録 {c['recorded']}）"
    return ("⛔ **同意の記録なし**：感情つき読み上げ・歌声変換・チャットでの自動作成・Live での通話は使えません。"
            "本人から同意を得たら、下に記録してください。")


def consent_values(sid):
    c = (spk.consent(sid) if sid else None) or {}
    return c.get("person", ""), c.get("date", ""), c.get("method"), c.get("note", ""), False, consent_md(sid)


def save_consent(sid, person, date, method, note, sure):
    if not sid:
        raise gr.Error("話者を選んでください")
    if not sure:
        raise gr.Error("「この人本人から同意を得ています」にチェックを入れてください")
    try:
        spk.set_consent(sid, person, date, method, note)
    except ValueError as e:
        raise gr.Error(str(e))
    gr.Info("同意を記録しました")
    return consent_md(sid), speaker_title(sid), False


def revoke_consent(sid):
    if sid:
        spk.clear_consent(sid)
        gr.Info("同意の記録を取り消しました")
    return consent_md(sid), speaker_title(sid)


def start_tune(sid):
    """声変換の設定の最適化（参照音声の選び直し・寄せ具合）を裏で動かす。2 分ほど。"""
    if not sid or not spk.vc_model(sid):
        raise gr.Error("声変換を学習した話者で使えます")
    engine.stop_vc_server()
    with open(voice_dir(sid) / "tune.log", "w", encoding="utf-8") as log:
        subprocess.Popen([str(PY), "-m", "studio.vc_tune", sid], cwd=ROOT, env=child_env(), stdout=log,
                         stderr=subprocess.STDOUT, creationflags=DETACH)
    return "⏳ 最適化を始めました（3分ほど。似ている度と自然さで選びます）。話者を選び直すと結果が出ます"


def tune_md(sid):
    st = engine.vc_settings(sid) if sid else {}
    if not st:
        return "まだ最適化していません"
    nat = f"　自然さ（AI）{st['mos_baseline']} → {st['mos']}" if st.get("mos") is not None else ""
    return (f"最適化済み（{st.get('tuned', '')}）: 似ている度 {st.get('baseline')} → **{st.get('score')}**{nat}"
            f"（参照音声 {st.get('ref')} / 寄せ具合 {st.get('cfg')}）")


def warm(sid):
    """話者を選んだら、声変換のモデルを裏で先に読み込んでおく（1 回目の待ちをなくす）。学習中はしない。"""
    import threading

    if not sid or any(_alive(read_status(VOICES / s).get("pid")) for s in spk.list_ids()):
        return
    threading.Thread(target=lambda: _safe(engine.warm_vc, sid), daemon=True).start()


def _safe(fn, *a):
    try:
        fn(*a)
    except Exception:
        pass


def finish_controls():
    """仕上げの設定欄（声変換・読み上げ共通）。"""
    with gr.Accordion("🎛 仕上げ（クリーンアップ・リップノイズ・響き・環境）", open=False):
        ai_clean = gr.Checkbox(False, label="✨ AI クリーンアップ（変換で出るザラつき・雑音を取る。+10秒ほど）")
        preset = gr.Dropdown(post.PRESETS, value="なし", label="環境（プリセット）")
        with gr.Row():
            lips = gr.Slider(0, 100, value=30, step=5, label="リップノイズ（話し始めの口の音）")
            reverb = gr.Slider(0, 100, value=0, step=5, label="リバーブ（響き）")
        with gr.Row():
            echo = gr.Slider(0, 100, value=0, step=5, label="エコー（やまびこ）")
            ambience = gr.Slider(0, 100, value=0, step=5, label="環境音（空気感）")
        redo = gr.Button("仕上げだけやり直す（変換し直さずに、この設定をかけ直す）")
    return [preset, reverb, echo, ambience, lips, ai_clean], redo


def result_area():
    out = gr.Audio(label="結果", type="filepath")
    save = gr.HTML()
    info = gr.Markdown(elem_classes="muted")
    with gr.Row():
        mos_btn = gr.Button("📏 自然さを AI で採点（20秒ほど）", size="sm", scale=0, min_width=220)
        mos_md = gr.Markdown()
    return out, save, info, gr.State(None), mos_btn, mos_md


# ---------------------------------------------------------------- 画面

# 話者カードは小さく並べる（画面が広いほど 1 行に多く）
CSS = style.CSS + """
#speakers .caption-label { font-size: 14px !important; white-space: normal !important; }
/* カードは 1 枚 130px 前後で横に並べる（少ないときに巨大化しないように） */
#speakers .grid-container { grid-template-columns: repeat(auto-fill, minmax(110px, 140px)) !important; }
"""


def open_live():
    """PC 専用の Voice Studio Live（リアルタイム声変換）を、この PC で開く。"""
    subprocess.Popen(["cmd", "/c", "start", "", str(ROOT / "start-live.bat")], cwd=ROOT)
    gr.Info("PC の画面に Voice Studio Live が開きます（少し待ってください）")


_first = (spk.list_ids() or [None])[0]
EMO_CHOICES = ["なし"] + list(engine.EMOTIONS)

with gr.Blocks(title="Voice Studio") as demo:
    gr.Markdown("# 🎤 Voice Studio\n**話者を選ぶ** → 下のタブで作る。新しい話者は「＋」から。")
    gallery = gr.Gallery(gallery_items(), label="話者（タップして選ぶ）", columns=6, height=190,
                         allow_preview=False, object_fit="cover", elem_id="speakers")
    cur = gr.State(_first)        # 使う話者
    target = gr.State(None)       # 追加・育てる対象（None = 新しい話者）

    # ================================================================ ふだんの画面
    with gr.Column(visible=bool(_first)) as use_col:
        title = gr.Markdown(speaker_title(_first))
        with gr.Tabs():
            # ---- 🎙 声変換
            with gr.Tab("🎙 声変換"):
                gr.Markdown("つけたい**抑揚・間・感情のとおりに自分で読んで録音** → 声だけがこの話者の声になります。",
                            elem_classes="muted")
                src = gr.Audio(sources=["microphone", "upload"], type="filepath", label="自分の録音（マイク or ファイル）")
                conv_btn = gr.Button("🎙 変換する", variant="primary", elem_classes="big")
                conv_out, conv_file, conv_info, conv_raw, conv_mos_btn, conv_mos = result_area()
                conv_fx, conv_redo = finish_controls()
                with gr.Accordion("⚙ 詳しい設定", open=False):
                    auto_pitch = gr.Checkbox(True, label="声の高さを話者に合わせる（男女差があるときはオン）")
                    semitone = gr.Slider(-12, 12, value=0, step=1, label="高さの微調整（半音）")
                    steps = gr.Slider(10, 100, value=50, step=5, label="品質（多いほど高品質・遅い）")
                with gr.Accordion("📦 まとめて変換（複数の録音を一度に）", open=False):
                    batch_files = gr.File(file_count="multiple", file_types=list(AUDIO_EXT + VIDEO_EXT), label="録音ファイル")
                    batch_btn = gr.Button("まとめて変換する")
                    batch_out = gr.HTML()
                with gr.Accordion("🔴 リアルタイム声変換（PC 専用・別アプリ）", open=False):
                    gr.Markdown("話しながらその場で声を変える機能は、PC 専用の **Voice Studio Live**（デスクトップの赤いアイコン）にあります。")
                    live_btn = gr.Button("🔴 Voice Studio Live を開く（PC の画面に開きます）")

            # ---- 📝 読み上げ
            with gr.Tab("📝 読み上げ"):
                text = gr.Textbox(label="読ませる文章（長くても大丈夫。文ごとに区切って読みます）", lines=4)
                with gr.Row():
                    emotion = gr.Dropdown(EMO_CHOICES, value="なし", label="感情（同意の記録がある話者のみ）", scale=2)
                    chain_vc = gr.Checkbox(True, label="声変換で仕上げる（声がより似る・おすすめ）", scale=2)
                speak_btn = gr.Button("📝 読み上げる", variant="primary", elem_classes="big")
                speak_out, speak_file, speak_info, speak_raw, speak_mos_btn, speak_mos = result_area()
                speak_fx, speak_redo = finish_controls()
                with gr.Accordion("🌿 自然さ（間・速さ・息継ぎ・話し言葉）", open=False):
                    gr.Markdown("人の声らしく聞こえるための工夫です（研究で効果が確かめられているものから順）。"
                                "どれが良いかは「👂 聞き比べ」タブで、耳で決められます。", elem_classes="muted")
                    nat_pauses = gr.Checkbox(True, label="間を自然にする（段落・句読点ごとに間の長さを変え、少しゆらぎを入れる）")
                    nat_speed = gr.Slider(0.85, 1.25, value=1.0, step=0.05,
                                          label="話す速さ（倍）。研究では、少し速めのほうが自然に聞こえやすい")
                    gr.Markdown("**ここから下は、同意の記録がある話者のみ**（本人の息の音や話し方の癖で、より本人らしくなるため）",
                                elem_classes="muted")
                    nat_breath = gr.Slider(0, 100, value=0, step=10,
                                           label="息継ぎ（本人の素材から取り出した息の音を文の頭に入れる。0 でオフ）")
                    nat_filler = gr.Slider(0, 100, value=0, step=10,
                                           label="話し言葉（「えっと」「あの」を時々入れる。逆効果のこともあります。0 でオフ）")
                    nat_creak = gr.Slider(0, 100, value=0, step=10,
                                          label="文末の脱力（文の終わりを少しかすれさせる。試験的。0 でオフ）")
                    with gr.Row():
                        breath_md = gr.Markdown()
                        breath_btn = gr.Button("息の音を素材から取り出し直す", scale=0, min_width=220)
                    breath_audio = gr.Audio(label="取り出した息の音（試聴）", type="filepath")
                nat_fx = [nat_pauses, nat_speed, nat_breath, nat_filler, nat_creak]
                with gr.Accordion("📖 読み方辞書（名前や専門用語の読みを登録）", open=False):
                    gr.Markdown("左に書き方、右に読み方（ひらがな・カタカナ）。全部の話者で使われます。", elem_classes="muted")
                    dict_df = gr.Dataframe(dict_rows(), headers=["書き方", "読み方"], col_count=(2, "fixed"),
                                           interactive=True, wrap=True)
                    dict_btn = gr.Button("辞書を保存")
                with gr.Accordion("⚙ 詳しい設定", open=False):
                    with gr.Row():
                        ckpt = gr.Dropdown(label="読み上げモデル（一番上が採点で一番良かったもの）", scale=3)
                        reload_ckpt = gr.Button("一覧を更新", scale=1)
                    instruct = gr.Textbox(label="話し方の指示（感情「なし」のときだけ。学習済みモデルでは効かないことがあります）",
                                          placeholder="例: 嬉しそうに")

            # ---- 👂 聞き比べ
            with gr.Tab("👂 聞き比べ"):
                gr.Markdown("同じ文を、**自然さの設定を 1 つだけ変えて** 2 通り作ります。どちらがどの設定かは隠してあるので、"
                            "聞いて**自然なほう**を選んでください。回数がたまったら、選ばれた設定を読み上げの初期設定にできます。"
                            "（読み上げタブの「声変換で仕上げる」「読み上げモデル」の設定を使います）", elem_classes="muted")
                ab_text = gr.Textbox(label="読ませる文章（空なら例文から選びます）", lines=2)
                ab_btn = gr.Button("👂 2つ作る", variant="primary", elem_classes="big")
                ab_msg = gr.Markdown()
                with gr.Row():
                    ab_a = gr.Audio(label="A", type="filepath")
                    ab_b = gr.Audio(label="B", type="filepath")
                with gr.Row():
                    vote_a = gr.Button("A のほうが自然")
                    vote_same = gr.Button("どちらとも言えない")
                    vote_b = gr.Button("B のほうが自然")
                ab_ai_btn = gr.Button("🤖 AI にも採点させる（20秒ほど）", size="sm")
                ab_ai_md = gr.Markdown()
                with gr.Accordion("📊 これまでの結果", open=False):
                    ab_tally_md = gr.Markdown()
                    ab_apply_btn = gr.Button("選ばれた設定を、読み上げの初期設定にする（各 3 回以上・6 割以上のもの）")
                ab_state = gr.State(None)

            # ---- 🎵 歌声
            with gr.Tab("🎵 歌声"):
                gr.Markdown("曲（または自分で歌った録音）の**歌声をこの話者の声に**変えます。メロディはそのまま。"
                            "同意の記録がある話者のみ使えます。", elem_classes="muted")
                song = gr.Audio(sources=["upload", "microphone"], type="filepath", label="曲 or 歌の録音")
                with gr.Row():
                    separate = gr.Checkbox(True, label="伴奏を分けて、あとで混ぜ直す（曲のとき）")
                    sing_semitone = gr.Slider(-12, 12, value=0, step=1, label="高さ（半音）。男性の歌→女性の声なら +12 など")
                    accomp_db = gr.Slider(-12, 6, value=0, step=1, label="伴奏の音量（dB）")
                sing_btn = gr.Button("🎵 歌声を変換する", variant="primary", elem_classes="big")
                sing_out, sing_file, sing_info, _sing_raw, sing_mos_btn, sing_mos = result_area()

            # ---- 🕘 履歴
            with gr.Tab("🕘 履歴"):
                with gr.Row():
                    hist = gr.Dropdown(label="これまでに作った音声（新しい順）", scale=4)
                    hist_reload = gr.Button("更新", scale=1)
                hist_audio = gr.Audio(label="試聴", type="filepath")
                hist_save = gr.HTML()

        grow_btn = gr.Button("🌱 この話者を育てる（素材を追加して再学習）")
        with gr.Accordion("⚙ この話者の設定（名前・アイコン・同意・最適化・削除）", open=False):
            with gr.Row():
                set_name = gr.Textbox(label="名前", scale=2)
                set_icon = gr.Image(label="アイコン（画像を入れると変更）", type="filepath", height=160, scale=1)
            set_btn = gr.Button("名前・アイコンを保存")

            gr.Markdown("#### 本人の同意の記録", elem_classes="step")
            c_md = gr.Markdown(consent_md(_first))
            with gr.Row():
                c_person = gr.Textbox(label="同意した人（本人の名前）")
                c_date = gr.Textbox(label="同意を得た日", placeholder="2026-10-18")
                c_method = gr.Dropdown(spk.CONSENT_METHODS, label="方法")
            c_note = gr.Textbox(label="メモ（同意の範囲・保存場所など）")
            c_sure = gr.Checkbox(False, label="この人本人から、上の内容で同意を得ています")
            with gr.Row():
                c_save = gr.Button("同意を記録する", variant="primary")
                c_revoke = gr.Button("同意の記録を取り消す")

            gr.Markdown("#### 声変換の最適化", elem_classes="step")
            tune_info = gr.Markdown(tune_md(_first))
            tune_btn = gr.Button("参照音声と設定を選び直す（2分ほど。素材を足したあとにおすすめ）")

            gr.Markdown("#### 削除", elem_classes="step")
            with gr.Row():
                del_sure = gr.Checkbox(False, label="本当に削除する")
                del_btn = gr.Button("この話者を削除", variant="stop")

    # ================================================================ 追加・育てる画面（STEP1〜4）
    with gr.Column(visible=not _first) as add_col:
        with gr.Row():
            add_title = gr.Markdown("## ＋ 新しい話者を追加")
            close_btn = gr.Button("✕ 閉じる", scale=0, min_width=90)

        gr.Markdown("### STEP1　素材を入れる", elem_classes="step")
        gr.Markdown("**YouTube の URL** か動画を入れると、BGM・効果音・残響・雑音を消して声だけを取り出し、話している人ごとに分けます。"
                    "合計 **30分〜1時間** あると上手に学習できます（最低10分）。", elem_classes="muted")
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
        with gr.Group(visible=False) as clean_box:
            clean_md = gr.Markdown()
            with gr.Row():
                clean_unused_btn = gr.Button("使わなかった声だけ削除（おすすめ）", variant="primary")
                clean_all_btn = gr.Button("取り出した結果をまるごと削除", variant="stop")
                clean_keep_btn = gr.Button("残す")
        clean_state = gr.State(None)

        gr.Markdown("### STEP4　学習する", elem_classes="step")
        gr.Markdown("押したら放っておいてOK（画面を閉じても続きます）。学習済みの話者は前回の続きから学習し、"
                    "終わったら前のモデルと比べて良いほうを自動で残します。", elem_classes="muted")
        with gr.Accordion("⚙ 学習の設定", open=False):
            size = gr.Radio(["1.7B", "0.6B"], value="1.7B", label="読み上げモデルの大きさ（メモリ不足なら 0.6B）")
            tts_epochs = gr.Slider(1, 20, value=3, step=1, label="文章読み上げの学習回数（周）。各周を自動で採点して一番良いものを使います")
            vc_steps = gr.Slider(500, 20000, value=3000, step=500, label="声変換の学習ステップ数")
            do_tts = gr.Checkbox(True, label="文章読み上げを学習する")
            do_vc = gr.Checkbox(True, label="声変換（とリアルタイム用）を学習する")
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

    # ================================================================ 全体の設定
    with gr.Accordion("⚙ 全体の設定", open=False):
        gr.Markdown("**🗑 ディスクの整理**（取り出した結果の大きいファイルを消す。人ごとに分けたファイルは残す）")
        with gr.Row():
            clean_days = gr.Slider(0, 30, value=7, step=1, label="何日より前のものを消すか")
            clean_btn = gr.Button("整理する")
        disk_info = gr.Markdown(disk_usage())
        test_sw = gr.Checkbox(test_mode(), label="🧪 テストモード（試験中の機能を表示する）")

    # ================================================================ 動き
    use_outputs = [cur, target, use_col, add_col, title, ckpt, set_name, set_icon]
    add_outputs = [target, use_col, add_col, add_title, new_fields, save_btn, raw_info]
    consent_outputs = [c_person, c_date, c_method, c_note, c_sure, c_md]

    def _after_select(sid):
        """話者が変わったら、話者ごとの欄（同意・最適化・履歴）を入れ替え、声変換のモデルを先読みする。"""
        warm(sid)
        return (*consent_values(sid), tune_md(sid), history_choices(sid), *nat_values(sid), *breath_view(sid),
                ab_tally(sid))

    after_outputs = consent_outputs + [tune_info, hist] + nat_fx + [breath_md, breath_audio, ab_tally_md]
    gallery.select(on_gallery, None, [cur, target, use_col, add_col, add_title, new_fields, save_btn, raw_info,
                                      title, ckpt, set_name]).then(_after_select, cur, after_outputs)
    grow_btn.click(show_add, cur, add_outputs)
    close_btn.click(show_use, cur, use_outputs).then(_after_select, cur, after_outputs)
    set_btn.click(save_settings, [cur, set_name, set_icon], [gallery, title, set_icon])
    del_btn.click(delete_speaker, [cur, del_sure], [gallery] + use_outputs + [del_sure])
    c_save.click(save_consent, [cur, c_person, c_date, c_method, c_note, c_sure], [c_md, title, c_sure])
    c_revoke.click(revoke_consent, cur, [c_md, title])
    tune_btn.click(start_tune, cur, tune_info)

    spk_outputs = [x for pair in zip(spk_audio, spk_check) for x in pair]
    job_outputs = [sep_info, sep_paths, sep_shown] + spk_outputs
    sep_btn.click(start_separate, [target, sep_files, sep_urls, dereverb, denoise, speakers], sep_job)
    sep_job.change(render_job, sep_job, job_outputs)
    gr.Timer(5).tick(tick_job, [sep_job, sep_shown], job_outputs)
    save_btn.click(save_material, [target, new_name, new_icon, sep_shown, sep_paths, uploads] + spk_check,
                   [target, cur, gallery, add_title, new_fields, save_btn, raw_info, uploads, title,
                    clean_box, clean_md, clean_state])
    clean_unused_btn.click(cleanup_unused, clean_state, [clean_box, sep_job]).then(render_job, sep_job, job_outputs)
    clean_all_btn.click(cleanup_all, clean_state, [clean_box, sep_job])
    clean_keep_btn.click(lambda: gr.update(visible=False), None, clean_box)

    start_btn.click(start_training, [target, size, tts_epochs, vc_steps, do_tts, do_vc, use_purity, purity, eval_sim],
                    status)
    stop_btn.click(stop_training, target, status)
    gr.Timer(5).tick(refresh, target, [status, log])

    conv_btn.click(do_convert, [cur, src, steps, auto_pitch, semitone] + conv_fx,
                   [conv_out, conv_file, conv_info, conv_raw]).then(history_choices, cur, hist)
    conv_redo.click(finish, [conv_raw] + conv_fx, [conv_out, conv_file])
    batch_btn.click(do_batch, [cur, batch_files] + conv_fx, batch_out).then(history_choices, cur, hist)
    speak_btn.click(do_speak, [cur, text, emotion, chain_vc, ckpt, instruct] + nat_fx + speak_fx,
                    [speak_out, speak_file, speak_info, speak_raw]).then(history_choices, cur, hist)
    speak_redo.click(finish, [speak_raw] + speak_fx, [speak_out, speak_file])
    sing_btn.click(do_sing, [cur, song, separate, sing_semitone, accomp_db],
                   [sing_out, sing_file, sing_info]).then(history_choices, cur, hist)
    hist_reload.click(history_choices, cur, hist)
    hist.change(history_pick, hist, [hist_audio, hist_save])
    dict_btn.click(dict_save, dict_df, dict_df)
    for b, md, res in ((conv_mos_btn, conv_mos, conv_out), (speak_mos_btn, speak_mos, speak_out),
                       (sing_mos_btn, sing_mos, sing_out)):
        b.click(do_mos, [cur, res], md)
    breath_btn.click(do_extract_breaths, cur, [breath_md, breath_audio])
    ab_btn.click(lambda: "", None, ab_ai_md).then(
        ab_make, [cur, ab_text, ckpt, chain_vc] + nat_fx, [ab_a, ab_b, ab_msg, ab_state])
    for b, choice in ((vote_a, "a"), (vote_b, "b"), (vote_same, "same")):
        b.click(lambda sid, st, c=choice: ab_vote(sid, st, c), [cur, ab_state], [ab_msg, ab_state, ab_tally_md])
    ab_ai_btn.click(ab_ai, ab_state, ab_ai_md)
    ab_apply_btn.click(ab_apply, cur, nat_fx)
    reload_ckpt.click(ckpt_choices, cur, ckpt)
    live_btn.click(open_live)

    test_sw.change(set_test_mode, test_sw, [t1])
    clean_btn.click(cleanup_separated, clean_days, disk_info)

    def _on_load(sid):
        i = spk.info(sid) if sid else None
        return (gr.update(value=gallery_items()), ckpt_choices(sid), gr.update(choices=sep_jobs()),
                (i["name"] if i else ""), speaker_title(sid), *_after_select(sid))

    demo.load(_on_load, cur, [gallery, ckpt, sep_job, set_name, title] + after_outputs)


if __name__ == "__main__":
    # アイコン（voices/）と結果（outputs/）を画面に出せるようにする
    opts = dict(allowed_paths=[str(OUTPUTS), str(VOICES.resolve())], css=CSS, theme=style.theme())
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
