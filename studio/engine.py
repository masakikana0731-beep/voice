"""音声の生成（UI から呼ぶ）。文章→音声 は Qwen3-TTS、録音→声変換 は Seed-VC。"""

import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from studio.common import ROOT, SEEDVC, SPEAKER, VC_CONFIG, VC_PY, VOICES, child_env, ffmpeg, read_status

OUT = ROOT / "outputs"
_lock = threading.Lock()
_cache = {"key": None, "model": None}


def tts_checkpoints(voice: str) -> list[str]:
    """新しい順。採点済みなら一番よく読めるもの（best.txt）を先頭にする。"""
    tdir = VOICES / voice / "tts"
    names = [p.name for p in sorted(tdir.glob("epoch-*"), reverse=True)]
    best = (tdir / "best.txt").read_text(encoding="utf-8").strip() if (tdir / "best.txt").exists() else None
    return [best] + [n for n in names if n != best] if best in names else names


def _load(key, path):
    import torch
    from qwen_tts import Qwen3TTSModel

    if _cache["key"] != key:
        _cache["model"] = None
        torch.cuda.empty_cache()
        _cache["model"] = Qwen3TTSModel.from_pretrained(
            str(path), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
        _cache["key"] = key
    return _cache["model"]


_vc = {"proc": None}
_vc_lock = threading.Lock()


def _vc_request(req: dict) -> None:
    """声変換の常駐係（studio/vc_server.py）に依頼して、終わるまで待つ。

    1 回目だけモデルの読み込みに時間がかかり、2 回目からはすぐに変換が始まる。
    常駐係が落ちていたら起動し直す。
    """
    import json

    with _vc_lock:
        p = _vc["proc"]
        if p is None or p.poll() is not None:
            p = subprocess.Popen(
                [str(VC_PY), str(ROOT / "studio" / "vc_run.py"), str(ROOT / "studio" / "vc_server.py")],
                cwd=SEEDVC, env=child_env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW)
            _vc["proc"] = p
        p.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        p.stdin.flush()
        tail = []
        for line in p.stdout:
            line = line.rstrip()
            if line == "@@DONE":
                return
            if line.startswith("@@ERR"):
                raise RuntimeError("声変換に失敗しました: " + line[5:] + "\n" + "\n".join(tail[-15:]))
            tail.append(line)
        _vc["proc"] = None
        raise RuntimeError("声変換の常駐係が止まりました:\n" + "\n".join(tail[-20:]))


def stop_vc_server() -> None:
    p = _vc["proc"]
    if p is not None and p.poll() is None:
        p.kill()
    _vc["proc"] = None


def unload() -> None:
    """読み込んだモデルを GPU から外す（学習に GPU メモリを譲るため）。声変換の常駐係も止める。"""
    import gc

    import torch

    stop_vc_server()
    from studio import quality

    quality.unload_asr()  # 読み上げの候補選びで読み込んだ Whisper も外す
    with _lock:
        _cache.update(key=None, model=None)
        gc.collect()
        torch.cuda.empty_cache()


def _max_tokens(text: str) -> int:
    """生成の上限（12 トークン/秒）。日本語は 1 秒 6〜8 文字なので、その 2〜3 倍の余裕を持たせる。
    モデルが壊れて話し終わらなくなっても、延々と生成し続けないようにする。"""
    return max(96, int(len(text) * 3.5) + 48)


# ---------------------------------------------------------------- 読み方辞書・文の区切り

DICT_FILE = VOICES / "_dictionary.json"


def load_dict() -> list[list[str]]:
    """読み方辞書 [[表記, 読み], ...]（全話者で共通）。"""
    import json

    try:
        return json.loads(DICT_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def save_dict(rows) -> None:
    import json

    rows = [[str(a).strip(), str(b).strip()] for a, b in rows if str(a).strip() and str(b).strip()]
    DICT_FILE.parent.mkdir(exist_ok=True)
    DICT_FILE.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


def apply_dict(text: str) -> str:
    # 長い表記から先に置き換える（「KIBA BREEZE」より先に「KIBA」が置き換わらないように）
    for a, b in sorted(load_dict(), key=lambda r: -len(r[0])):
        text = text.replace(a, b)
    return text


def split_text(text: str, max_chars: int = 60) -> list[str]:
    """文ごと（。！？改行）に区切り、短い文はまとめて max_chars 以下のかたまりにする。
    長い文章を一度に読ませると崩れやすいので、かたまりごとに読ませてつなぐ。"""
    import re

    sents = [x.strip() for x in re.split(r"(?<=[。！？!?])|\n+", text) if x and x.strip()]
    chunks, cur = [], ""
    for x in sents:
        while len(x) > max_chars:  # 句点のない長い文は読点で切る
            cut = x.rfind("、", 0, max_chars)
            cut = cut + 1 if cut > 10 else max_chars
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(x[:cut])
            x = x[cut:]
        if cur and len(cur) + len(x) > max_chars:
            chunks.append(cur)
            cur = ""
        cur += x
    if cur:
        chunks.append(cur)
    return chunks or [text]


def _items(text: str, nat: dict, rng) -> list[tuple[str, str]]:
    """読ませる文のかたまり [(文, 終わりの記号)]。間を自然にするときは 1 文ずつ（間を文ごとに決めるため）。"""
    from studio import natural

    text = apply_dict(text)
    items = natural.split_sentences(text) if nat.get("pauses") else [(c, "") for c in split_text(text)]
    return natural.add_fillers(items, float(nat.get("filler", 0)), rng)


# 読み上げを何個まとめて作って選ぶか。あゆみんで実測（studio/tts_lab.py、8 文）:
#   1 個だけ 総合 88.7・読み間違い 13%（ときどき言葉を飛ばす・早口で崩れる）→ 4 個から選ぶ 総合 100・読み間違い 3%、
#   似ている度 0.733 → 0.796。まとめて作るので時間はほぼ同じ（97 秒 → 105 秒）
TTS_BEST_OF = 4
TTS_CER_OK = 0.15  # 読み間違いがこれより多い候補は選ばない


def _pick_tts(voice: str, items, cands, sr) -> list:
    """かたまりごとに、読み間違いが少ないもののうち本人の声にいちばん似ている候補を選ぶ。"""
    import soundfile as sf

    from studio import quality

    with tempfile.TemporaryDirectory() as tmp:
        files, owner = [], []
        for i, ws in enumerate(cands):
            for k, w in enumerate(ws):
                f = Path(tmp) / f"c{i:03d}__k{k}.wav"
                sf.write(f, w, sr)
                files.append(f)
                owner.append((i, k))
        try:
            hyps = quality.transcribe(files)
            vdir = VOICES / voice
            refs = sorted((vdir / "raw").glob("*.wav")) or [vdir / "ref_hq.wav"]
            j = quality.judge(files, refs, mos=False, print_cache=vdir / "voiceprint.npy")
        except Exception as e:  # 採点できなければ 1 個目を使う
            print(f"[speak] 選べませんでした（1 個目を使います）: {e}", flush=True)
            return [ws[0] for ws in cands]
    best = []
    for i, ws in enumerate(cands):
        want = len(quality.kana(items[i][0]))
        # (読み間違い, 似ている度, 候補の番号, 余計な声の文字数)。余計な声 = 書き起こしが元の文より長い分。
        # 学習した素材（ラジオ）のクセで、話し終わりに「はぁ」などを付けることがあるため
        sc = [(quality.cer(items[i][0], hyps[n]), j[files[n].name]["sim"], k,
               max(0, len(quality.kana(hyps[n])) - want)) for n, (ii, k) in enumerate(owner) if ii == i]
        ok = [s for s in sc if s[0] <= TTS_CER_OK]
        clean = [s for s in ok if s[3] <= 1] or ok
        if clean:
            # 正しく読めていることを最優先（一番少ない読み間違い +2% まで）、その中で一番似ているもの。
            # 似ている度を先にすると「ラジオ」→「ダチョウ」のような読み間違いの候補を選ぶことがあった
            least = min(s[0] for s in clean)
            k = max((s for s in clean if s[0] <= least + 0.02), key=lambda s: s[1])[2]
        else:
            k = min(sc)[2]
        best.append(ws[k])
    return best


def _generate(voice: str, items, checkpoint: str | None, instruct: str = "", sub: str = "tts", best_of: int = 1):
    """文ごとに読み上げる（_lock の中で呼ぶ）。戻り値: (音声のリスト, サンプリング周波数, 説明)
    best_of > 1 なら、学習済みモデルで同じかたまりを best_of 個まとめて作り、一番良いものを選ぶ。"""
    from tts import fetch_model

    vdir = VOICES / voice
    wavs = []
    if checkpoint:
        model = _load(("ft", voice, sub, checkpoint), vdir / sub / checkpoint)
        n = max(1, int(best_of))
        cands = []
        for c, _ in items:
            # 公式の学習コードは言語指定なし（Auto）の形式で学習するので、読み上げ時も Auto にそろえる。
            # "Japanese" を指定すると学習時と違う先頭トークンになり、話し終わらない・雑音になる
            w, sr = model.generate_custom_voice(text=[c] * n, language=["Auto"] * n, speaker=[SPEAKER] * n,
                                                instruct=[instruct] * n if instruct else None,
                                                max_new_tokens=_max_tokens(c))
            cands.append(w)
        if n > 1:
            return _pick_tts(voice, items, cands, sr), sr, f"学習済みモデル {checkpoint}（{n} 個から選択）"
        return [ws[0] for ws in cands], sr, f"学習済みモデル {checkpoint}"
    if not (vdir / "ref.wav").exists():
        raise RuntimeError("まだ学習データの準備（または学習）をしていません")
    model = _load(("base",), fetch_model("Qwen/Qwen3-TTS-12Hz-1.7B-Base"))
    ref_text = (vdir / "ref.txt").read_text(encoding="utf-8")
    for c, _ in items:
        w, sr = model.generate_voice_clone(text=c, language="Japanese", ref_audio=str(vdir / "ref.wav"),
                                           ref_text=ref_text, max_new_tokens=_max_tokens(c))
        wavs.append(w[0])
    return wavs, sr, "未学習（参照音声からのクローン）"


def _check_natural(voice: str, nat: dict) -> None:
    from studio import natural
    from studio.speakers import require_consent

    for name in natural.needs_consent(nat):
        require_consent(voice, name)


def speak(voice: str, text: str, checkpoint: str | None, instruct: str = "", sub: str = "tts",
          nat: dict | None = None, seed: int | None = None, best_of: int = 1) -> tuple[Path, str]:
    """学習済みなら学習した声で、未学習なら参照音声からのクローンで読み上げる。
    読み方辞書を当て、文ごとに区切って読ませてから間を入れてつなぐ。

    nat: 自然さの設定（studio/natural.py）。省略すると以前と同じ（決まった間・速さそのまま）。
    息継ぎ・文末の脱力は声変換のあとにかけるので、ここではなく finish_natural で入れる。"""
    import soundfile as sf

    from studio import natural

    nat = {**natural.OFF, **(nat or {})}
    _check_natural(voice, nat)
    rng = np.random.default_rng(seed)
    items = _items(text, nat, rng)
    with _lock:
        t0 = time.time()
        wavs, sr, how = _generate(voice, items, checkpoint, instruct, sub, best_of)
        out = OUT / voice / f"tts_{time.strftime('%Y%m%d-%H%M%S')}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, natural.assemble(wavs, [k for _, k in items], sr, nat, rng), sr)
    return out, f"{how} / {len(items)} 文 / {time.time() - t0:.1f}秒"


def finish_natural(voice: str, wav, nat: dict | None, seed: int | None = None) -> Path:
    """息継ぎ・文末の脱力を入れる（声変換のあとにかける）。どちらもオフなら何もしない。"""
    import soundfile as sf

    from studio import natural

    nat = nat or {}
    if not (nat.get("breath") or nat.get("creak")):
        return Path(wav)
    _check_natural(voice, nat)
    y, sr = sf.read(wav, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    out = Path(wav).with_name(Path(wav).stem + "_nat.wav")
    sf.write(out, natural.post(y, sr, voice, nat, np.random.default_rng(seed)), sr)
    return out


def mos(paths) -> dict:
    """自然さの自動採点（1〜5。studio/mos.py を .venv-sep で動かす）。{ファイル名: 点}"""
    import json

    from studio.common import SEP_PY

    with tempfile.TemporaryDirectory() as tmp:
        res = Path(tmp) / "mos.json"
        subprocess.run([str(SEP_PY), "-m", "studio.mos", *[str(Path(p).resolve()) for p in paths], "-o", str(res)],
                       cwd=ROOT, env=child_env(), check=True, capture_output=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
        return json.loads(res.read_text(encoding="utf-8"))


def speak_variants(voice: str, text: str, checkpoint: str | None, nats: list[dict], chain_vc: bool = True,
                   name: str = "ab") -> list[Path]:
    """聞き比べ用。同じ文・同じ読み上げ結果から、自然さの設定だけを変えた音声を作る。
    （言いよどみの有無が違うときだけ、文が変わるので読み上げ直す）"""
    import random

    import soundfile as sf

    from studio import natural
    from studio.speakers import vc_model

    nats = [{**natural.OFF, **n} for n in nats]
    for n in nats:
        _check_natural(voice, n)
    seed = random.randrange(1 << 30)
    gen, outs = {}, []
    for i, nat in enumerate(nats):
        key = (bool(nat["pauses"]), float(nat["filler"]))
        if key not in gen:
            items = _items(text, nat, np.random.default_rng(seed))
            with _lock:
                gen[key] = (_generate(voice, items, checkpoint, best_of=TTS_BEST_OF), items)
        (wavs, sr, _), items = gen[key]
        raw = OUT / voice / f"{name}{i}_{time.strftime('%Y%m%d-%H%M%S')}.wav"
        raw.parent.mkdir(parents=True, exist_ok=True)
        sf.write(raw, natural.assemble(wavs, [k for _, k in items], sr, nat, np.random.default_rng(seed)), sr)
        out = raw
        if chain_vc and vc_model(voice):
            out, _ = convert(voice, str(raw), name=f"{name}{i}vc")
            raw.unlink(missing_ok=True)
        outs.append(finish_natural(voice, out, nat, seed))
    return outs


EMOTIONS = {
    "嬉しい": "とても嬉しそうに、明るく弾んだ声で",
    "楽しい": "楽しそうに、笑顔で話すように",
    "悲しい": "悲しそうに、少し声を落として",
    "泣きそう": "今にも泣き出しそうな、震える声で",
    "怒り": "怒った様子で、強い口調で",
    "驚き": "とても驚いた様子で",
    "照れ": "照れくさそうに、少し恥ずかしそうに",
    "ささやき": "耳元でささやくように、小さな声で",
    "元気": "元気いっぱいに、はきはきと",
    "落ち着き": "落ち着いた、穏やかな声で",
}


def speak_emotion(voice: str, text: str, emotion: str, nat: dict | None = None) -> tuple[Path, str]:
    """感情つき読み上げ。感情の指示が効く内蔵の声（Ono_Anna）で読ませてから、この話者の声に変換する。
    本人の同意を記録した話者だけ使える。nat は自然さの設定（息継ぎ・文末の脱力は finish_natural で）。"""
    import soundfile as sf

    from studio import natural
    from studio.speakers import require_consent
    from tts import fetch_model

    require_consent(voice, "感情つき読み上げ")
    nat = {**natural.OFF, **(nat or {})}
    instruct = EMOTIONS.get(emotion, emotion)
    rng = np.random.default_rng()
    items = _items(text, nat, rng)
    with _lock:
        t0 = time.time()
        model = _load(("custom",), fetch_model("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"))
        wavs = []
        for c, _ in items:
            w, sr = model.generate_custom_voice(text=c, language="Japanese", speaker="Ono_Anna", instruct=instruct,
                                                max_new_tokens=_max_tokens(c))
            wavs.append(w[0])
        tmp = OUT / voice / f"emo_src_{time.strftime('%Y%m%d-%H%M%S')}.wav"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        sf.write(tmp, natural.assemble(wavs, [k for _, k in items], sr, nat, rng), sr)
        t1 = time.time() - t0
    out, info = convert(voice, str(tmp), name="emo")
    tmp.unlink(missing_ok=True)
    return out, f"感情「{emotion}」で読み上げ（{t1:.1f}秒）→ 声変換（{info}）"


SR_VC = 44100


def vc_reference(vdir: Path, target_sec: float = 12.0) -> Path:
    """声変換の参照音声（約 12 秒）。1 クリップ（約 8 秒）だけより、声の特徴をつかみやすい。

    Seed-VC は「参照 + 変換する区間」を 30 秒の窓で処理するので、参照は 12 秒程度に抑える。
    """
    import numpy as np
    import soundfile as sf

    ref = vdir / "ref_vc.wav"
    if ref.exists() and ref.stat().st_mtime >= (vdir / "ref_hq.wav").stat().st_mtime:
        return ref
    pieces, total = [], 0.0
    files = [vdir / "ref_hq.wav"] + sorted((vdir / "clips_hq").glob("*.wav"))
    used = set()
    for f in files:
        if f.name in used or total >= target_sec:
            continue
        y, sr = sf.read(f, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if len(y) / sr < 3 or np.sqrt((y ** 2).mean()) < 0.02:  # 短すぎ・小さすぎる区間は使わない
            continue
        y = y[: int(10 * sr)]  # 長いクリップは 10 秒まで使う
        pieces += [y, np.zeros(int(0.2 * sr), dtype=np.float32)]
        total += len(y) / sr
        used.add(f.name)
    sf.write(ref, np.concatenate(pieces)[: int(14 * SR_VC)], SR_VC)
    return ref


def split_at_silence(y, sr, max_sec: float = 15.0):
    """無音のところで区切って max_sec 以下の区間にする（区切りは無音の真ん中）。"""
    import librosa

    iv = librosa.effects.split(y, top_db=35, frame_length=2048, hop_length=512)
    if len(iv) == 0:
        return [(0, len(y))]
    cuts = [0]
    seg_start = 0
    for (s0, e0), (s1, _) in zip(iv[:-1], iv[1:]):
        mid = (e0 + s1) // 2  # 声と声のあいだ（無音）の真ん中
        if s1 - seg_start > max_sec * sr and mid > seg_start:
            cuts.append(mid)
            seg_start = mid
    cuts.append(len(y))
    out = []
    for s0, e0 in zip(cuts[:-1], cuts[1:]):
        # 無音なしで max_sec を超える長い発話は、仕方なく等分する
        n = max(1, int(np.ceil((e0 - s0) / (max_sec * sr))))
        step = (e0 - s0) // n
        out += [(s0 + i * step, e0 if i == n - 1 else s0 + (i + 1) * step) for i in range(n)]
    return out


def vc_settings(voice: str) -> dict:
    """話者ごとの声変換の最適設定（studio/vc_tune.py が決める）。{"ref": ファイル名, "cfg": 0.7, "steps": 50}"""
    import json

    try:
        return json.loads((VOICES / voice / "vc_settings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _vc_ref(voice: str) -> Path:
    st = vc_settings(voice)
    if st.get("ref") and (VOICES / voice / st["ref"]).exists():
        return VOICES / voice / st["ref"]
    return vc_reference(VOICES / voice)


def warm_vc(voice: str) -> None:
    """声変換の常駐係に、この話者のモデルを先に読み込ませておく（1 回目の待ちをなくす）。"""
    from studio.speakers import vc_model

    model = vc_model(voice)
    if model:
        _vc_request({"warm": True, "checkpoint": str(model[0]), "config": str(model[1])})


# 何回変換して選ぶか。あゆみんで実測: 1 回 平均 83.8 点 → 3 回から選ぶ 87.4 点（studio/vc_lab.py、本物 = 100）。
# 2 回目以降の変換は 1 区間 1〜3 秒なので、3 回でも待ちは 10〜20 秒ほど増えるだけ
BEST_OF = 3


def _pick_best(voice: str, tmp: Path, n: int, best_of: int) -> list[Path]:
    """区切りごとに、本人の声にいちばん似ている変換結果を選ぶ（似ている度だけで選ぶ。自然さの採点は遅いため）。"""
    from studio import quality

    dirs = [tmp / "converted"] + [tmp / f"converted{k}" for k in range(1, best_of)]
    cand = tmp / "cands"
    cand.mkdir()
    files = []
    for k, d in enumerate(dirs):
        for i in range(n):
            if (d / f"{i:03d}.wav").exists():
                dst = cand / f"k{k}__{i:03d}.wav"
                shutil.copy(d / f"{i:03d}.wav", dst)
                files.append(dst)
    vdir = VOICES / voice
    refs = sorted((vdir / "raw").glob("*.wav")) or [vdir / "ref_hq.wav"]
    try:
        j = quality.judge(files, refs, mos=False, print_cache=vdir / "voiceprint.npy")
    except Exception as e:  # 採点役が動かなければ 1 回目の結果を使う
        print(f"[convert] 選べませんでした（1 回目を使います）: {e}", flush=True)
        return [dirs[0] / f"{i:03d}.wav" for i in range(n)]
    return [max((f for f in files if f.name.endswith(f"__{i:03d}.wav")), key=lambda f: j[f.name]["sim"])
            for i in range(n)]


def convert(voice: str, source: str, steps: int | None = None, auto_pitch: bool = True, semitone: int = 0,
            name: str = "vc", best_of: int | None = None) -> tuple[Path, str]:
    """録音を学習した声に変換する。抑揚（声の高さの動き）・間はそのまま残る。

    無音で 15 秒以下に区切って変換し、つなぎ直す（Seed-VC 内部の区切りで音が途切れるのを防ぐ）。
    best_of 回（省略時は BEST_OF 回）変換して、区切りごとに一番似ているものを選ぶ。
    """
    import soundfile as sf

    vdir = VOICES / voice
    if not (vdir / "ref_hq.wav").exists():
        raise RuntimeError("まだ学習データの準備（または学習）をしていません")
    ref = _vc_ref(voice)
    st = vc_settings(voice)
    steps = int(steps or st.get("steps", 50))
    from studio.speakers import vc_model

    model = vc_model(voice)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # ブラウザ録音（webm 等）でも読めるよう、先に wav へ変換しておく
        src_wav = tmp / "source.wav"
        subprocess.run([ffmpeg(), "-v", "error", "-y", "-i", str(Path(source).resolve()), "-ac", "1", "-ar", str(SR_VC),
                        str(src_wav)], check=True)
        y, sr = sf.read(src_wav, dtype="float32")
        seg_dir, out_dir = tmp / "segments", tmp / "converted"
        seg_dir.mkdir()
        spans = split_at_silence(y, sr)
        for i, (s0, e0) in enumerate(spans):
            sf.write(seg_dir / f"{i:03d}.wav", y[s0:e0], sr)

        how = "学習済みモデル" if model else "未学習（参照音声からのゼロショット変換）"
        req = {"segments": str(seg_dir), "target": str(ref), "steps": int(steps),
               "auto_pitch": bool(auto_pitch), "semitone": int(semitone), "cfg": float(st.get("cfg", 0.7)),
               "checkpoint": str(model[0]) if model else None, "config": str(model[1]) if model else None}
        best_of = BEST_OF if best_of is None else max(1, int(best_of))
        for k in range(best_of):  # 変換は毎回少しずつ違うので、何回か変換して区切りごとに一番似ているものを選ぶ
            _vc_request({**req, "out": str(out_dir if k == 0 else tmp / f"converted{k}")})
        outs = [out_dir / f"{i:03d}.wav" for i in range(len(spans))]
        if not all(f.exists() for f in outs):
            raise RuntimeError("声変換に失敗しました（結果のファイルがありません）")
        if best_of > 1:
            outs = _pick_best(voice, tmp, len(spans), best_of)
            how += f" / {best_of} 回変換して一番似ているものを選択"

        # つなぎ目は無音の中なので、10ms だけ重ねてなめらかにつなぐ
        fade = int(0.01 * sr)
        joined = None
        for f in outs:
            z, zsr = sf.read(f, dtype="float32")
            if z.ndim > 1:
                z = z.mean(axis=1)
            if joined is None or len(z) <= fade or len(joined) <= fade:
                joined = z if joined is None else np.concatenate([joined, z])
                continue
            w = np.linspace(0, 1, fade, dtype=np.float32)
            joined = np.concatenate([joined[:-fade], joined[-fade:] * (1 - w) + z[:fade] * w, z[fade:]])
        out = OUT / voice / f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, joined, zsr)
    return out, f"{how} / {len(spans)} 区間 / {time.time() - t0:.1f}秒"


def sing(voice: str, source: str, separate: bool = True, semitone: int = 0, accomp_db: float = 0.0) -> tuple[Path, str]:
    """歌声変換。曲から歌声と伴奏を分け、歌声だけをこの話者の声に変えて、伴奏と混ぜ直す。
    メロディ（音の高さ）はそのまま残す。本人の同意を記録した話者だけ使える。"""
    import soundfile as sf

    from studio.common import SEP_PY
    from studio.speakers import require_consent

    require_consent(voice, "歌声変換")
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if separate:
            subprocess.run([str(SEP_PY), "-m", "studio.split_song", str(Path(source).resolve()), "-o", str(tmp)],
                           cwd=ROOT, env=child_env(), check=True, capture_output=True)
            vocal, accomp = tmp / "vocals.wav", tmp / "accompaniment.wav"
        else:
            vocal, accomp = Path(source), None
        # 歌は音の高さを話者に合わせない（メロディが変わってしまうため）。高さは半音単位で手動で変える
        conv, info = convert(voice, str(vocal), auto_pitch=False, semitone=semitone, name="sing_voice",
                             best_of=1)  # 曲は長いので 1 回だけ
        v, sr = sf.read(conv, dtype="float32")
        if accomp is not None and accomp.exists():
            a, asr = sf.read(accomp, dtype="float32", always_2d=True)
            a = a.mean(axis=1)
            if asr != sr:
                import librosa

                a = librosa.resample(a, orig_sr=asr, target_sr=sr)
            n = max(len(v), len(a))
            mix = np.zeros(n, dtype=np.float32)
            mix[: len(v)] += v
            mix[: len(a)] += a * (10 ** (accomp_db / 20))
            peak = np.abs(mix).max()
            if peak > 0.95:
                mix *= 0.95 / peak
        else:
            mix = v
        out = OUT / voice / f"sing_{time.strftime('%Y%m%d-%H%M%S')}.wav"
        sf.write(out, mix, sr)
        conv.unlink(missing_ok=True)
    return out, f"歌声変換（伴奏{'あり' if separate else 'なし'}）/ {time.time() - t0:.1f}秒"


def history(voice: str, limit: int = 40) -> list[Path]:
    """作った音声（保存用の mp3、新しい順）。"""
    d = OUT / voice
    if not d.exists():
        return []
    return sorted(d.glob("*.mp3"), key=lambda f: f.stat().st_mtime, reverse=True)[:limit]


def status_text(voice: str) -> str:
    st = read_status(VOICES / voice)
    if not st:
        return "まだ学習していません"
    lines = [st.get("message", "")]
    if st.get("progress") is not None and st.get("stage") not in ("done", "error"):
        lines.append(f"進み具合: {st['progress'] * 100:.0f}%" + (f"（残り約{st['eta_min']}分）" if st.get("eta_min") else ""))
    if st.get("error"):
        lines.append("⚠ " + st["error"])
    if st.get("minutes"):
        lines.append(f"学習データ: {st.get('clips')} クリップ / {st['minutes']}分")
    lines.append(f"更新: {st.get('updated', '')}")
    return "\n".join(l for l in lines if l)
