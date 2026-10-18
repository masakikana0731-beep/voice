"""人の声として自然に聞こえるようにする処理（読み上げ用）。

研究で効果が確かめられているものから順に入れている:
  - 間: 句読点ごとに間の長さを変え、少しゆらぎを入れる。話す速さも変えられる
  - 息継ぎ: 話者の素材から取り出した本物の息の音を、文の頭に入れる
  - 話し言葉モード: 「えっと」などを時々入れる（逆効果のこともあるので、初期設定はオフ）
  - 文末の脱力: 文の終わりを少しかすれさせる（試験的。効果は小さめ）

設定は話者ごとに voices/<id>/natural.json（読み上げで使った設定・聞き比べの結果で変わる）。
息継ぎ・話し言葉・文末の脱力は、本人の素材や話し方の癖でより本人らしくするので、同意の記録がある話者だけ使える。
"""

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from studio.common import VOICES, ffmpeg

DEFAULTS = {"pauses": True, "speed": 1.0, "breath": 0, "filler": 0, "creak": 0}
OFF = {"pauses": False, "speed": 1.0, "breath": 0, "filler": 0, "creak": 0}  # 以前と同じ読み上げ
CONSENT_KEYS = {"breath": "息継ぎ", "filler": "話し言葉モード", "creak": "文末の脱力"}

# 間の長さ（秒）。読点は短く、句点・疑問は長め、改行（段落）はさらに長く
PAUSE = {"、": 0.28, "。": 0.55, "！": 0.5, "？": 0.6, "\n": 0.9, "": 0.45}
FILLERS = ["えっと、", "あの、", "えー、", "うーん、", "まあ、"]


# ---------------------------------------------------------------- 設定

def settings(voice: str) -> dict:
    try:
        st = json.loads((VOICES / voice / "natural.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st = {}
    return {**DEFAULTS, **{k: v for k, v in st.items() if k in DEFAULTS}}


def save_settings(voice: str, **kw) -> dict:
    st = {**settings(voice), **{k: v for k, v in kw.items() if k in DEFAULTS}}
    (VOICES / voice / "natural.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    return st


def needs_consent(nat: dict) -> list[str]:
    """この設定で、同意の記録が要る機能の名前。"""
    return [name for k, name in CONSENT_KEYS.items() if nat.get(k)]


# ---------------------------------------------------------------- 文章の側（区切り・言いよどみ）

def split_sentences(text: str, max_chars: int = 60) -> list[tuple[str, str]]:
    """文章をかたまりに区切り、[(かたまり, 終わりの記号)] を返す。終わりの記号で後ろの間の長さが決まる。

    1 文ずつ読ませると、モデルが 1 文ずつ丁寧に読んで半分ほどの速さになり不自然だった（実測 7.7 秒 → 16.1 秒）。
    そのため、続いている文は max_chars までまとめて読ませる（文の間の間はモデルに任せる）。
    段落（改行）はまたがず、長すぎる文は読点で切る（記号は「、」）。"""
    items = []
    for m in re.finditer(r"([^。！？!?\n]*[。！？!?]+)|([^。！？!?\n]+)|(\n+)", text):
        if m.group(3):
            if items:
                items[-1][1] = "\n"
            continue
        s = m.group(0).strip()
        if not s:
            continue
        end = {"!": "！", "?": "？"}.get(s[-1], s[-1])
        items.append([s, end if end in "。！？" else ""])
    cut_items = []
    for s, kind in items:
        while len(s) > max_chars:
            cut = s.rfind("、", 0, max_chars)
            cut = cut + 1 if cut > 10 else max_chars
            cut_items.append([s[:cut], "、"])
            s = s[cut:]
        cut_items.append([s, kind])
    merged = []
    for s, kind in cut_items:
        if merged and merged[-1][1] not in ("\n", "、") and len(merged[-1][0]) + len(s) <= max_chars:
            merged[-1] = [merged[-1][0] + s, kind]
        else:
            merged.append([s, kind])
    return [(s, k) for s, k in merged] or [(text, "")]


def add_fillers(items, amount: float, rng) -> list[tuple[str, str]]:
    """文の頭に「えっと、」などを時々入れる。amount 0〜100（100 で 2 文に 1 回ほど）。続けては入れない。"""
    if amount <= 0:
        return items
    p, out, last = 0.5 * amount / 100, [], -9
    for i, (s, k) in enumerate(items):
        if i - last >= 2 and rng.random() < p and not any(s.startswith(f[:2]) for f in FILLERS):
            s = FILLERS[rng.integers(len(FILLERS))] + s
            last = i
        out.append((s, k))
    return out


# ---------------------------------------------------------------- 音の側（間・速さ）

def trim(w, sr, top_db: float = 40, keep: float = 0.05):
    """前後の無音を切る（間の長さをこちらで決めるため）。語尾が消えないよう少し残す。"""
    import librosa

    if len(w) < sr * 0.1:
        return w
    _, (s, e) = librosa.effects.trim(w, top_db=top_db, frame_length=1024, hop_length=256)
    return w[max(0, s - int(keep * sr)): min(len(w), e + int(keep * sr))]


def stretch(y, sr, speed: float):
    """話す速さを変える（声の高さは変えない）。ffmpeg の atempo を使う。"""
    import soundfile as sf

    if abs(speed - 1.0) < 0.01:
        return y
    with tempfile.TemporaryDirectory() as t:
        a, b = Path(t) / "a.wav", Path(t) / "b.wav"
        sf.write(a, y, sr)
        subprocess.run([ffmpeg(), "-v", "error", "-y", "-i", str(a), "-filter:a", f"atempo={speed:.3f}", str(b)],
                       check=True)
        z, _ = sf.read(b, dtype="float32")
    return z


def assemble(wavs, kinds, sr, nat: dict, rng):
    """文ごとの音声を、間を入れてつなぐ。pauses がオンなら句読点に合わせた間（少しゆらぎあり）、
    オフなら以前と同じ 0.35 秒の決まった間。"""
    parts = []
    for i, (w, k) in enumerate(zip(wavs, kinds)):
        w = np.asarray(w, dtype=np.float32)
        parts.append(trim(w, sr) if nat.get("pauses") else w)
        if i < len(wavs) - 1:
            sec = PAUSE.get(k, 0.45) * rng.uniform(0.85, 1.2) if nat.get("pauses") else 0.35
            parts.append(np.zeros(int(sec * sr), dtype=np.float32))
    if nat.get("pauses"):  # 頭と終わりに少し余白（頭の息継ぎを入れる場所にもなる）
        parts = [np.zeros(int(0.15 * sr), dtype=np.float32)] + parts + [np.zeros(int(0.3 * sr), dtype=np.float32)]
    return stretch(np.concatenate(parts), sr, float(nat.get("speed", 1.0)))


# ---------------------------------------------------------------- 息継ぎ

def _runs(mask):
    """True が続く区間 [(始め, 終わり)]。"""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _find_breaths(y, sr):
    """話し始める直前の、声帯が鳴っていない「スーッ」という音（吸う息）を探す。"""
    import librosa
    from scipy.ndimage import median_filter

    frame, hop = 1024, 256
    S = np.abs(librosa.stft(y, n_fft=frame, hop_length=hop))
    f = librosa.fft_frequencies(sr=sr, n_fft=frame)
    tot = S.sum(axis=0) + 1e-9
    low = S[(f >= 70) & (f < 400)].sum(axis=0) / tot      # 声の高さの成分（声帯が鳴っていると大きい）
    mid = S[(f >= 800) & (f < 7000)].sum(axis=0) / tot    # 息の「スーッ」の成分
    db = 20 * np.log10(librosa.feature.rms(S=S, frame_length=frame)[0] + 1e-9)
    loud = db[db > db.max() - 40]
    speech_db = float(np.median(loud)) if len(loud) else -30.0
    voiced = median_filter(((db > speech_db - 20) & (low > 0.2)).astype(np.uint8), size=5).astype(bool)
    breathy = (db > speech_db - 45) & (db < speech_db - 6) & (mid > 0.45) & (low < 0.2) & ~voiced
    found = []
    segs = _runs(voiced)
    for (_, e0), (s1, e1) in zip(segs[:-1], segs[1:]):
        if (s1 - e0) * hop / sr < 0.3:  # 息を吸うには短すぎる間
            continue
        a = max(e0 + int(0.08 * sr / hop), s1 - int(1.0 * sr / hop))
        runs = [(a + i, a + j) for i, j in _runs(breathy[a:s1])]
        if not runs:
            continue
        i, j = max(runs, key=lambda r: r[1] - r[0])
        dur, lead = (j - i) * hop / sr, (s1 - j) * hop / sr
        # すぐ後に声が続くもの（lead が小さい）は「す」「は」などの子音のことが多いので除く
        if not (0.15 <= dur <= 0.9 and 0.03 <= lead <= 0.35):
            continue
        seg = y[i * hop: j * hop + frame]
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr)))
        next_db = float(np.median(db[s1:e1])) if e1 > s1 else speech_db
        rel = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9) - next_db
        # 高い音が強すぎる（「し」「す」の擦れる音）・大きすぎるものも除く
        if centroid > 4000 or not (-30 <= rel <= -11):
            continue
        found.append({"seg": seg, "dur": round(float(dur), 3), "rel_db": round(float(rel), 1),
                      "lead": round(float(lead), 3), "centroid": int(centroid)})
    return found


def extract_breaths(voice: str, max_n: int = 40) -> int:
    """話者の素材（raw/）から息の音を取り出して voices/<id>/breaths/ に保存する。見つかった数を返す。"""
    import soundfile as sf

    vdir = VOICES / voice
    found = []
    for f in sorted((vdir / "raw").glob("*.wav")):
        y, sr = sf.read(f, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != 44100:
            import librosa

            y, sr = librosa.resample(y, orig_sr=sr, target_sr=44100), 44100
        found += _find_breaths(y, sr)
    # 長さ 0.25〜0.6 秒・声より 15〜25dB 小さいくらいの、ふつうの息を優先する
    found.sort(key=lambda b: abs(b["dur"] - 0.4) + abs(b["rel_db"] + 20) / 15)
    out = vdir / "breaths"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    index, preview = [], []
    for n, b in enumerate(found[:max_n]):
        seg = b.pop("seg")
        fade = min(int(0.02 * 44100), len(seg) // 4)
        seg[:fade] *= np.linspace(0, 1, fade)
        seg[-fade:] *= np.linspace(1, 0, fade)
        sf.write(out / f"b{n:02d}.wav", seg, 44100)
        index.append({"file": f"b{n:02d}.wav", **b})
        if n < 12:
            preview += [seg / (np.abs(seg).max() + 1e-9) * 0.3, np.zeros(int(0.5 * 44100), dtype=np.float32)]
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    if preview:
        sf.write(out / "preview.wav", np.concatenate(preview), 44100)
    return len(index)


def breath_info(voice: str) -> tuple[int, Path | None]:
    d = VOICES / voice / "breaths"
    n = len(list(d.glob("b*.wav"))) if d.exists() else 0
    return n, (d / "preview.wav" if (d / "preview.wav").exists() else None)


def _load_breaths(voice: str, sr: int):
    import soundfile as sf

    d = VOICES / voice / "breaths"
    try:
        index = json.loads((d / "index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for b in index:
        y, bsr = sf.read(d / b["file"], dtype="float32")
        if bsr != sr:
            import librosa

            y = librosa.resample(y, orig_sr=bsr, target_sr=sr)
        out.append((y, float(b["rel_db"])))
    return out


def add_breaths(y, sr, voice: str, amount: float, rng):
    """文の頭（長めの間のあと）に本物の息の音を入れる。間が足りなければ少し広げる。
    amount 0〜100: 入れる割合（最初の文には必ず入れる）。"""
    import librosa

    breaths = _load_breaths(voice, sr)
    if amount <= 0 or not breaths:
        return y
    iv = librosa.effects.split(y, top_db=35, frame_length=1024, hop_length=256)
    if len(iv) == 0:
        return y
    speech_rms = np.sqrt(np.mean(np.concatenate([y[s:e] for s, e in iv]) ** 2) + 1e-12)
    p = 0.3 + 0.6 * amount / 100
    parts, adds, cur, total = [], [], 0, 0
    for k, (s, e) in enumerate(iv):
        prev_end = iv[k - 1][1] if k else 0
        if k and (s - prev_end) < 0.3 * sr:
            continue
        if k and rng.random() >= p:
            continue
        b, rel = breaths[rng.integers(len(breaths))]
        b = b * (speech_rms * 10 ** (np.clip(rel, -28, -12) / 20) / (np.sqrt(np.mean(b ** 2)) + 1e-9))
        lead = int(rng.uniform(0.04, 0.12) * sr)  # 息を吸い終わってから話し始めるまで
        room = (s - prev_end - int(0.1 * sr)) if k else s
        extra = max(0, len(b) + lead + int(0.05 * sr) - room)
        parts.append(y[cur:s])
        total += s - cur
        parts.append(np.zeros(extra, dtype=np.float32))
        total += extra
        cur = s
        adds.append((total - lead - len(b), b))
    parts.append(y[cur:])
    z = np.concatenate(parts).astype(np.float32)
    for st, b in adds:
        if st < 0:
            b, st = b[-st:], 0
        z[st:st + len(b)] += b[: len(z) - st]
    return z


# ---------------------------------------------------------------- 文末の脱力（かすれ）

def add_creak(y, sr, amount: float, rng):
    """長めの間の前（文末）の最後 0.12〜0.22 秒を、不規則に震わせて少し弱める（きしみ声に近づける）。試験的。"""
    import librosa

    if amount <= 0:
        return y
    iv = librosa.effects.split(y, top_db=35, frame_length=1024, hop_length=256)
    z, amt = y.copy(), amount / 100
    for k, (s, e) in enumerate(iv):
        nxt = iv[k + 1][0] if k + 1 < len(iv) else len(y)
        if nxt - e < 0.3 * sr:
            continue
        a = max(s, e - int(rng.uniform(0.12, 0.22) * sr))
        n = e - a
        env, t = np.zeros(n, dtype=np.float32), 0
        while t < n:  # 25〜55Hz の不規則な脈（きしみ声の周期）
            period = max(8, int(sr / rng.uniform(25, 55)))
            m = min(period, n - t)
            env[t:t + m] = np.exp(-np.arange(m) / (period * 0.3))
            t += period
        ramp = np.linspace(0, 1, n, dtype=np.float32) ** 1.5  # 終わりに向かって強く
        z[a:e] *= (1 - amt * 0.8 * ramp * (1 - env)) * (1 - 0.35 * amt * ramp)
    return z


def post(y, sr, voice: str, nat: dict, rng):
    """声変換のあとにかける分（文末の脱力 → 息継ぎ）。声変換の前に入れると息が声に変えられてしまうため。"""
    y = add_creak(y, sr, float(nat.get("creak", 0)), rng)
    return add_breaths(y, sr, voice, float(nat.get("breath", 0)), rng)
