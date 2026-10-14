"""話者モデル（声のプリセット）の管理: 名前・アイコン・学習状況。

1 人分は voices/<id>/ にまとめて保存し、そのまま再学習に回せる:
  meta.json   表示名・作成日・素材の出どころなど
  icon.png    アイコン（256x256）
  raw/        学習の素材（抽出した声・録音）。追加して再学習できる
  tts/        文章読み上げの学習済みモデル（best.txt が使うもの）
  vc/         声変換の学習済みモデル（ft_model.pth と設定）
  ref_*.wav   参照音声
"""

import json
import re
import shutil
import time
from pathlib import Path

from studio.common import AUDIO_EXT, VC_CONFIG, VOICES, SEEDVC, read_status

ICON_SIZE = 256
COLORS = ["#e8590c", "#1971c2", "#2f9e44", "#9c36b5", "#c2255c", "#0c8599", "#e67700", "#5f3dc4"]


def meta(sid: str) -> dict:
    f = VOICES / sid / "meta.json"
    try:
        m = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        m = {}
    m.setdefault("name", sid)
    return m


def save_meta(sid: str, **kw) -> None:
    m = meta(sid)
    m.update(kw, updated=time.strftime("%Y-%m-%d %H:%M"))
    (VOICES / sid / "meta.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


def vc_model(sid: str) -> tuple[Path, Path] | None:
    """声変換の学習済みモデル（チェックポイント, 設定）。話者フォルダになければ学習の作業場所を見る。"""
    for d in (VOICES / sid / "vc", SEEDVC / "runs" / sid):
        ck, cfg = d / "ft_model.pth", d / Path(VC_CONFIG).name
        if ck.exists() and cfg.exists():
            return ck, cfg
    return None


def keep_vc_model(sid: str) -> None:
    """学習が終わった声変換モデルを話者フォルダにコピーしておく（話者ごとにまとめて保存するため）。"""
    src = SEEDVC / "runs" / sid
    dst = VOICES / sid / "vc"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("ft_model.pth", Path(VC_CONFIG).name):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)


def info(sid: str) -> dict:
    vdir = VOICES / sid
    raws = [p for p in (vdir / "raw").glob("*") if p.suffix.lower() in AUDIO_EXT]
    st = read_status(vdir)
    tts_ok = (vdir / "tts" / "best.txt").exists() or any((vdir / "tts").glob("epoch-*"))
    return {
        "id": sid, "name": meta(sid)["name"], "icon": str(icon(sid)),
        "vc": vc_model(sid) is not None, "tts": tts_ok, "raw_files": len(raws),
        "minutes": st.get("minutes"), "training": st.get("stage") not in (None, "done", "error", "stopped", "prep_done"),
    }


def list_ids() -> list[str]:
    if not VOICES.exists():
        return []
    ids = [p.name for p in VOICES.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))]
    # 学習済みを先に、あとは名前順
    return sorted(ids, key=lambda s: (vc_model(s) is None, meta(s)["name"]))


def new_id(name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_-]+", "", name)[:24] or "spk" + time.strftime("%y%m%d%H%M%S")
    sid, i = base, 2
    while (VOICES / sid).exists():
        sid, i = f"{base}_{i}", i + 1
    return sid


def create(name: str, icon_file: str | None = None) -> str:
    sid = new_id(name)
    (VOICES / sid / "raw").mkdir(parents=True)
    save_meta(sid, name=name.strip(), created=time.strftime("%Y-%m-%d %H:%M"))
    if icon_file:
        set_icon(sid, icon_file)
    return sid


def set_icon(sid: str, image_file: str) -> None:
    """画像を中央で正方形に切り抜いて 256x256 で保存する。"""
    from PIL import Image, ImageOps

    im = ImageOps.exif_transpose(Image.open(image_file)).convert("RGB")
    im = ImageOps.fit(im, (ICON_SIZE, ICON_SIZE), Image.LANCZOS)
    im.save(VOICES / sid / "icon.png")


def _font(size):
    from PIL import ImageFont

    for f in ("C:/Windows/Fonts/YuGothB.ttc", "C:/Windows/Fonts/meiryob.ttc", "C:/Windows/Fonts/meiryo.ttc"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def icon(sid: str) -> Path:
    """アイコン。未設定なら、名前の頭文字の入った丸いアイコンを作る。"""
    f = VOICES / sid / "icon.png"
    if f.exists():
        return f
    from PIL import Image, ImageDraw

    name = meta(sid)["name"]
    color = COLORS[sum(map(ord, sid)) % len(COLORS)]
    im = Image.new("RGB", (ICON_SIZE, ICON_SIZE), "#1f1f23")
    d = ImageDraw.Draw(im)
    d.ellipse([16, 16, ICON_SIZE - 16, ICON_SIZE - 16], fill=color)
    ch = name[:1].upper()
    font = _font(120)
    box = d.textbbox((0, 0), ch, font=font)
    d.text(((ICON_SIZE - box[2] - box[0]) / 2, (ICON_SIZE - box[3] - box[1]) / 2), ch, fill="white", font=font)
    auto = VOICES / sid / "icon_auto.png"
    im.save(auto)
    return auto


def plus_icon() -> Path:
    """「＋ 新しい話者を追加」のカード用。"""
    f = VOICES / "_plus.png"
    if not f.exists():
        from PIL import Image, ImageDraw

        VOICES.mkdir(exist_ok=True)
        im = Image.new("RGB", (ICON_SIZE, ICON_SIZE), "#1f1f23")
        d = ImageDraw.Draw(im)
        d.ellipse([16, 16, ICON_SIZE - 16, ICON_SIZE - 16], outline="#888", width=6)
        d.rectangle([118, 68, 138, 188], fill="#bbb")
        d.rectangle([68, 118, 188, 138], fill="#bbb")
        im.save(f)
    return f


def delete(sid: str) -> None:
    """話者を消す（素材・学習済みモデルごと）。取り消し用に _trash へ移すだけにする。"""
    trash = VOICES / "_trash"
    trash.mkdir(exist_ok=True)
    shutil.move(str(VOICES / sid), str(trash / f"{sid}_{time.strftime('%Y%m%d-%H%M%S')}"))
