"""Voice Studio の共通設定（パス・状態ファイル・子プロセス起動）。"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOICES = ROOT / "voices"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
VC_PY = ROOT / ".venv-vc" / "Scripts" / "python.exe"
SEP_PY = ROOT / ".venv-sep" / "Scripts" / "python.exe"
SEEDVC = ROOT / "third_party" / "seed-vc"
QWEN_FT = ROOT / "third_party" / "Qwen3-TTS" / "finetuning"
# F0（声の高さの動き）を条件にするモデル。元の録音の抑揚をそのまま保てる
VC_CONFIG = "configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
SPEAKER = "target"

VIDEO_EXT = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v", ".ts")
AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".mp4", ".webm")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def safe_name(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|\s]+', "_", name.strip())
    return name or "voice"


def voice_dir(name: str) -> Path:
    d = VOICES / safe_name(name)
    (d / "raw").mkdir(parents=True, exist_ok=True)
    return d


def list_voices() -> list[str]:
    return sorted(p.name for p in VOICES.iterdir() if p.is_dir()) if VOICES.exists() else []


def read_status(vdir: Path) -> dict:
    try:
        return json.loads((vdir / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_status(vdir: Path, **kw) -> None:
    st = read_status(vdir)
    st.update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    tmp = vdir / "status.json.tmp"
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    # 画面が status.json を読んでいる瞬間だと Windows では置き換えに失敗する（アクセス拒否）。
    # 少し待って何度かやり直し、それでもだめなら今回の更新はあきらめる（学習は止めない）
    for _ in range(20):
        try:
            tmp.replace(vdir / "status.json")
            return
        except PermissionError:
            time.sleep(0.1)


def child_env() -> dict:
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_SYMLINKS_WARNING="1",
               TQDM_DISABLE="1", PYTHONUNBUFFERED="1")  # 進捗バーでログが埋まらないように
    # winget で入れた ffmpeg の場所（新しいシェルでないと PATH に載らないため）
    links = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links"
    env["PATH"] = f"{links};{env.get('PATH', '')}"
    return env


def ffmpeg() -> str:
    from shutil import which

    exe = which("ffmpeg", path=child_env()["PATH"])
    if not exe:  # winget のエイリアスが作られない環境向け: パッケージの実体を探す
        pkgs = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
        found = sorted(pkgs.glob("Gyan.FFmpeg*/*/bin/ffmpeg.exe"))
        exe = str(found[-1]) if found else None
    if not exe:
        raise RuntimeError("ffmpeg が見つかりません（winget install Gyan.FFmpeg）")
    return exe


def run_logged(cmd: list, log: Path, cwd: Path | None = None, on_line=None) -> int:
    """子プロセスを実行し、出力をログに追記する。on_line で 1 行ずつ受け取れる。"""
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))}\n")
        f.flush()
        p = subprocess.Popen(
            [str(c) for c in cmd], cwd=cwd, env=child_env(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        )
        for line in p.stdout:
            f.write(line)
            f.flush()
            if on_line:
                on_line(line.rstrip())
        return p.wait()
