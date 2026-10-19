"""精度検証: 無料の公開コーパス（JSUT）で、学習の設定やプログラムの変更で精度がどこまで上がるかを自動で調べる。

  python -m studio.bench run       # 準備から全部の実験まで順に実行（止めても、次回は続きから）
  python -m studio.bench report    # bench/report.md と bench/report.png を作り直す
  python -m studio.bench rescore   # 保存してある音声を全部、今の採点役で採点し直す（採点の方法を変えたとき）
  python -m studio.bench adopt     # 一番良かった設定を学習の既定（studio/train_defaults.json）にする

実験の流れ（前の結果を見て次を決める）:
  0. 基準: 本物の録音（= 100 点）、学習なし（参照音声からのクローン・汎用の声変換）
  1. 読み上げ: 学習率 5 通り（素材 24 分・4 周、周ごとに採点）。一番良いのが最後の周なら 8 周でも試す
  2. 読み上げ: プログラムの変更 4 通りを、一番良い学習率で 1 つずつ試す
     （ウォームアップ＋コサイン / 話者の特徴を全クリップの平均に / サブ損失の重み 1.0 / まとめる量 2 倍）
  3. 読み上げ: 良かった変更を全部入れたもの（2 つ以上あれば）
  4. 読み上げ: 素材の量 10 / 60 / 120 分（一番良い設定で）
  5. 声変換: 学習率 3 通り（24 分・4000 ステップ、1000 ごとに採点）
  6. 声変換: 素材の量 60 / 120 分（一番良い学習率で）

採点（studio/quality.py）: 読み間違い（Whisper）・似ている度（ECAPA）・自然さ（UTMOSv2）→ 本物を 100 とした総合点。
結果は bench/results.jsonl（1 行 = 1 つのモデル）。
JSUT（https://sites.google.com/site/shinnosuketakamichi/publication/jsut）は女性 1 名・約 10 時間の読み上げ音声。
研究目的なら無料で、商用利用は不可。ここで作るモデルは検証専用で、採点が終わったら消す。
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

from studio.common import PY, ROOT, SEEDVC, VC_CONFIG, VC_PY, VOICES, child_env, run_logged

BENCH = ROOT / "bench"
RESULTS = BENCH / "results.jsonl"
STATE = BENCH / "state.json"
TEST = BENCH / "test"
RUNS = BENCH / "runs"
LOGS = BENCH / "logs"
CORPUS_ZIP = ROOT / "corpora" / "jsut_ver1.1.zip"
CORPUS_URL = "http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip"
PREP = ROOT / "corpora" / "jsut_prep"

SIZES = [10, 24, 60, 120]          # 素材の量（分）
N_TEST, N_PRINT, N_VC = 12, 30, 8  # 採点用の文の数・本物の声の特徴に使う数・声変換の元音声の数
TTS_EPOCHS = 4
LRS = [2e-6, 5e-6, 1e-5, 2e-5, 5e-5]
P0 = {"lr": 2e-6, "schedule": "const", "spk_avg": 0, "sub_weight": 0.3, "grad_accum": 4}
VARIANTS = {  # 名前: (変更, 説明)
    "cosine": ({"schedule": "cosine"}, "学習率をウォームアップ＋コサインで変える"),
    "spk_avg": ({"spk_avg": 1}, "話者の特徴を全クリップの平均にする"),
    "sub1": ({"sub_weight": 1.0}, "細かい音（サブ）の損失の重みを 0.3 → 1.0"),
    "ga8": ({"grad_accum": 8}, "まとめる量を 2 倍（1 回の更新に 16 文）"),
}
VC_LRS = [1e-5, 3e-5, 1e-4]
VC_STEPS, VC_SAVE = 4000, 1000


# ---------------------------------------------------------------- 状態・結果

def state(**kw) -> dict:
    BENCH.mkdir(exist_ok=True)
    try:
        st = json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st = {}
    if kw:
        st.update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
        STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    return st


def results() -> list[dict]:
    """結果の一覧。総合点は、保存した値ではなく、今の計算方法（quality.total）で毎回計算し直す。"""
    from studio import quality

    try:
        rows = [json.loads(l) for l in RESULTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    except OSError:
        return []
    base = next((r for r in rows if r["kind"] == "real"), None)
    for r in rows:
        if base and r.get("sim") is not None and r.get("cer") is not None:
            r["total"] = quality.total(r, base)
            # 似せ度: 声の似ている度だけを、別人 = 0%・本物 = 100% に直したもの
            floor = base.get("sim_other", 0.35)
            r["like"] = round(100 * max(0.0, min(1.2, (r["sim"] - floor) / (base["sim"] - floor))), 1)
    return rows


def add_result(row: dict) -> None:
    BENCH.mkdir(exist_ok=True)
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps({**row, "time": time.strftime("%Y-%m-%d %H:%M")}, ensure_ascii=False) + "\n")


def base_row() -> dict | None:
    return next((r for r in results() if r["kind"] == "real"), None)


def rows_of(exp: str) -> list[dict]:
    return [r for r in results() if r.get("exp") == exp and r.get("total") is not None]


def best_row(exps) -> dict | None:
    rows = [r for e in exps for r in rows_of(e)]
    return max(rows, key=lambda r: r["total"]) if rows else None


def done(exp: str) -> bool:
    return any(r.get("exp") == exp for r in results())


def log(msg: str) -> None:
    print(f"[bench] {msg}", flush=True)
    state(message=msg)


# ---------------------------------------------------------------- 準備

def download() -> None:
    if CORPUS_ZIP.exists() and CORPUS_ZIP.stat().st_size > 2_600_000_000:
        return
    log("JSUT コーパスをダウンロード中（2.7GB）")
    CORPUS_ZIP.parent.mkdir(exist_ok=True)
    subprocess.run(["curl", "-sL", "--retry", "10", "-C", "-", "-o", str(CORPUS_ZIP), CORPUS_URL], check=True)


def prepare() -> None:
    """JSUT を学習用（24kHz / 44.1kHz）に整え、データ量ごとの「話者」（voices/_bench_d*）と採点用の文を作る。"""
    import zipfile

    import librosa
    import numpy as np
    import soundfile as sf

    if (TEST / "test.json").exists():
        return
    download()
    log("JSUT を展開・整形中")
    with zipfile.ZipFile(CORPUS_ZIP) as z:
        names = z.namelist()
        trans = next(n for n in names if n.endswith("basic5000/transcript_utf8.txt"))
        top = trans[: -len("basic5000/transcript_utf8.txt")]
        lines = z.read(trans).decode("utf-8").splitlines()
        texts = dict(l.split(":", 1) for l in lines if ":" in l)
        ids = sorted(texts)
        random.Random(0).shuffle(ids)
        test_ids, print_ids = ids[:N_TEST], ids[N_TEST:N_TEST + N_PRINT]
        vc_ids, pool = ids[N_TEST + N_PRINT:N_TEST + N_PRINT + N_VC], ids[N_TEST + N_PRINT + N_VC:]
        for d in ("c24", "c44"):
            (PREP / d).mkdir(parents=True, exist_ok=True)

        def convert(i):
            """1 文を 24kHz と 44.1kHz に（前後の無音は 0.15 秒だけ残す）。長さ（秒）を返す。"""
            f24, f44 = PREP / "c24" / f"{i}.wav", PREP / "c44" / f"{i}.wav"
            if f24.exists() and f44.exists():
                return sf.info(f24).duration
            import io

            y, sr = sf.read(io.BytesIO(z.read(f"{top}basic5000/wav/{i}.wav")), dtype="float32")
            _, (s, e) = librosa.effects.trim(y, top_db=40)
            y = y[max(0, s - int(0.15 * sr)): e + int(0.15 * sr)]
            sf.write(f44, librosa.resample(y, orig_sr=sr, target_sr=44100), 44100, subtype="PCM_16")
            sf.write(f24, librosa.resample(y, orig_sr=sr, target_sr=24000), 24000, subtype="PCM_16")
            return len(y) / sr

        for n in names:  # ライセンスの説明も一緒に置いておく
            if n.startswith(top) and n.count("/") == 1 and n.lower().endswith((".txt", ".md")):
                (ROOT / "corpora" / Path(n).name).write_bytes(z.read(n))

        # 学習用: 先頭から順に足して、それぞれの量に達するまで（10 分 ⊂ 24 分 ⊂ 60 分 ⊂ 120 分）
        used, total = [], 0.0
        for i in pool:
            if total >= max(SIZES) * 60:
                break
            total += convert(i)
            used.append((i, total))
        for size in SIZES:
            vdir = VOICES / f"_bench_d{size}"
            shutil.rmtree(vdir, ignore_errors=True)
            for d in ("clips", "clips_hq", "raw"):
                (vdir / d).mkdir(parents=True)
            chosen = [i for i, t in used if t <= size * 60 + 5]
            with open(vdir / "clips.jsonl", "w", encoding="utf-8") as f:
                for i in chosen:
                    os.link(PREP / "c24" / f"{i}.wav", vdir / "clips" / f"{i}.wav")
                    os.link(PREP / "c44" / f"{i}.wav", vdir / "clips_hq" / f"{i}.wav")
                    f.write(json.dumps({"audio": str(vdir / "clips" / f"{i}.wav"), "text": texts[i],
                                        "ref_audio": str(vdir / "ref.wav")}, ensure_ascii=False) + "\n")
            ref = min(chosen, key=lambda i: abs(sf.info(PREP / "c24" / f"{i}.wav").duration - 8))
            shutil.copy(PREP / "c24" / f"{ref}.wav", vdir / "ref.wav")
            shutil.copy(PREP / "c44" / f"{ref}.wav", vdir / "ref_hq.wav")
            (vdir / "ref.txt").write_text(texts[ref], encoding="utf-8")
            (vdir / "meta.json").write_text(json.dumps({"name": f"検証用 JSUT {size}分"}, ensure_ascii=False),
                                            encoding="utf-8")
            log(f"素材 {size} 分: {len(chosen)} 文")

        # 採点用: 本物の録音（読み上げの正解）・本物の声の特徴・声変換の元にする文
        for d in ("real", "print", "vc_src"):
            shutil.rmtree(TEST / d, ignore_errors=True)
            (TEST / d).mkdir(parents=True)
        tts = []
        for k, i in enumerate(test_ids):
            sec = convert(i)
            shutil.copy(PREP / "c24" / f"{i}.wav", TEST / "real" / f"{k:03d}.wav")
            tts.append({"name": f"{k:03d}.wav", "text": texts[i], "sec": round(sec, 3)})
        for i in print_ids:
            convert(i)
            shutil.copy(PREP / "c44" / f"{i}.wav", TEST / "print" / f"{i}.wav")
        vc = [{"name": f"{k:03d}.wav", "text": texts[i]} for k, i in enumerate(vc_ids)]

    # 声変換の元: 内蔵の別の声（Ono_Anna）で読ませる（実在の人の録音を使わないため）
    log("声変換の元にする音声を作成中")
    from studio import engine
    from tts import fetch_model

    model = engine._load(("custom",), fetch_model("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"))
    import torch

    for k, v in enumerate(vc):
        torch.manual_seed(k)
        w, sr = model.generate_custom_voice(text=v["text"], language="Japanese", speaker="Ono_Anna",
                                            max_new_tokens=engine._max_tokens(v["text"]))
        sf.write(TEST / "vc_src" / v["name"], librosa.resample(np.asarray(w[0], dtype="float32"), orig_sr=sr,
                                                               target_sr=44100), 44100)
    engine.unload()
    (TEST / "test.json").write_text(json.dumps({"tts": tts, "vc": vc}, ensure_ascii=False, indent=1),
                                    encoding="utf-8")


# ---------------------------------------------------------------- 採点（別プロセスで呼ぶ）

def _speech_sec(path) -> float:
    import librosa
    import soundfile as sf

    y, sr = sf.read(path, dtype="float32")
    if len(y) < sr * 0.1:
        return len(y) / sr
    _, (s, e) = librosa.effects.trim(y, top_db=40)
    return (e - s) / sr


def score_sets(sets: dict, texts: dict, base: dict | None) -> dict:
    """sets = {ラベル: [wav...]}、texts = {wav の名前: 正解の文}。ラベルごとの平均点を返す。"""
    from studio import quality

    allf = [f for fs in sets.values() for f in fs]
    hyps = dict(zip(map(str, allf), quality.transcribe(allf)))
    quality.unload_asr()
    j = quality.judge(allf, sorted((TEST / "print").glob("*.wav")))
    out = {}
    for label, fs in sets.items():
        rows = [{"cer": quality.cer(texts[Path(f).name.split("__")[-1]], hyps[str(f)]),
                 **j[Path(f).name]} for f in fs]
        m = {k: quality.mean(rows, k) for k in ("cer", "sim", "mos")}
        m["total"] = quality.total(m, base) if base else 100.0
        m["sample"] = hyps[str(fs[0])][:40]
        out[label] = m
    return out


def eval_tts(eid: str, voice: str, sub: str | None, meta: dict) -> None:
    """読み上げモデル（sub 以下の各周、None なら学習なしのクローン）で採点用の文を読ませて採点する。"""
    import soundfile as sf
    import torch

    from studio import engine
    from studio.common import SPEAKER

    test = json.loads((TEST / "test.json").read_text(encoding="utf-8"))["tts"]
    out = RUNS / eid
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    ckpts = sorted((VOICES / voice / sub).glob("epoch-*")) if sub else [None]
    sets = {}
    for ck in ckpts:
        label = ck.name if ck else "zero"
        files = []
        for k, t in enumerate(test):
            torch.manual_seed(1000 + k)  # どのモデルも同じ乱数で読ませる（比べやすくする）
            if ck:
                model = engine._load(("ft", voice, sub, ck.name), ck)
                w, sr = model.generate_custom_voice(text=t["text"], language="Auto", speaker=SPEAKER,
                                                    max_new_tokens=engine._max_tokens(t["text"]))
                w = w[0]
            else:
                (w,), sr, _ = engine._generate(voice, [(t["text"], "")], None)
            f = out / f"{label}__{t['name']}"
            sf.write(f, w, sr)
            files.append(f)
        sets[label] = files
        print(f"[bench] {eid} {label} 生成完了", flush=True)
    engine.unload()
    texts = {t["name"]: t["text"] for t in test}
    real_sec = sum(_speech_sec(TEST / "real" / t["name"]) for t in test)
    for label, m in score_sets(sets, texts, base_row()).items():
        m["dur"] = round(sum(_speech_sec(f) for f in sets[label]) / real_sec, 3)  # 話す長さ（本物 = 1）
        epoch = int(label.split("-")[1]) if label.startswith("epoch") else 0
        add_result({**meta, "exp": eid, "epoch": epoch, **m})
        print(f"[bench] {eid} {label}: {m}", flush=True)


def eval_vc(eid: str, voice: str, ckpts: list, config: str | None, meta: dict) -> None:
    """声変換モデル（[(ラベル, チェックポイント)]、チェックポイント None は汎用モデル）で採点する。"""
    from studio import engine

    test = json.loads((TEST / "test.json").read_text(encoding="utf-8"))["vc"]
    ref = engine.vc_reference(VOICES / voice)
    out = RUNS / eid
    shutil.rmtree(out, ignore_errors=True)
    sets = {}
    for label, ck in ckpts:
        d = out / label
        cmd = [str(VC_PY), str(ROOT / "studio" / "vc_run.py"), str(ROOT / "studio" / "vc_batch.py"),
               "--segments", str(TEST / "vc_src"), "--target", str(ref), "--out", str(d), "--diffusion-steps", "30"]
        if ck:
            cmd += ["--checkpoint", str(ck), "--config", str(config)]
        subprocess.run(cmd, cwd=SEEDVC, env=child_env(), check=True, capture_output=True)
        files = []
        for t in test:  # 名前を「ラベル__元の名前」にそろえる（採点で名前が重ならないように）
            dst = out / f"{label}__{t['name']}"
            shutil.move(str(d / t["name"]), dst)
            files.append(dst)
        shutil.rmtree(d, ignore_errors=True)
        sets[label] = files
        print(f"[bench] {eid} {label} 変換完了", flush=True)
    texts = {t["name"]: t["text"] for t in test}
    for label, m in score_sets(sets, texts, base_row()).items():
        step = int(label[1:]) if label.startswith("s") else 0
        add_result({**meta, "exp": eid, "step": step, **m})
        print(f"[bench] {eid} {label}: {m}", flush=True)


def eval_real() -> None:
    test = json.loads((TEST / "test.json").read_text(encoding="utf-8"))["tts"]
    files = []
    for t in test:
        f = RUNS / "real" / f"real__{t['name']}"
        f.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(TEST / "real" / t["name"], f)
        files.append(f)
    m = score_sets({"real": files}, {t["name"]: t["text"] for t in test}, None)["real"]
    # 別人の声（声変換の元にした内蔵の声 Ono_Anna）の似ている度 = 総合点で 0 点になる下限
    from studio import quality

    other = quality.judge(sorted((TEST / "vc_src").glob("*.wav")), sorted((TEST / "print").glob("*.wav")), mos=False)
    m["sim_other"] = quality.mean(list(other.values()), "sim")
    add_result({"exp": "real", "kind": "real", "stage": "base", "data": 0, **m, "dur": 1.0})


# ---------------------------------------------------------------- 実験

def tts_id(data: int, p: dict, epochs: int) -> str:
    return (f"tts_d{data}_lr{p['lr']:g}_{p['schedule']}_spk{p['spk_avg']}_sub{p['sub_weight']:g}"
            f"_ga{p['grad_accum']}_e{epochs}")


def tts_exp(stage: str, data: int, p: dict, epochs: int = TTS_EPOCHS, note: str = "") -> str:
    eid = tts_id(data, p, epochs)
    if done(eid):
        return eid
    voice, sub = f"_bench_d{data}", f"runs/{eid}"
    lg = LOGS / f"{eid}.log"
    log(f"読み上げ [{stage}] 素材 {data}分 / {note or eid} を学習中")
    shutil.rmtree(VOICES / voice / sub, ignore_errors=True)  # 途中で止めた前回の分
    t0 = time.time()
    cmd = [PY, "-m", "studio.train_tts", voice, "--epochs", epochs, "--keep", epochs, "--out", sub,
           "--lr", p["lr"], "--schedule", p["schedule"], "--spk-avg", p["spk_avg"], "--sub-weight", p["sub_weight"],
           "--grad-accum", p["grad_accum"]]
    meta = {"kind": "tts", "stage": stage, "data": data, "params": p, "note": note}
    if run_logged(cmd, lg) != 0:
        add_result({**meta, "exp": eid, "failed": "学習に失敗", "total": None})
    else:
        meta["train_min"] = round((time.time() - t0) / 60, 1)
        log(f"読み上げ [{stage}] {note or eid} を採点中")
        if run_logged([PY, "-m", "studio.bench", "_eval-tts", eid, voice, sub, json.dumps(meta)], lg) != 0:
            add_result({**meta, "exp": eid, "failed": "採点に失敗", "total": None})
    shutil.rmtree(VOICES / voice / sub, ignore_errors=True)  # 検証用のモデルは採点が終わったら消す（1 周 4GB）
    report()
    return eid


def vc_id(data: int, lr: float, steps: int) -> str:
    return f"vc_d{data}_lr{lr:g}_s{steps}"


def vc_exp(stage: str, data: int, lr: float, steps: int = VC_STEPS) -> str:
    eid = vc_id(data, lr, steps)
    if done(eid):
        return eid
    voice, run = f"_bench_d{data}", f"_bench_{eid}"
    rdir = SEEDVC / "runs" / run
    shutil.rmtree(rdir, ignore_errors=True)
    lg = LOGS / f"{eid}.log"
    log(f"声変換 [{stage}] 素材 {data}分 / 学習率 {lr:g} を学習中（{steps} ステップ）")
    env = {**child_env(), "SEEDVC_LR": str(lr), "SEEDVC_KEEP_ALL": "1"}
    t0 = time.time()
    with open(lg, "a", encoding="utf-8") as f:
        rc = subprocess.run([str(VC_PY), str(ROOT / "studio" / "vc_run.py"), "train.py", "--config", VC_CONFIG,
                             "--dataset-dir", str(VOICES / voice / "clips_hq"), "--run-name", run, "--batch-size", "1",
                             "--max-steps", str(steps), "--max-epochs", "100000", "--save-every", str(VC_SAVE),
                             "--num-workers", "0"], cwd=SEEDVC, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    meta = {"kind": "vc", "stage": stage, "data": data, "params": {"lr": lr}, "train_min": round((time.time() - t0) / 60, 1)}
    if rc != 0 or not (rdir / "ft_model.pth").exists():
        add_result({**meta, "exp": eid, "failed": "学習に失敗", "total": None})
    else:
        ckpts = sorted(rdir.glob("DiT_epoch_*_step_*.pth"), key=lambda p: int(p.stem.split("_")[-1]))
        pairs = [(f"s{int(c.stem.split('_')[-1])}", str(c)) for c in ckpts] + [(f"s{steps}", str(rdir / "ft_model.pth"))]
        cfg = str(next(rdir.glob("config_*.yml")))
        log(f"声変換 [{stage}] 学習率 {lr:g} を採点中")
        if run_logged([PY, "-m", "studio.bench", "_eval-vc", eid, voice, json.dumps(pairs), cfg, json.dumps(meta)],
                      lg) != 0:
            add_result({**meta, "exp": eid, "failed": "採点に失敗", "total": None})
    shutil.rmtree(rdir, ignore_errors=True)
    report()
    return eid


def run_all() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    state(running=True, pid=os.getpid(), started=time.strftime("%Y-%m-%d %H:%M"))
    if not (TEST / "test.json").exists():
        if subprocess.run([str(PY), "-m", "studio.bench", "_prepare"], cwd=ROOT, env=child_env()).returncode != 0:
            raise SystemExit("準備に失敗しました")

    # 0. 基準
    if not done("real"):
        log("基準: 本物の録音を採点中")
        run_logged([PY, "-m", "studio.bench", "_eval-real"], LOGS / "base.log")
    if not done("tts_zero"):
        log("基準: 学習なしの読み上げ（参照音声からのクローン）を採点中")
        run_logged([PY, "-m", "studio.bench", "_eval-tts", "tts_zero", "_bench_d24", "-",
                    json.dumps({"kind": "tts", "stage": "base", "data": 24, "note": "学習なし（クローン）"})],
                   LOGS / "base.log")
    if not done("vc_zero"):
        log("基準: 学習なしの声変換（汎用モデル）を採点中")
        run_logged([PY, "-m", "studio.bench", "_eval-vc", "vc_zero", "_bench_d24", json.dumps([["zero", None]]), "-",
                    json.dumps({"kind": "vc", "stage": "base", "data": 24, "note": "学習なし（汎用モデル）"})],
                   LOGS / "base.log")
    report()

    # 1. 学習率
    s1 = [tts_exp("学習率", 24, {**P0, "lr": lr}, note=f"学習率 {lr:g}") for lr in LRS]
    b1 = best_row(s1)
    if not b1:
        raise SystemExit("学習率の実験がすべて失敗しました")
    p1 = b1["params"]
    if b1["epoch"] == TTS_EPOCHS:  # 最後の周が一番良い → まだ伸びるかもしれないので長く学習してみる
        b1 = best_row(s1 + [tts_exp("学習率", 24, p1, epochs=8, note=f"学習率 {p1['lr']:g}・8 周")])
    base_eid = b1["exp"]
    epochs = int(base_eid.rsplit("_e", 1)[1])

    # 2. プログラムの変更を 1 つずつ
    s2 = {name: tts_exp("変更", 24, {**p1, **ch}, epochs, note=desc) for name, (ch, desc) in VARIANTS.items()}
    winners = [n for n, e in s2.items() if (best_row([e]) or {}).get("total", 0) > b1["total"] + 0.5]

    # 3. 良かった変更を全部
    s3 = []
    if len(winners) >= 2:
        p = dict(p1)
        for n in winners:
            p.update(VARIANTS[n][0])
        s3 = [tts_exp("組み合わせ", 24, p, epochs, note="良かった変更を全部: " + "・".join(winners))]
    bb = best_row([base_eid] + list(s2.values()) + s3)

    # 4. 素材の量
    for d in (10, 60, 120):
        tts_exp("素材の量", d, bb["params"], epochs, note=f"素材 {d} 分")

    # 5. 声変換の学習率 → 6. 素材の量
    v1 = [vc_exp("学習率", 24, lr) for lr in VC_LRS]
    vb = best_row(v1)
    if vb:
        for d in (60, 120):
            vc_exp("素材の量", d, vb["params"]["lr"])
    try:
        rescore()  # 全部を同じ物差しで採点し直してから、まとめる
    except Exception as e:  # 失敗しても、それまでの結果はそのまま使える
        print(f"[bench] 採点し直しに失敗しました: {e}", flush=True)
    log("すべての実験が終わりました")
    state(running=False, finished=time.strftime("%Y-%m-%d %H:%M"))


# ---------------------------------------------------------------- まとめ

def _fmt(r, keys=("total", "like", "sim", "mos", "cer")) -> str:
    def f(k):
        v = r.get(k)
        if v is None:
            return "-"
        return {"cer": f"{v:.0%}", "total": f"{v:.1f}", "like": f"{v:.0f}%"}.get(k, f"{v:.3f}")
    return " | ".join(f(k) for k in keys)


def summary() -> dict:
    """到達レベルの要約（画面・adopt 用）。"""
    rs = results()
    base = base_row()
    tts24 = [r for r in rs if r["kind"] == "tts" and r.get("data") == 24 and r.get("total") is not None
             and r["stage"] != "base"]
    zero = next((r for r in rs if r.get("exp") == "tts_zero"), None)
    vc24 = [r for r in rs if r["kind"] == "vc" and r.get("data") == 24 and r.get("total") is not None
            and r["stage"] != "base"]
    vzero = next((r for r in rs if r.get("exp") == "vc_zero"), None)
    return {"base": base, "tts_zero": zero, "tts_best": max(tts24, key=lambda r: r["total"]) if tts24 else None,
            "vc_zero": vzero, "vc_best": max(vc24, key=lambda r: r["total"]) if vc24 else None}


def report() -> Path:
    rs = results()
    s = summary()
    BENCH.mkdir(exist_ok=True)
    L = ["# 精度検証の結果（JSUT・女性 1 名の読み上げ音声）", "",
         f"更新: {time.strftime('%Y-%m-%d %H:%M')}　状態: {state().get('message', '')}", "",
         "総合点 = 本物の録音を 100 とした点（似ている度 5 割・自然さ 3 割・読み間違いの少なさ 2 割。似ている度は別人の声を 0 とする）。", ""]
    if s["base"]:
        b = s["base"]
        L += [f"本物の録音（100 点）: 似ている度 {b['sim']:.3f} / 自然さ {b['mos']:.2f} / 読み間違い {b['cer']:.0%}（Whisper の誤差）",
              "",
              f"別人の声（似ている度の 0 点）: 似ている度 {b.get('sim_other', 0.35):.3f}", ""]
    L += ["## 到達レベル（素材 24 分）", "", "| | 総合 | 似せ度 | 似ている度 | 自然さ | 読み間違い | 設定 |", "|---|---|---|---|---|---|---|"]
    for k, name in (("tts_zero", "読み上げ・学習なし"), ("tts_best", "読み上げ・一番良い設定"),
                    ("vc_zero", "声変換・学習なし"), ("vc_best", "声変換・一番良い設定")):
        r = s.get(k)
        if r:
            how = r.get("note") or ""
            if k == "tts_best":
                how = f"{r['note']}（{r['epoch']} 周目）"
            if k == "vc_best":
                how = f"学習率 {r['params']['lr']:g}・{r['step']} ステップ"
            L.append(f"| {name} | {_fmt(r)} | {how} |")
    L += ["", "## 読み上げ（周ごと）", "", "| 段階 | 素材 | 内容 | 周 | 総合 | 似せ度 | 似ている度 | 自然さ | 読み間違い | 長さ | 学習(分) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rs:
        if r["kind"] == "tts":
            if r.get("failed"):
                L.append(f"| {r['stage']} | {r['data']}分 | {r.get('note', '')} | - | ⚠ {r['failed']} | | | | | | |")
                continue
            L.append(f"| {r['stage']} | {r['data']}分 | {r.get('note', '')} | {r.get('epoch', '-')} | {_fmt(r)} | "
                     f"{r.get('dur', '-')} | {r.get('train_min', '-')} |")
    L += ["", "## 声変換（ステップごと）", "", "| 段階 | 素材 | 学習率 | ステップ | 総合 | 似せ度 | 似ている度 | 自然さ | 読み間違い | 学習(分) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rs:
        if r["kind"] == "vc":
            lr = f"{r['params']['lr']:g}" if r.get("params") else "学習なし"
            if r.get("failed"):
                L.append(f"| {r['stage']} | {r['data']}分 | {lr} | - | ⚠ {r['failed']} | | | | | |")
                continue
            L.append(f"| {r['stage']} | {r['data']}分 | {lr} | {r.get('step', '-')} | {_fmt(r)} | {r.get('train_min', '-')} |")
    L += ["", "「似せ度」は声の似ている度だけを、別人 = 0%・本物 = 100% に直したもの。",
          "「長さ」は本物と同じ文を読んだときの長さの比（1 = 本物と同じ速さ。大きいほどゆっくり）。",
          "聞き比べ用の音声は bench/runs/<実験>/ にあります。"]
    out = BENCH / "report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    try:
        chart(rs)
    except Exception as e:  # グラフが描けなくても表は残す
        print(f"[bench] グラフを描けませんでした: {e}", flush=True)
    return out


def chart(rs) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    # 学習率 × 周
    for e in sorted({r["exp"] for r in rs if r["kind"] == "tts" and r.get("stage") == "学習率"}):
        pts = sorted((r["epoch"], r["total"]) for r in rows_of(e))
        if pts:
            ax[0].plot(*zip(*pts), marker="o", label=rows_of(e)[0].get("note"))
    ax[0].set(title="読み上げ: 学習率と周回数", xlabel="周", ylabel="総合点（本物=100）")
    ax[0].legend(fontsize=8)
    # 素材の量
    best = {}  # 素材の量ごとの一番良い点（24 分は全実験の中で一番良いもの）
    for r in rs:
        if r["kind"] == "tts" and r.get("total") is not None and r["stage"] != "base":
            best[r["data"]] = max(best.get(r["data"], 0), r["total"])
    if best:
        xs = sorted(best)
        ax[1].plot(xs, [best[x] for x in xs], marker="o")
    ax[1].set(title="読み上げ: 素材の量", xlabel="素材（分）", ylabel="総合点（一番良い周）")
    # 声変換
    for e in sorted({r["exp"] for r in rs if r["kind"] == "vc" and r.get("stage") != "base"}):
        pts = sorted((r["step"], r["total"]) for r in rows_of(e))
        if pts:
            r0 = rows_of(e)[0]
            ax[2].plot(*zip(*pts), marker="o", label=f"{r0['data']}分 lr {r0['params']['lr']:g}")
    ax[2].set(title="声変換: 学習率とステップ", xlabel="ステップ", ylabel="総合点")
    ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(BENCH / "report.png", dpi=110)
    plt.close(fig)


def rescore() -> None:
    """保存してある全部の音声（bench/runs/）を、今の採点役でまとめて採点し直す（似ている度・自然さ）。
    途中で採点の方法を変えたときに、実験どうしを同じ物差しで比べるため。読み間違いはそのまま。"""
    from studio import quality

    try:
        rows = [json.loads(l) for l in RESULTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    except OSError:
        return
    files = sorted(RUNS.glob("*/*__*.wav"))
    log(f"全部の音声（{len(files)} 本）を採点し直しています")
    # 名前が重ならないよう「実験名__ラベル__番号」の一覧で渡し、結果は（実験, ラベル）ごとに平均する
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        links = []
        for f in files:
            dst = Path(tmp) / f"{f.parent.name}__{f.name}"
            os.link(f, dst)
            links.append(dst)
        j = quality.judge(links, sorted((TEST / "print").glob("*.wav")))
        other = quality.judge(sorted((TEST / "vc_src").glob("*.wav")), sorted((TEST / "print").glob("*.wav")),
                              mos=False)
    groups = {}
    for name, v in j.items():
        exp, label, _ = name.split("__")
        groups.setdefault((exp, label), []).append(v)
    for r in rows:
        if r.get("sim") is None:
            continue
        label = ("real" if r["kind"] == "real" else "zero" if r["stage"] == "base"
                 else f"epoch-{r['epoch']:03d}" if r["kind"] == "tts" else f"s{r['step']}")
        vals = groups.get((r["exp"], label))
        if vals:
            r["sim"], r["mos"] = quality.mean(vals, "sim"), quality.mean(vals, "mos")
        if r["kind"] == "real":
            r["sim_other"] = quality.mean(list(other.values()), "sim")
    shutil.copy(RESULTS, RESULTS.with_suffix(".before_rescore.jsonl"))
    RESULTS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    report()


def adopt() -> dict:
    """一番良かった設定を、これからの学習の既定にする（studio/train_defaults.json）。"""
    s = summary()
    if not s["tts_best"]:
        raise SystemExit("まだ読み上げの結果がありません")
    tb = s["tts_best"]
    out = {"tts": {**tb["params"], "epochs": tb["epoch"]}, "from": f"精度検証 {time.strftime('%Y-%m-%d')}",
           "level": {"tts_total": tb["total"], "tts_zero_total": (s["tts_zero"] or {}).get("total")}}
    if s["vc_best"]:
        vb = s["vc_best"]
        out["vc"] = {"lr": vb["params"]["lr"], "steps": vb["step"]}
        out["level"].update(vc_total=vb["total"], vc_zero_total=(s["vc_zero"] or {}).get("total"))
    f = ROOT / "studio" / "train_defaults.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report", "rescore", "adopt", "_prepare", "_eval-real", "_eval-tts",
                                    "_eval-vc"])
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    if a.cmd == "run":
        try:
            run_all()
        finally:
            if state().get("running"):
                state(running=False)
    elif a.cmd == "report":
        print(report())
    elif a.cmd == "rescore":
        rescore()
    elif a.cmd == "adopt":
        adopt()
    elif a.cmd == "_prepare":
        prepare()
    elif a.cmd == "_eval-real":
        eval_real()
    elif a.cmd == "_eval-tts":
        eid, voice, sub, meta = a.args
        eval_tts(eid, voice, None if sub == "-" else sub, json.loads(meta))
    elif a.cmd == "_eval-vc":
        eid, voice, pairs, cfg, meta = a.args
        eval_vc(eid, voice, json.loads(pairs), None if cfg == "-" else cfg, json.loads(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
