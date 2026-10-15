"""Seed-VC のリアルタイム変換画面（real-time-gui.py）の日本語化。

元のファイルは書き換えず、起動時に表示の文字だけを差し替える（studio/vc_run.py から使う）。
ボタンなどの動作は key で判定しているので、表示を変えても動きは変わらない。
"""

JA = {
    "Seed-VC - GUI": "リアルタイム声変換（Seed-VC）",
    "Load reference audio": "なりたい声（参照音声）",
    "choose an audio file": "ファイルを選ぶ",
    "Choose an audio file": "参照音声のファイルを選んでください",
    "Sound Device": "マイクと出力",
    "Device type": "方式",
    "WASAPI Exclusive Device": "WASAPI 排他モード",
    "Input Device": "入力（マイク）",
    "Output Device": "出力",
    "Reload devices": "機器を読み直す",
    "Use model sampling rate": "モデルのサンプリング周波数を使う",
    "Use device sampling rate": "機器のサンプリング周波数を使う",
    "Sampling rate:": "サンプリング周波数:",
    "Regular settings": "基本の設定",
    "Diffusion steps": "品質（多いほど自然・遅れが増える）",
    "Inference cfg rate": "声の寄せ具合（cfg）",
    "Max prompt length (s)": "参照音声の使う長さ（秒）",
    "Activation threshold": "反応する音量のしきい値",
    "Performance settings": "速さの設定",
    "Block time": "処理の区切り（短いほど遅れが少ない）",
    "Crossfade length": "つなぎ目の重なり",
    "Extra content encoder context time (left)": "前を見る長さ（内容）",
    "Extra DiT context time (left)": "前を見る長さ（声）",
    "Extra context time (right)": "後ろを見る長さ",
    "Start Voice Conversion": "▶ 変換スタート",
    "Stop Voice Conversion": "■ 停止",
    "Input listening": "マイクの音をそのまま聞く",
    "Voice Conversion": "声を変換する",
    "Algorithm delay (ms):": "仕組み上の遅れ（ミリ秒）:",
    "Inference time (ms):": "変換にかかった時間（ミリ秒）:",
    "audio file path contains non-ascii characters": "ファイルの場所（フォルダ名）に日本語などが入っていると使えません",
    "WAV Files": "WAV ファイル",
    "FLAC Files": "FLAC ファイル",
    "OGG Files": "OGG ファイル",
    "Opus Files": "Opus ファイル",
}


def patch(source: str) -> str:
    """ソース中の文字列リテラル（"..."）のうち、表示用のものだけを日本語に置き換える。"""
    for en, ja in JA.items():
        source = source.replace(f'"{en}"', f'"{ja}"')
    return source
