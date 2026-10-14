"""スマホなど他の端末から開くための準備（パスワード・HTTPS 証明書・接続先の表示）。

設定は .studio/ に保存（git には入れない）:
  password.txt  ログインのパスワード（ユーザー名は voice）。消すと次回起動時に作り直す
  cert.pem / key.pem  自己署名の HTTPS 証明書（スマホのブラウザでマイクを使うには HTTPS が必須）
"""

import datetime
import ipaddress
import secrets
import socket

from studio.common import ROOT

CONF = ROOT / ".studio"
USER = "voice"


def lan_ip() -> str:
    """この PC の家庭内 LAN の IP アドレス（外部には何も送らない。経路の決定に使うだけ）。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 80))  # 文書用の予約アドレス。実際に通信はしない
        return s.getsockname()[0]
    finally:
        s.close()


def password() -> str:
    CONF.mkdir(exist_ok=True)
    f = CONF / "password.txt"
    if not f.exists():
        # 打ちやすいよう、紛らわしい文字（0/O, 1/l など）を除いた 8 文字
        f.write_text("".join(secrets.choice("abcdefghjkmnpqrstuvwxyz23456789") for _ in range(8)), encoding="utf-8")
    return f.read_text(encoding="utf-8").strip()


def cert(ip: str) -> tuple[str, str]:
    """自己署名証明書を作る（IP が変わったら作り直す）。"""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    CONF.mkdir(exist_ok=True)
    crt, key, mark = CONF / "cert.pem", CONF / "key.pem", CONF / "cert_ip.txt"
    if crt.exists() and key.exists() and mark.exists() and mark.read_text() == ip:
        return str(crt), str(key)
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Voice Studio")])
    now = datetime.datetime.now(datetime.timezone.utc)
    c = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key())
         .serial_number(x509.random_serial_number())
         .not_valid_before(now - datetime.timedelta(days=1)).not_valid_after(now + datetime.timedelta(days=3650))
         .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address(ip)),
                                                     x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                                                     x509.DNSName("localhost")]), critical=False)
         .sign(k, hashes.SHA256()))
    crt.write_bytes(c.public_bytes(serialization.Encoding.PEM))
    key.write_bytes(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                    serialization.NoEncryption()))
    mark.write_text(ip)
    return str(crt), str(key)


TAILSCALE = r"C:\Program Files\Tailscale\tailscale.exe"


def tailscale_url() -> str | None:
    """Tailscale で外出先から開ける URL（https://<PC名>.<tailnet>.ts.net）。未設定なら None。

    `tailscale serve` が Tailscale 内向けに本物の HTTPS 証明書でこの UI を中継する。
    Tailscale の外（インターネット）には公開されない。
    """
    import json
    import subprocess
    from pathlib import Path

    if not Path(TAILSCALE).exists():
        return None
    try:
        st = json.loads(subprocess.run([TAILSCALE, "status", "--json"], capture_output=True, text=True,
                                       timeout=10).stdout)
        if st.get("BackendState") != "Running":
            return None
        # 中継が未設定なら設定する（--bg で PC 再起動後も残る）
        subprocess.run([TAILSCALE, "serve", "--bg", "--yes", "http://localhost:7860"],
                       capture_output=True, text=True, timeout=15)
        return "https://" + st["Self"]["DNSName"].rstrip(".")
    except Exception:
        return None


def show_qr(url: str) -> None:
    import qrcode

    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.print_ascii(invert=True)


def show() -> str | None:
    """起動画面に、スマホ用の URL と QR コードを表示する。Tailscale の URL を返す。"""
    ts = tailscale_url()
    print("\n" + "=" * 60)
    if ts:
        print(" スマホから開く（スマホの Tailscale アプリをオンにして、この URL か QR コードから）")
        print(f"   {ts}")
        print("=" * 60)
        show_qr(ts)
    else:
        print(" Tailscale につながっていないので、この PC からだけ開けます: http://127.0.0.1:7860")
        print("=" * 60)
    return ts
