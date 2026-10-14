# Voice Studio を（必要なら）起動し、スマホ用リンクが開ける状態か確認して結果を表示する。
# 使い方: powershell -NoProfile -ExecutionPolicy Bypass -File open-studio.ps1
$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
$url = "https://kumoshibito.tailfd662b.ts.net/"
$ts = "C:\Program Files\Tailscale\tailscale.exe"

function Test-Up {
    try { (Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 10).StatusCode -eq 200 } catch { $false }
}

# 1. Tailscale がつながっているか（切れていれば接続し直す）
$state = (& $ts status --json | ConvertFrom-Json).BackendState
if ($state -ne "Running") {
    & $ts up | Out-Null
    Start-Sleep 5
    $state = (& $ts status --json | ConvertFrom-Json).BackendState
}

# 2. 中継（tailscale serve）が設定されているか
$serve = (& $ts serve status 2>&1) -join "`n"
if ($serve -notmatch "localhost:7860") {
    & $ts serve --bg --yes https+insecure://localhost:7860 | Out-Null
}

# 3. Voice Studio が起動しているか（していなければ起動）
$listening = Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue
$started = $false
if (-not $listening) {
    Start-Process -FilePath "$root\start-studio.bat" -WorkingDirectory $root -WindowStyle Minimized
    $started = $true
}

# 4. 開けるようになるまで待つ（最大 2 分）
$ok = $false
for ($i = 0; $i -lt 40; $i++) {
    if (Test-Up) { $ok = $true; break }
    Start-Sleep 3
}

$pw = (Get-Content "$root\.studio\password.txt" -ErrorAction SilentlyContinue | Select-Object -First 1)
Write-Output "tailscale=$state"
Write-Output "started_now=$started"
Write-Output "reachable=$ok"
Write-Output "url=$url"
Write-Output "user=voice"
Write-Output "password=$pw"
Write-Output "qr=$root\outputs\voice_studio_qr.png"
