@echo off
rem Claude Code install + voice repo + Remote Control (double-click to run)
setlocal
set "REPO=https://github.com/masakikana0731-beep/voice"
set "BRANCH=ccr-96507d6e-dwl2v2"
set "DIR=%USERPROFILE%\voice"

echo [1/4] Git ...
where git >nul 2>nul || winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
set "PATH=%ProgramFiles%\Git\cmd;%PATH%"

echo [2/4] Claude Code ...
where claude >nul 2>nul || powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://claude.ai/install.ps1 | iex"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

echo [3/4] voice folder ...
if not exist "%DIR%\.git" git clone -b %BRANCH% %REPO% "%DIR%"
if not exist "%DIR%" mkdir "%DIR%"
cd /d "%DIR%"

echo [4/4] Remote Control ...
echo   Scan the QR code with your phone (Claude app).
claude rc
if errorlevel 1 (
  echo.
  echo   Remote Control failed. Starting Claude to log in.
  echo   After login, type  /rc  and press Enter.
  claude
)
pause
