@echo off
REM ============================================================
REM  LoRA training console  (Qwen-Image-2.1 LoRA on AutoDL)
REM
REM  Double-click  =  start the local console + open the browser.
REM
REM  Usage:
REM    train_web.cmd            start on the default port (7870)
REM    train_web.cmd 8888       start on another port
REM
REM  The console listens on 127.0.0.1 only (it can reach the cloud
REM  instance, so it must not be exposed to the LAN).
REM
REM  Closing this window stops the LOCAL console. A training run that
REM  was already launched keeps running on the cloud instance - that
REM  is the whole point. Use the "stop training" button to stop it.
REM
REM  NOTE: keep this file ASCII-only and CRLF-terminated. cmd.exe
REM  mis-parses LF-only .bat files (it drops the REM prefix and then
REM  tries to run every word as a command).
REM ============================================================
setlocal
chcp 65001 >nul

set "HERE=%~dp0"
set "PORT=%~1"
if "%PORT%"=="" set "PORT=7870"

REM ---- Python interpreter (never hard-code a machine path) ----
REM  1) PY from paths.local.cmd  (private to this machine, git-ignored)
REM  2) otherwise the first python on PATH
REM  3) otherwise fail with instructions
if exist "%HERE%paths.local.cmd" call "%HERE%paths.local.cmd"
if not defined PY for /f "delims=" %%i in ('where python 2^>nul') do if not defined PY set "PY=%%i"
if not defined PY set "PY=python"

"%PY%" -c "import sys" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found: %PY%
    echo         Install Python and put it on PATH, or copy
    echo         paths.local.cmd.example to paths.local.cmd and set PY there.
    pause
    exit /b 1
)

pushd "%HERE%"

REM ---- preflight: paramiko is required for every SSH call ----
"%PY%" -c "import paramiko" 1>nul 2>nul
if errorlevel 1 (
    echo [ERROR] paramiko is missing in: %PY%
    echo         Install it:  "%PY%" -m pip install paramiko
    popd
    pause
    exit /b 1
)

"%PY%" "%HERE%train_web\server.py" %PORT% --open
set "RC=%ERRORLEVEL%"

popd
pause
exit /b %RC%
