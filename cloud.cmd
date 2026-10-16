@echo off
REM ============================================================
REM  Cloud ComfyUI control  (Qwen-Image-2.1 on AutoDL)
REM
REM  Double-click  =  cloud.cmd menu  (interactive menu)
REM
REM  Subcommands:
REM    cloud.cmd            interactive menu
REM    cloud.cmd start      start the service and wait until ready
REM    cloud.cmd stop       stop the service
REM    cloud.cmd restart    stop then start
REM    cloud.cmd status     status: service / GPU / queue / log tail
REM    cloud.cmd queue      queue only (with per-task detail)
REM    cloud.cmd logs [n]   tail the cloud comfy.log (default 40 lines)
REM    cloud.cmd menu       interactive menu
REM    cloud.cmd where      which config file is used (no SSH needed)
REM
REM  If the cloud instance is powered off this fails fast on purpose:
REM  no long retry loop. Boot it in the AutoDL console, wait 1-2 min.
REM
REM  NOTE: keep this file ASCII-only and CRLF-terminated. cmd.exe
REM  mis-parses LF-only .bat files (it drops the REM prefix and then
REM  tries to run every word as a command).
REM ============================================================
setlocal

REM Keep the console in UTF-8: the Python side prints UTF-8, and a GBK
REM console would decode those bytes into mojibake.
chcp 65001 >nul

set "HERE=%~dp0"

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

REM Double-clicked (no args) -> pause at the end so the window stays readable.
REM Called from a terminal with args -> do not pause.
set "PAUSEIT="
if "%~1"=="" set "PAUSEIT=1"

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

"%PY%" "%HERE%cloud_service.py" %*
set "RC=%ERRORLEVEL%"

popd
if defined PAUSEIT pause
exit /b %RC%
