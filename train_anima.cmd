@echo off
REM ============================================================
REM  Anima LoRA training console  (LOCAL: this PC, RTX 3070 8GB)
REM
REM  Double-click  =  start the local console + open the browser.
REM
REM  Usage:
REM    train_anima.cmd            start on the default port (8791)
REM    train_anima.cmd 8899       start on another port
REM
REM  This console trains on THIS machine, using kohya sd-scripts.
REM  A training run goes into a DETACHED child process: closing this
REM  window does NOT stop training (use the "stop training" button).
REM
REM  Backend (sd-scripts + venv) is installed by:
REM    python train_anima\setup_backend.py
REM  This launcher offers to do it on first run.
REM
REM  NOTE: keep this file ASCII-only and CRLF-terminated. cmd.exe
REM  mis-parses LF-only .bat files (it drops the REM prefix and then
REM  tries to run every word as a command).
REM ============================================================
setlocal
chcp 65001 >nul

REM --- Chinese output must stay readable AND must actually appear: when stdout
REM --- is redirected to a file Python block-buffers it, and serve_forever()
REM --- then blocks forever, so the startup banner would never be flushed.
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"

set "HERE=%~dp0"
set "PORT=%~1"
if "%PORT%"=="" set "PORT=8791"

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

if not exist "%HERE%train_anima\server.py" (
    echo [ERROR] train_anima\server.py not found next to this .cmd
    popd
    pause
    exit /b 1
)

REM sd-scripts location: ANIMA_SD_DIR wins, else ask paths.py to resolve it.
set "SD=%ANIMA_SD_DIR%"
if "%SD%"=="" for /f "delims=" %%i in ('call "%PY%" "%HERE%paths.py" --get ANIMA_SD_DIR 2^>nul') do set "SD=%%i"
if exist "%SD%\anima_train_network.py" goto run_server

echo.
echo   [!] Training backend not installed yet:
echo       %SD%\anima_train_network.py
echo.
echo       Install now? It clones kohya sd-scripts and creates a venv
echo       (inherits torch from the existing Python 3.12; needs network).
echo.
set /p "ANS=  Type Y to install, anything else to cancel: "
if /i "%ANS%"=="Y" goto install_backend
echo   cancelled.
popd
pause
exit /b 3

:install_backend
echo.
"%PY%" "%HERE%train_anima\setup_backend.py"
if errorlevel 1 goto install_failed

:run_server
"%PY%" -u "%HERE%train_anima\server.py" --port %PORT% --open
set "RC=%ERRORLEVEL%"
popd
pause
exit /b %RC%

:install_failed
echo.
echo [ERROR] Backend install failed - see the messages above.
popd
pause
exit /b 1
