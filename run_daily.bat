@echo off
REM ============================================================
REM  HK Stock Prediction - Daily Run
REM
REM  IMPORTANT for editing this file:
REM    1. Keep it ASCII only. No CJK / non-ASCII characters.
REM       cmd.exe reads .bat in the OEM codepage; UTF-8 CJK text makes the
REM       parser emit garbage command names and syntax errors.
REM    2. Save with CRLF line endings. LF-only .bat files mis-parse.
REM    3. Do not put ")" inside echo text inside an if-block; it closes the
REM       block early. This script uses goto instead of parenthesised blocks.
REM ============================================================
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
if not exist "%SCRIPT_DIR%run_daily.bat" goto :dir_error
pushd "%SCRIPT_DIR%"
if errorlevel 1 goto :dir_error

REM Use the venv interpreter by absolute path. Do NOT rely on activate.bat:
REM if activation fails, bare "python" silently resolves to the global
REM interpreter, which cannot import the project package.
set "PY=%SCRIPT_DIR%venv\Scripts\python.exe"
if not exist "%PY%" goto :py_error

REM Redirection into a missing directory fails with "access denied".
if not exist "%SCRIPT_DIR%logs" mkdir "%SCRIPT_DIR%logs" >nul 2>&1

REM --- Market calendar gate -------------------------------------------------
REM Single source of truth: src/trading_calendar.py (weekends + HK public holidays).
REM Exit 0 = trading day, 3 = market closed, other = check failed (fail open).
echo Checking HK market calendar...
"%PY%" -c "import sys; from src.trading_calendar import should_run_today; ok, reason = should_run_today(); print(reason); sys.exit(0 if ok else 3)"
if errorlevel 3 goto :market_closed
if errorlevel 1 goto :calendar_degraded
goto :run_steps

REM --- Market closed --------------------------------------------------------
:market_closed
echo Market closed - skipping train, cleanup and predict.
>> "%SCRIPT_DIR%logs\run_log.txt" echo [%date% %time%] Market closed - skipped
goto :finish

REM --- Calendar lookup degraded, proceed anyway -----------------------------
:calendar_degraded
echo WARNING: calendar check failed - continuing anyway, fail-open.
>> "%SCRIPT_DIR%logs\run_log.txt" echo [%date% %time%] WARNING calendar check failed - continuing anyway

REM --- Shared run steps -----------------------------------------------------
:run_steps
echo ====================================
echo  HK Stock Prediction - Daily Run
echo  Started: %date% %time%
echo ====================================
>> "%SCRIPT_DIR%logs\run_log.txt" echo ====================================
>> "%SCRIPT_DIR%logs\run_log.txt" echo Started at %date% %time%

echo.
echo [1/3] Training model. Voting ~5-15 min; Stacking and Blending take longer.
"%PY%" "%SCRIPT_DIR%src\train_model.py"
>> "%SCRIPT_DIR%logs\run_log.txt" echo [1/3] Training finished with code %errorlevel%

echo.
echo [2/3] Cleaning up old records older than 60 days.
"%PY%" "%SCRIPT_DIR%src\cleanup_old.py"
>> "%SCRIPT_DIR%logs\run_log.txt" echo [2/3] Cleanup finished with code %errorlevel%

echo.
echo [3/3] Predicting and uploading to Supabase.
"%PY%" "%SCRIPT_DIR%src\predict_upload.py"
>> "%SCRIPT_DIR%logs\run_log.txt" echo [3/3] Predict finished with code %errorlevel%

echo.
echo ====================================
echo  Completed: %date% %time%
echo ====================================
>> "%SCRIPT_DIR%logs\run_log.txt" echo Finished at %date% %time%
>> "%SCRIPT_DIR%logs\run_log.txt" echo ====================================

:finish
popd
REM Keep the window open only when launched interactively for debugging.
REM Never pause under Task Scheduler, otherwise the job hangs forever.
if "%HK_PAUSE%"=="1" pause
endlocal
exit /b 0

REM --- Fatal setup errors ---------------------------------------------------
:dir_error
echo ERROR: cannot resolve the script directory, got "%SCRIPT_DIR%".
echo ERROR: run this file from inside the project root.
pause
exit /b 1

:py_error
echo ERROR: venv interpreter not found at "%PY%".
echo ERROR: create it with:  python -m venv venv
pause
exit /b 1