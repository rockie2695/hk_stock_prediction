@echo off
cd /d "%~dp0"

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Market calendar gate / 交易日曆判斷
:: Single source of truth lives in src/trading_calendar.py - covers weekends AND
:: HK public holidays. Exit code 3 = closed market, skip everything.
:: 判斷邏輯集中於 src/trading_calendar.py，同時處理週末與香港公眾假期。
:: 回傳碼 3 代表休市，跳過所有步驟。
echo Checking HK market calendar...
python -c "import sys; from src.trading_calendar import should_run_today; ok, reason = should_run_today(); print(reason); sys.exit(0 if ok else 3)"
if errorlevel 3 (
    echo Market closed - skipping
    echo Market closed - %date% %time%>> logs\run_log.txt
    goto :end
)
if errorlevel 1 (
    echo WARNING: calendar check failed, continuing anyway (fail-open)
    echo WARNING: calendar check failed at %date% %time%>> logs\run_log.txt
)

echo ====================================
echo  HK Stock Prediction - Daily Run
echo  Started: %date% %time%
echo ====================================
echo ==================================== >> logs\run_log.txt
echo Started at %date% %time% >> logs\run_log.txt

:: Train model
echo.
echo [1/3] Training model (Voting ~5-15 min; Stacking/Blending longer)...
echo [1/3] Training model... >> logs\run_log.txt
python src\train_model.py

:: Cleanup old records (keep 60 days)
echo.
echo [2/3] Cleaning up old records (60+ days)...
echo [2/3] Cleaning up old records... >> logs\run_log.txt
python src\cleanup_old.py

:: Predict and upload
echo.
echo [3/3] Predicting and uploading to Supabase...
echo [3/3] Predicting... >> logs\run_log.txt
python src\predict_upload.py

:: Log finish time
echo.
echo ====================================
echo  Completed: %date% %time%
echo ====================================
echo Finished at %date% %time% >> logs\run_log.txt
echo ==================================== >> logs\run_log.txt

:end
:: Keep window open if double-clicked (for debugging)
pause
