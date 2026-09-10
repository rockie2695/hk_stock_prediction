@echo off
cd /d "%~dp0"

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Check if today is Saturday (6) or Sunday (7)
for /f %%a in ('powershell -command "(Get-Date).DayOfWeek.value__"') do set DOW=%%a

if "%DOW%"=="6" (
    echo Saturday - Market closed, skipping
    echo Saturday - Market closed, skipping >> logs\run_log.txt
    goto :end
)
if "%DOW%"=="7" (
    echo Sunday - Market closed, skipping
    echo Sunday - Market closed, skipping >> logs\run_log.txt
    goto :end
)

echo ====================================
echo  HK Stock Prediction - Daily Run
echo  Started: %date% %time%
echo ====================================
echo ==================================== >> logs\run_log.txt
echo Started at %date% %time% >> logs\run_log.txt

:: Train model
echo.
echo [1/3] Training model (this may take 3-8 minutes)...
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
