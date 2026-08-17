@echo off
cd /d "%~dp0"

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Check if today is Saturday (6) or Sunday (7)
for /f %%a in ('powershell -command "(Get-Date).DayOfWeek.value__"') do set DOW=%%a

if "%DOW%"=="6" (
    echo Saturday - Market closed, skipping >> logs\run_log.txt
    goto :end
)
if "%DOW%"=="7" (
    echo Sunday - Market closed, skipping >> logs\run_log.txt
    goto :end
)

:: Log start time
echo ================================== >> logs\run_log.txt
echo Started at %date% %time% >> logs\run_log.txt

:: Train model
echo Training model... >> logs\run_log.txt
python src\train_model.py >> logs\run_log.txt 2>&1

:: Cleanup old records (keep 60 days)
echo Cleaning up old records... >> logs\run_log.txt
python src\cleanup_old.py >> logs\run_log.txt 2>&1

:: Predict and upload
echo Predicting... >> logs\run_log.txt
python src\predict_upload.py >> logs\run_log.txt 2>&1

:: Log finish time
echo Finished at %date% %time% >> logs\run_log.txt
echo ================================== >> logs\run_log.txt

:end
:: Keep window open if double-clicked (for debugging)
if "%1"=="" pause
