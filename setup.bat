@echo off
echo ========================================
echo  Hong Kong Stock Prediction - Setup
echo ========================================
echo.
if exist venv\Scripts\activate (
    echo Virtual environment already exists. Activating...
) else (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        echo Try running this script as Administrator.
        pause
        exit /b 1
    )
)
echo Activating virtual environment...
call venv\Scripts\activate
echo.
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)
echo.
echo ========================================
echo  Setup complete!
echo  1. Copy .env.example to .env
echo  2. Fill in your Supabase credentials
echo  3. Run: python src\init_database.py
echo  4. Run: python src\train_model.py
echo ========================================
pause
