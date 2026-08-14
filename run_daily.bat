@echo off
cd /d "%~dp0"
echo Running at %date% %time% >> logs\run_log.txt
python src\init_database.py >> logs\run_log.txt 2>&1
python src\predict_upload.py >> logs\run_log.txt 2>&1
echo Finished at %date% %time% >> logs\run_log.txt
