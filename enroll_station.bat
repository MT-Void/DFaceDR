@echo off
cd /d %~dp0

REM Activate virtual environment
call .venv\Scripts\activate

REM Run enrollment station
python enroll_station.py

pause
