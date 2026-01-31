@echo off
cd /d %~dp0
call .venv\Scripts\activate

python verify_unknown.py --context office

pause
