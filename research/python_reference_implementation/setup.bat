@echo off
REM setup.bat

REM Create virtual environment
python -m venv .venv

REM Activate virtual environment (Windows)
call .venv\Scripts\activate.bat

REM Install requirements
pip install -r requirements.txt

REM Run setup script
python setup.py

echo Setup completed successfully! You can now run the main script with 'python main.py'.
pause