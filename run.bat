@echo off
REM Windows run script

IF NOT EXIST .venv (
    python install.py
    IF ERRORLEVEL 1 EXIT /B 1
)

call .venv\Scripts\activate

REM Run system test
python test_system.py

REM Start FastAPI server
start /B python -m uvicorn src.api.claims_api:app --reload --port 8000

REM Start Streamlit dashboard
python -m streamlit run src/dashboard/enhanced_dashboard.py 