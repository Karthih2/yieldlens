@echo off
echo ======================================================================
echo           YieldLens — Semiconductor Yield Intelligence Engine
echo ======================================================================
echo Starting FastAPI Backend Server on http://127.0.0.1:8000 ...
echo Web Frontend UI available at http://127.0.0.1:8000/ui/
echo API Interactive Documentation at http://127.0.0.1:8000/docs
echo ======================================================================

python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
pause
