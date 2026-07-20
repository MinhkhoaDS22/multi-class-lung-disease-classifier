@echo off
chcp 65001 > nul
echo ============================================
echo    MediScan AI - Khoi dong Backend Python
echo ============================================
echo.

cd /d "%~dp0mediscan_backend"

echo [1/2] Kiem tra Python...
python --version
if %errorlevel% neq 0 (
    echo LỖI: Python chưa được cài đặt!
    pause
    exit /b 1
)

echo.
echo [2/2] Cai dat thu vien (neu chua co)...
pip install -r requirements.txt --quiet

echo.
echo ============================================
echo    Server API dang chay tai:
echo    http://localhost:8000
echo    Docs: http://localhost:8000/docs
echo ============================================
echo.
echo Nhan Ctrl+C de dung server
echo.

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause
