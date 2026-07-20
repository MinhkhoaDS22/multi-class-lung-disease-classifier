@echo off
chcp 65001 > nul
echo ============================================
echo    MediScan AI - Khoi dong Flutter Web
echo ============================================
echo.

cd /d "%~dp0mediscan_app"

echo Kiem tra Flutter...
flutter --version

echo.
echo ============================================
echo    Flutter Web dang chay tai:
echo    http://localhost:3000
echo ============================================
echo.
echo Nhan Ctrl+C de dung server
echo.

flutter run -d chrome --web-port 3000

pause
