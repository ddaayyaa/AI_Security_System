@echo off
echo =====================================================
echo   SENTINEL PRO - Advanced AI Security System
echo =====================================================
echo.

:: Step 1: Install base requirements
echo [1/3] Installing base requirements...
pip install flask Pillow numpy -q

:: Step 2: Try OpenCV
echo [2/3] Installing OpenCV...
pip install opencv-python -q

:: Step 3: Try face_recognition (needs CMake + dlib)
echo [3/3] Attempting face_recognition install...
pip install face-recognition -q
if %errorlevel% neq 0 (
    echo [WARN] face_recognition not installed - running in DEMO MODE
    echo        For full recognition: install CMake from cmake.org
    echo        Then run: pip install dlib face-recognition
)

echo.
echo =====================================================
echo   Starting server at http://localhost:5000
echo   Admin login: admin / admin123
echo =====================================================
echo.
python app.py
pause
