@echo off
REM Carnot Web - Startup Script for Windows
REM This script starts both the backend and frontend servers

echo.
echo 🚀 Starting Carnot Web Application...
echo.

REM Check if virtual environment exists for backend
if not exist "backend\venv" (
    echo ❌ Backend virtual environment not found!
    echo Please run: cd backend ^&^& python -m venv venv ^&^& venv\Scripts\activate ^&^& pip install -r requirements.txt
    pause
    exit /b 1
)

REM Check if node_modules exists for frontend
if not exist "frontend\node_modules" (
    echo ❌ Frontend node_modules not found!
    echo Please run: cd frontend ^&^& npm install
    pause
    exit /b 1
)

REM Start backend in new window
echo 📦 Starting backend server on http://localhost:8000...
start "Carnot Backend" cmd /k "cd backend && venv\Scripts\activate && uvicorn app.main:app --reload --port 8000"

REM Wait a moment for backend to start
timeout /t 3 /nobreak > nul

REM Start frontend in new window
echo ⚛️  Starting frontend server on http://localhost:5173...
start "Carnot Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ✅ Servers are starting in separate windows...
echo.
echo 📊 Backend:  http://localhost:8000
echo 📊 API Docs: http://localhost:8000/docs
echo 🌐 Frontend: http://localhost:5173
echo.
echo Close the server windows to stop the application
echo.
pause

