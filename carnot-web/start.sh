#!/bin/bash

# Carnot Web - Startup Script
# This script starts both the backend and frontend servers

echo "🚀 Starting Carnot Web Application..."
echo ""

# Function to cleanup background processes on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit
}

trap cleanup EXIT INT TERM

# Get absolute path to script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if virtual environment exists (either in backend or parent directory)
if [ -d "$SCRIPT_DIR/backend/venv" ]; then
    VENV_PATH="$SCRIPT_DIR/backend/venv"
elif [ -d "$SCRIPT_DIR/../.venv312" ]; then
    VENV_PATH="$SCRIPT_DIR/../.venv312"
elif [ -d "$SCRIPT_DIR/../.venv311" ]; then
    VENV_PATH="$SCRIPT_DIR/../.venv311"
else
    echo "❌ Backend virtual environment not found!"
    echo "Please activate your Python virtual environment and run: pip install -r backend/requirements.txt"
    exit 1
fi

echo "Using virtual environment: $VENV_PATH"

# Check if node_modules exists for frontend
if [ ! -d "frontend/node_modules" ]; then
    echo "❌ Frontend node_modules not found!"
    echo "Please run: cd frontend && npm install"
    exit 1
fi

# Start backend
echo "📦 Starting backend server on http://localhost:8000..."
cd backend
source $VENV_PATH/bin/activate
uvicorn app.main:app --reload --port 8000 > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Start frontend
echo "⚛️  Starting frontend server on http://localhost:5173..."
cd frontend
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ Servers are starting..."
echo ""
echo "📊 Backend:  http://localhost:8000"
echo "📊 API Docs: http://localhost:8000/docs"
echo "🌐 Frontend: http://localhost:5173"
echo ""
echo "📝 Logs:"
echo "   Backend:  tail -f backend.log"
echo "   Frontend: tail -f frontend.log"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Wait for processes
wait

