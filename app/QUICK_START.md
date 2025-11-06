# 🚀 Carnot Web - Quick Start Guide

Get up and running in 5 minutes!

## Prerequisites Check

Make sure you have these installed:
```bash
python --version   # Should be 3.8+
node --version     # Should be 16+
npm --version      # Comes with Node.js
```

Don't have them? 
- [Install Python](https://www.python.org/downloads/)
- [Install Node.js](https://nodejs.org/)

## Installation (One-Time Setup)

### Step 1: Backend Setup
```bash
cd app/backend
python -m venv venv
source venv/bin/activate          # macOS/Linux
# OR
venv\Scripts\activate             # Windows

pip install -r requirements.txt
cd ..
```

### Step 2: Frontend Setup
```bash
cd app/frontend
npm install
cd ..
```

## Running the Application

### Option A: Using Startup Scripts (Recommended)

**macOS/Linux:**
```bash
cd app
./start.sh
```

**Windows:**
```cmd
cd app
start.bat
```

### Option B: Manual Start (Two Terminals)

**Terminal 1 - Backend:**
```bash
cd app/backend
source venv/bin/activate    # macOS/Linux
# OR
venv\Scripts\activate       # Windows

uvicorn app.main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd app/frontend
npm run dev
```

## Access the Application

Once both servers are running:

🌐 **Frontend**: http://localhost:5173
📊 **Backend API**: http://localhost:8000
📚 **API Docs**: http://localhost:8000/docs

## First Steps

1. **Upload a file**: Drag and drop a file on the Data Management page
2. **Create a dataset**: Click "Create Dataset" button
3. **Browse files**: Navigate through directories like macOS Finder
4. **Select files**: Check boxes next to files you want
5. **Try AI search**: Ask the chatbot to "find txt files"
6. **Add metadata**: Fill in dataset name and description
7. **Save**: Click "Save Dataset" and see it appear in your list!

## Troubleshooting

### Backend won't start
```bash
# Make sure you're in the virtual environment
cd backend
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Frontend won't start
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Port already in use
**Backend (change port):**
```bash
uvicorn app.main:app --reload --port 8001
```
Then update `frontend/src/services/api.js` line 3:
```javascript
const API_BASE_URL = 'http://localhost:8001/api'
```

**Frontend (auto-assigns new port):**
- Vite will automatically use the next available port

### Can't connect to backend
1. Make sure backend is running on port 8000
2. Check browser console for errors
3. Visit http://localhost:8000/docs to verify backend is up

## Stop the Application

- **If using startup scripts**: Press `Ctrl+C`
- **If manual**: Press `Ctrl+C` in both terminal windows

## What's Next?

- 📖 Read `README.md` for full feature overview
- 🎨 Check `UI_GUIDE.md` for UI/UX details
- 📚 Read `SETUP.md` for detailed setup instructions
- 💻 Explore `IMPLEMENTATION_SUMMARY.md` for technical details

## Common Tasks

### Reset Database
```bash
cd app/backend
rm carnot_web.db
# Restart backend - database will be recreated
```

### Clear Uploaded Files
```bash
cd app/backend
rm -rf uploaded_files/*
```

### Build for Production
```bash
# Frontend
cd frontend
npm run build
# Built files will be in frontend/dist/

# Backend
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## Need Help?

- 📝 Check the logs in terminal output
- 🔍 Use browser Developer Tools (F12) to see errors
- 📚 Review API docs at http://localhost:8000/docs
- 📖 Read detailed documentation files

---

**Happy coding with Carnot! 🎉**

