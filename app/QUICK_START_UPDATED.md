# 🚀 Carnot Web - Quick Start Guide (Updated for Your Setup)

Get up and running in 5 minutes using your existing `.venv312`!

## Installation (One-Time Setup)

### Step 1: Backend Setup (using your existing .venv312)
```bash
cd app/backend
source ../.venv312/bin/activate   # Use existing venv312
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

### Option A: Using Startup Script (Recommended)

The startup script will automatically find and use your `.venv312`:

```bash
cd app
./start.sh
```

### Option B: Manual Start (Two Terminals)

**Terminal 1 - Backend:**
```bash
cd app/backend
source ../.venv312/bin/activate
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

## Your Existing Data

The file browser will automatically show:
- `data/` - Your existing data directory (e.g., enron-eval-medium)
- `uploaded_files/` - New files you upload through the web interface

---

**Happy coding with Carnot! 🎉**

