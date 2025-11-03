# Carnot Web - Setup Guide

This guide will help you get the Carnot Web application up and running.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher** - [Download Python](https://www.python.org/downloads/)
- **Node.js 16 or higher** - [Download Node.js](https://nodejs.org/)
- **npm** (comes with Node.js) or **yarn**

Verify installations:
```bash
python --version  # or python3 --version
node --version
npm --version
```

## Installation Steps

### Step 1: Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a Python virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

4. Install Python dependencies:
```bash
pip install -r requirements.txt
```

5. (Optional) Create a `.env` file for environment variables:
```bash
cp .env.example .env
```

### Step 2: Frontend Setup

1. Open a new terminal window/tab and navigate to the frontend directory:
```bash
cd frontend
```

2. Install Node.js dependencies:
```bash
npm install
```

Or if you prefer yarn:
```bash
yarn install
```

## Running the Application

You'll need to run both the backend and frontend servers simultaneously. Use two terminal windows/tabs:

### Terminal 1: Start the Backend

```bash
cd backend
source venv/bin/activate  # Activate virtual environment (if not already active)
uvicorn app.main:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
```

The backend API is now running at: **http://localhost:8000**

API Documentation: **http://localhost:8000/docs**

### Terminal 2: Start the Frontend

```bash
cd frontend
npm run dev
```

You should see:
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  press h to show help
```

The frontend is now running at: **http://localhost:5173**

## Access the Application

Open your web browser and navigate to:

**http://localhost:5173**

You should see the Carnot Web interface with two tabs:
- **Data Management** - Create datasets, upload files
- **User Chat** - Placeholder for chat functionality

## Quick Test

1. **Upload a file:**
   - On the Data Management page, use the file upload section to upload a test file
   
2. **Create a dataset:**
   - Click "Create Dataset" button
   - Browse through the file system
   - Select some files using checkboxes
   - Try the AI search chatbot (e.g., type "find txt files")
   - Fill in dataset name and annotation
   - Click "Save Dataset"

3. **View your dataset:**
   - You should be redirected back to the Data Management page
   - Your new dataset should appear in the list

## Troubleshooting

### Backend Issues

**Port 8000 already in use:**
```bash
# Use a different port
uvicorn app.main:app --reload --port 8001

# Then update frontend/.env with:
# VITE_API_URL=http://localhost:8001/api
```

**Import errors:**
```bash
# Make sure virtual environment is activated and dependencies installed
pip install -r requirements.txt
```

**Database errors:**
```bash
# Delete the database and let it recreate
rm carnot_web.db
# Restart the backend
```

### Frontend Issues

**Port 5173 already in use:**
```bash
# Vite will automatically try the next available port
# Or specify a different port in vite.config.js
```

**Module not found errors:**
```bash
# Delete node_modules and reinstall
rm -rf node_modules
npm install
```

**API connection errors:**
- Make sure the backend is running on port 8000
- Check the browser console for CORS errors
- Verify the API URL in frontend/src/services/api.js

## Development Tips

### Backend Hot Reload
The `--reload` flag enables automatic reloading when you modify Python files.

### Frontend Hot Reload
Vite automatically reloads when you modify React files.

### API Documentation
Visit http://localhost:8000/docs to see interactive API documentation (Swagger UI).

### Database Inspection
The SQLite database file is created at `backend/carnot_web.db`. You can inspect it with:
```bash
sqlite3 carnot_web.db
.tables
.schema datasets
```

## Production Build

### Backend
For production, use a production-grade server like Gunicorn:
```bash
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Frontend
Build the frontend for production:
```bash
cd frontend
npm run build
```

The built files will be in `frontend/dist/` and can be served with any static file server.

## Next Steps

- Explore the codebase
- Customize the UI styling in `frontend/src/index.css`
- Enhance the search functionality with Carnot integration
- Add authentication if needed
- Deploy to a production server

## Getting Help

If you encounter issues:
1. Check the terminal output for error messages
2. Check the browser console for frontend errors
3. Review the API documentation at http://localhost:8000/docs
4. Check that both servers are running

Enjoy using Carnot Web! 🚀

