# Carnot Web Application

A modern web interface for the Carnot data processing engine, featuring dataset management and AI-powered file search.

## Architecture

This application consists of two main components:

- **Backend**: FastAPI (Python) server providing REST API
- **Frontend**: React application with modern UI

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload --port 8000
```

The API will be available at http://localhost:8000

API documentation: http://localhost:8000/docs

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The web app will be available at http://localhost:5173

## Features

### Data Management
- **Dataset Management**: Create, view, and delete datasets
- **File Upload**: Drag-and-drop file uploads
- **File Browser**: macOS Finder-like interface to browse files and directories
- **AI-Powered Search**: Natural language file search using a chatbot interface

### Dataset Creator
- Browse uploaded files and existing data directories
- Select files using checkboxes
- Use AI chatbot to find specific files quickly
- Add dataset annotations and metadata
- Save datasets for later use

## Project Structure

```
app/
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI application
│   │   ├── database.py       # Database models and config
│   │   ├── routes/           # API endpoints
│   │   ├── models/           # Pydantic schemas
│   │   └── services/         # Business logic
│   └── requirements.txt
│
└── frontend/
    ├── src/
    │   ├── components/       # React components
    │   ├── pages/           # Page components
    │   ├── services/        # API client
    │   └── App.jsx          # Main app component
    └── package.json
```

## API Endpoints

### Files
- `GET /api/files/browse?path=/` - Browse directory contents
- `POST /api/files/upload` - Upload a file
- `GET /api/files/uploaded` - List uploaded files

### Datasets
- `GET /api/datasets` - List all datasets
- `POST /api/datasets` - Create new dataset
- `GET /api/datasets/{id}` - Get dataset details
- `PUT /api/datasets/{id}` - Update dataset
- `DELETE /api/datasets/{id}` - Delete dataset

### Search
- `POST /api/search` - Search for files using natural language

## Development

### Backend
The backend uses:
- FastAPI for the web framework
- SQLAlchemy for database ORM
- SQLite for data storage (easily replaceable with PostgreSQL)
- Async/await for performance

### Frontend
The frontend uses:
- React 18 with hooks
- React Router for navigation
- Tailwind CSS for styling
- Axios for API calls
- Lucide React for icons

## Future Enhancements

- Enhanced Carnot engine integration for search
- User authentication and authorization
- Dataset versioning
- Advanced file filtering and sorting
- Dataset collaboration features
- User chat functionality implementation

