# Carnot Web Backend

FastAPI backend for the Carnot Web application.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file (optional):
```bash
cp .env.example .env
```

4. Run the server:
```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at http://localhost:8000

API documentation will be available at http://localhost:8000/docs

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

