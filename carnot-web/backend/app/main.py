from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import files, datasets, search
from app.database import init_db

app = FastAPI(title="Carnot Web API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
@app.on_event("startup")
async def startup():
    await init_db()

# Include routers
app.include_router(files.router, prefix="/api/files", tags=["files"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(search.router, prefix="/api/search", tags=["search"])

@app.get("/")
async def root():
    return {"message": "Carnot Web API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

