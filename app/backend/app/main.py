from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app.routes import conversations, datasets, files, query, search


# Initialize database
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(title="Carnot Web API", lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(files.router, prefix="/api/files", tags=["files"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(query.router, prefix="/api/query", tags=["query"])
app.include_router(conversations.router, prefix="/api/conversations", tags=["conversations"])

@app.get("/")
async def root():
    return {"message": "Carnot Web API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

