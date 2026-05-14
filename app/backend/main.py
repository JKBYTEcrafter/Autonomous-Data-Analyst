from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys

# Ensure project root is in sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from app.backend.api.upload import router as upload_router
from app.backend.api.clean import router as clean_router
from app.backend.api.eda import router as eda_router
from app.backend.api.automl import router as automl_router
from app.backend.api.insights import router as insights_router
from app.backend.api.reports import router as reports_router

app = FastAPI(
    title="Autonomous Data Analyst API",
    description="Backend API for the Autonomous Data Analyst platform.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Create required directories on startup."""
    dirs = ["datasets", "app/models", "reports_output"]
    for d in dirs:
        path = os.path.join(_ROOT, d)
        os.makedirs(path, exist_ok=True)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Autonomous Data Analyst API v2!", "docs": "/docs"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "2.0.0"}


app.include_router(upload_router, prefix="/api/v1", tags=["Data Ingestion"])
app.include_router(clean_router, prefix="/api/v1", tags=["Data Cleaning"])
app.include_router(eda_router, prefix="/api/v1", tags=["Exploratory Data Analysis"])
app.include_router(automl_router, prefix="/api/v1", tags=["AutoML"])
app.include_router(insights_router, prefix="/api/v1", tags=["AI Insights"])
app.include_router(reports_router, prefix="/api/v1", tags=["Reports"])

if __name__ == "__main__":
    uvicorn.run("app.backend.main:app", host="0.0.0.0", port=8000, reload=True)
