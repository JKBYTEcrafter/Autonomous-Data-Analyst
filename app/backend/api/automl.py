from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from app.automl.engine import run_automl_pipeline

router = APIRouter()

class AutoMLRequest(BaseModel):
    file_path: str
    target_column: str
    problem_type: Optional[str] = None

@router.post("/automl/train")
async def train_models(request: AutoMLRequest):
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    try:
        results = run_automl_pipeline(
            file_path=request.file_path,
            target_col=request.target_column,
            problem_type=request.problem_type
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running AutoML pipeline: {str(e)}")
