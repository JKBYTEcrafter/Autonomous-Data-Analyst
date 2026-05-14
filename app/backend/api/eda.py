from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from app.utils.data_ingestion import load_dataset
from app.utils.eda import (
    generate_summary_statistics, generate_correlation_matrix,
    generate_distribution_plots, generate_categorical_plots,
    generate_boxplots, generate_statistical_tests, detect_problem_type
)

router = APIRouter()


class EDARequest(BaseModel):
    file_path: str


class ProblemDetectionRequest(BaseModel):
    file_path: str
    target_column: str


class StatTestRequest(BaseModel):
    file_path: str
    target_column: Optional[str] = None


@router.post("/eda/summary")
async def get_summary(request: EDARequest):
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        df = load_dataset(request.file_path)
        summary = generate_summary_statistics(df)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")


@router.post("/eda/plots")
async def get_plots(request: EDARequest):
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        df = load_dataset(request.file_path)
        corr_matrix = generate_correlation_matrix(df)
        dist_plots = generate_distribution_plots(df)
        cat_plots = generate_categorical_plots(df)
        box_plots = generate_boxplots(df)

        return {
            "correlation_matrix": corr_matrix,
            "distribution_plots": dist_plots,
            "categorical_plots": cat_plots,
            "box_plots": box_plots
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating plots: {str(e)}")


@router.post("/eda/statistical_tests")
async def get_statistical_tests(request: StatTestRequest):
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        df = load_dataset(request.file_path)
        results = generate_statistical_tests(df, target_col=request.target_column)
        return {"statistical_tests": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running statistical tests: {str(e)}")


@router.post("/eda/detect_problem")
async def detect_problem(request: ProblemDetectionRequest):
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        df = load_dataset(request.file_path)
        problem_type = detect_problem_type(df, request.target_column)
        return {"problem_type": problem_type}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting problem type: {str(e)}")
