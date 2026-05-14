from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from app.utils.data_ingestion import load_dataset, infer_dataset_schema
from app.utils.eda import generate_summary_statistics
from app.reports.generator import generate_html_report, save_html_report

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
REPORTS_DIR = os.path.join(_BASE_DIR, "reports_output")

router = APIRouter()


class ReportRequest(BaseModel):
    file_path: str
    ai_insights: Optional[str] = ""
    leaderboard: Optional[list] = None
    problem_type: Optional[str] = ""
    model_insights: Optional[str] = ""


@router.post("/reports/generate")
async def generate_report(request: ReportRequest):
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        df = load_dataset(request.file_path)
        schema = infer_dataset_schema(df)
        summary_stats = generate_summary_statistics(df)

        html_content = generate_html_report(
            file_path=request.file_path,
            schema=schema,
            summary_stats=summary_stats,
            ai_insights=request.ai_insights or "",
            leaderboard=request.leaderboard or [],
            problem_type=request.problem_type or "",
            model_insights=request.model_insights or ""
        )

        saved_path = save_html_report(
            html_content,
            output_dir=REPORTS_DIR,
            base_name=os.path.splitext(os.path.basename(request.file_path))[0]
        )

        return {
            "message": "Report generated successfully",
            "report_path": saved_path,
            "html_content": html_content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")
