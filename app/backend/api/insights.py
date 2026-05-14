from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from app.insights.generator import generate_eda_insights, generate_model_insights, query_dataset

router = APIRouter()


class EDAInsightsRequest(BaseModel):
    summary_stats: dict


class ModelInsightsRequest(BaseModel):
    leaderboard: list
    problem_type: str


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class QueryRequest(BaseModel):
    file_path: str
    query: str
    chat_history: Optional[List[ChatMessage]] = []


@router.post("/insights/eda")
async def get_eda_insights(request: EDAInsightsRequest):
    try:
        insights = generate_eda_insights(request.summary_stats)
        return {"insights": insights}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/insights/model")
async def get_model_insights(request: ModelInsightsRequest):
    try:
        insights = generate_model_insights(request.leaderboard, request.problem_type)
        return {"insights": insights}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/insights/query")
async def chat_with_data(request: QueryRequest):
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        history = [{"role": m.role, "content": m.content} for m in (request.chat_history or [])]
        response = query_dataset(request.file_path, request.query, chat_history=history)
        return {"response": response}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
