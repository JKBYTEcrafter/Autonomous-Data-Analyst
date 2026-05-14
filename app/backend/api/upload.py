from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import sys
import os

# Add root to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from app.utils.data_ingestion import save_uploaded_file, load_dataset, infer_dataset_schema

router = APIRouter()

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(('.csv', '.xls', '.xlsx', '.json')):
        raise HTTPException(status_code=400, detail="Invalid file format. Supported: CSV, Excel, JSON")
    
    try:
        # Save file
        file_path = save_uploaded_file(file)
        
        # Load and infer
        df = load_dataset(file_path)
        schema = infer_dataset_schema(df)
        
        return JSONResponse(content={
            "message": "File uploaded and parsed successfully",
            "file_path": file_path,
            "schema": schema
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the file: {str(e)}")
