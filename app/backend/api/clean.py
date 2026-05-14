from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from app.utils.data_ingestion import load_dataset, infer_dataset_schema, DATASETS_DIR
from app.utils.data_cleaning import handle_missing_values, encode_categorical, scale_features, remove_outliers

router = APIRouter()


class CleanRequest(BaseModel):
    file_path: str
    missing_value_strategy: str = 'mean'
    encode_strategy: str = 'label'
    scale_strategy: str = 'standard'
    outlier_method: str = 'iqr'
    remove_duplicates: bool = True


@router.post("/clean")
async def clean_dataset(request: CleanRequest):
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        df = load_dataset(request.file_path)

        # 0. Remove duplicate rows
        if request.remove_duplicates:
            df = df.drop_duplicates().reset_index(drop=True)

        # 1. Handle missing values
        df = handle_missing_values(df, strategy=request.missing_value_strategy)

        # 2. Remove outliers
        df = remove_outliers(df, method=request.outlier_method)

        # 3. Encode categorical features
        df = encode_categorical(df, strategy=request.encode_strategy)

        # 4. Scale features
        df = scale_features(df, strategy=request.scale_strategy)

        # Save the cleaned dataset — fix: use os.path.splitext to avoid replacing all dots
        base, ext = os.path.splitext(request.file_path)
        cleaned_file_path = f"{base}_cleaned.csv"
        df.to_csv(cleaned_file_path, index=False)

        # Infer new schema
        schema = infer_dataset_schema(df)

        return {
            "message": "Dataset cleaned successfully",
            "cleaned_file_path": cleaned_file_path,
            "schema": schema,
            "rows_after_cleaning": int(len(df))
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning dataset: {str(e)}")
