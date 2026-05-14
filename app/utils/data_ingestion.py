import pandas as pd
import os
import uuid
from fastapi import UploadFile

# Use absolute path relative to THIS file so it works regardless of CWD
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASETS_DIR = os.path.join(_BASE_DIR, "datasets")


def save_uploaded_file(file: UploadFile) -> str:
    """Saves the uploaded file to the datasets directory and returns the absolute file path."""
    os.makedirs(DATASETS_DIR, exist_ok=True)

    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    saved_filename = f"{file_id}{file_extension}"
    file_path = os.path.join(DATASETS_DIR, saved_filename)

    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    return file_path


def load_dataset(file_path: str) -> pd.DataFrame:
    """Loads a dataset into a Pandas DataFrame based on file extension."""
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.csv':
        try:
            return pd.read_csv(file_path)
        except UnicodeDecodeError:
            import chardet
            with open(file_path, 'rb') as f:
                result = chardet.detect(f.read())
            return pd.read_csv(file_path, encoding=result['encoding'])
    elif file_extension in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    elif file_extension == '.json':
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


def infer_dataset_schema(df: pd.DataFrame) -> dict:
    """Infers the schema of the dataset: numerical, categorical, datetime, etc."""
    duplicate_count = int(df.duplicated().sum())

    schema = {
        "num_rows": int(len(df)),
        "num_cols": int(len(df.columns)),
        "numerical_cols": df.select_dtypes(include=['number']).columns.tolist(),
        "categorical_cols": df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist(),
        "datetime_cols": df.select_dtypes(include=['datetime']).columns.tolist(),
        "memory_usage": int(df.memory_usage(deep=True).sum()),
        "duplicate_rows": duplicate_count,
        "columns": {}
    }

    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_count = int(df[col].nunique())
        missing_count = int(df[col].isnull().sum())
        schema["columns"][col] = {
            "dtype": dtype,
            "unique_count": unique_count,
            "missing_count": missing_count,
            "missing_percentage": round(float(missing_count / len(df)) * 100, 2) if len(df) > 0 else 0.0
        }

    return schema
