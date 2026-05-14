"""
Demo script: Generates Iris CSV and runs end-to-end pipeline test locally
(without FastAPI). Used to verify the core modules work before starting servers.
"""
import sys
import os
# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

import pandas as pd
from sklearn.datasets import load_iris
from app.utils.data_ingestion import infer_dataset_schema
from app.utils.eda import generate_summary_statistics, detect_problem_type
from app.utils.data_cleaning import handle_missing_values, detect_duplicates

DATASETS_DIR = os.path.join(_ROOT, "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)

print("=" * 60)
print("  AUTONOMOUS DATA ANALYST — End-to-End Demo (Iris Dataset)")
print("=" * 60)

# 1. Generate Iris dataset
print("\n[1/5] Loading Iris dataset...")
iris = load_iris(as_frame=True)
df = iris.frame
df['target_name'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
iris_path = os.path.join(DATASETS_DIR, "iris_demo.csv")
df.to_csv(iris_path, index=False)
print(f"      Saved to: {iris_path}")
print(f"      Shape: {df.shape}")

# 2. Schema inference
print("\n[2/5] Inferring schema...")
schema = infer_dataset_schema(df)
print(f"      Rows          : {schema['num_rows']}")
print(f"      Columns       : {schema['num_cols']}")
print(f"      Numerical     : {schema['numerical_cols']}")
print(f"      Categorical   : {schema['categorical_cols']}")
print(f"      Duplicate rows: {schema['duplicate_rows']}")

# 3. EDA summary
print("\n[3/5] Generating EDA summary statistics...")
summary = generate_summary_statistics(df)
num_recs = summary.get("numerical", [])
print(f"      Numerical stats for {len(num_recs)} columns computed ✅")
for rec in num_recs[:2]:
    col = rec.get("index", "?")
    mean = rec.get("mean", "?")
    std = rec.get("std", "?")
    skew = rec.get("skewness", "?")
    print(f"       → {col}: mean={mean:.3f}, std={std:.3f}, skew={skew:.3f}")

# 4. Problem type detection
print("\n[4/5] Detecting problem type...")
problem = detect_problem_type(df, "target")
print(f"      Detected: {problem} ✅")

# 5. Data cleaning
print("\n[5/5] Running data cleaning pipeline...")
dupes = detect_duplicates(df)
print(f"      Duplicates found: {dupes['duplicate_count']}")
df_clean = handle_missing_values(df, strategy='mean')
print(f"      Missing values handled ✅")
print(f"      Rows before: {len(df)} | Rows after: {len(df_clean)}")

print("\n" + "=" * 60)
print("  ✅ ALL MODULES PASSED — Pipeline is functional!")
print("=" * 60)
print("\nNext steps:")
print("  1. Set GEMINI_API_KEY in your .env file")
print("  2. Start backend : uvicorn app.backend.main:app --reload")
print("  3. Start frontend: streamlit run app.py")
print("  4. Open http://localhost:8501 and upload iris_demo.csv from ./datasets/")
