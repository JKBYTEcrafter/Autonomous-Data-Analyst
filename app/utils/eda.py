import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from scipy import stats as scipy_stats


def _sanitize(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def generate_summary_statistics(df: pd.DataFrame) -> dict:
    """Generates detailed summary statistics — all values serialization-safe."""
    summary = {}
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    if len(num_cols) > 0:
        desc = df[num_cols].describe().T
        desc['skewness'] = df[num_cols].skew()
        desc['kurtosis'] = df[num_cols].kurtosis()
        desc['missing'] = df[num_cols].isnull().sum()
        desc['missing_pct'] = (df[num_cols].isnull().sum() / len(df) * 100).round(2)
        summary['numerical'] = _sanitize(desc.reset_index().to_dict(orient='records'))

    if len(cat_cols) > 0:
        cat_summary = df[cat_cols].describe().T
        summary['categorical'] = _sanitize(cat_summary.reset_index().to_dict(orient='records'))

    summary['shape'] = {'rows': int(len(df)), 'cols': int(len(df.columns))}
    summary['missing_total'] = int(df.isnull().sum().sum())
    summary['duplicate_rows'] = int(df.duplicated().sum())

    return summary


def generate_correlation_matrix(df: pd.DataFrame) -> str:
    """Generates a correlation matrix as a JSON string for Plotly."""
    num_cols = df.select_dtypes(include=['number'])
    if num_cols.empty or num_cols.shape[1] < 2:
        return ""
    corr = num_cols.corr()
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        title="Correlation Matrix",
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
    return fig.to_json()


def generate_distribution_plots(df: pd.DataFrame) -> dict:
    """Generates histogram + box plots for numerical columns."""
    plots = {}
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols[:15]:  # Cap at 15 to avoid timeout
        fig = px.histogram(
            df, x=col, marginal="box",
            title=f"Distribution of {col}",
            color_discrete_sequence=["#4ECDC4"]
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        plots[col] = fig.to_json()
    return plots


def generate_categorical_plots(df: pd.DataFrame) -> dict:
    """Generates bar charts for categorical columns."""
    plots = {}
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
    for col in cat_cols[:10]:  # Cap at 10
        val_counts = df[col].value_counts().nlargest(20).reset_index()
        val_counts.columns = [col, 'count']
        fig = px.bar(
            val_counts, x=col, y='count',
            title=f"Frequency of {col}",
            color_discrete_sequence=["#FF6B6B"]
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        plots[col] = fig.to_json()
    return plots


def generate_boxplots(df: pd.DataFrame) -> dict:
    """Generates box plots for outlier visualization."""
    plots = {}
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols[:15]:
        fig = px.box(df, y=col, title=f"Boxplot — {col}", color_discrete_sequence=["#FFE66D"])
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        plots[col] = fig.to_json()
    return plots


def generate_statistical_tests(df: pd.DataFrame, target_col: str = None) -> dict:
    """
    Runs statistical tests:
    - Normality (Shapiro-Wilk) for each numeric col
    - Chi-square for categorical cols vs target (if provided and categorical)
    - T-test between two groups if target is binary
    """
    results = {}
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Normality tests
    normality = {}
    for col in num_cols[:10]:
        clean = df[col].dropna()
        if len(clean) < 3:
            continue
        sample = clean.sample(min(5000, len(clean)), random_state=42)
        stat, p = scipy_stats.shapiro(sample)
        normality[col] = {
            "statistic": round(float(stat), 4),
            "p_value": round(float(p), 6),
            "is_normal": bool(p > 0.05),
            "interpretation": "Normally distributed" if p > 0.05 else "Not normally distributed"
        }
    results['normality_tests'] = normality

    # Chi-Square tests (categorical vs target)
    if target_col and target_col in cat_cols:
        chi_results = {}
        for col in cat_cols:
            if col == target_col:
                continue
            contingency = pd.crosstab(df[col], df[target_col])
            chi2, p, dof, _ = scipy_stats.chi2_contingency(contingency)
            chi_results[col] = {
                "chi2_statistic": round(float(chi2), 4),
                "p_value": round(float(p), 6),
                "degrees_of_freedom": int(dof),
                "is_significant": bool(p < 0.05),
                "interpretation": "Significant association" if p < 0.05 else "No significant association"
            }
        results['chi_square_tests'] = chi_results

    # T-tests: numeric cols vs binary target
    if target_col and target_col in df.columns:
        unique_vals = df[target_col].dropna().unique()
        if len(unique_vals) == 2:
            ttest_results = {}
            g1 = df[df[target_col] == unique_vals[0]]
            g2 = df[df[target_col] == unique_vals[1]]
            for col in num_cols[:10]:
                if col == target_col:
                    continue
                t_stat, p = scipy_stats.ttest_ind(
                    g1[col].dropna(), g2[col].dropna(), equal_var=False
                )
                ttest_results[col] = {
                    "t_statistic": round(float(t_stat), 4),
                    "p_value": round(float(p), 6),
                    "is_significant": bool(p < 0.05),
                    "interpretation": f"Significant difference between groups" if p < 0.05 else "No significant difference"
                }
            results['t_tests'] = ttest_results

    return results


def detect_problem_type(df: pd.DataFrame, target_col: str) -> str:
    """Detects if the problem is classification, regression, or clustering."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    target = df[target_col]

    if pd.api.types.is_numeric_dtype(target):
        unique_vals = target.nunique()
        if unique_vals <= 2:
            return "Binary Classification"
        elif unique_vals <= 10 or unique_vals < len(df) * 0.05:
            return "Multi-Class Classification"
        return "Regression"
    else:
        unique_vals = target.nunique()
        if unique_vals == 2:
            return "Binary Classification"
        return "Multi-Class Classification"
