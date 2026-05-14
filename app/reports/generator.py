import os
import sys
import json
from datetime import datetime

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _BASE_DIR)

from app.utils.data_ingestion import load_dataset, infer_dataset_schema
from app.utils.eda import generate_summary_statistics


def generate_html_report(
    file_path: str,
    schema: dict,
    summary_stats: dict,
    ai_insights: str = "",
    leaderboard: list = None,
    problem_type: str = "",
    model_insights: str = ""
) -> str:
    """
    Generates a self-contained HTML report for the dataset analysis.
    Returns the HTML string.
    """
    df = load_dataset(file_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = os.path.basename(file_path)

    # Build numerical summary table
    num_table_html = ""
    if summary_stats.get("numerical"):
        rows = ""
        for rec in summary_stats["numerical"]:
            rows += f"""<tr>
                <td>{rec.get('index', '')}</td>
                <td>{rec.get('mean', 'N/A')}</td>
                <td>{rec.get('std', 'N/A')}</td>
                <td>{rec.get('min', 'N/A')}</td>
                <td>{rec.get('50%', 'N/A')}</td>
                <td>{rec.get('max', 'N/A')}</td>
                <td>{rec.get('missing_pct', 'N/A')}%</td>
                <td>{rec.get('skewness', 'N/A')}</td>
            </tr>"""
        num_table_html = f"""
        <h2>📊 Numerical Column Statistics</h2>
        <table>
            <tr><th>Column</th><th>Mean</th><th>Std</th><th>Min</th><th>Median</th><th>Max</th><th>Missing %</th><th>Skewness</th></tr>
            {rows}
        </table>"""

    # Build leaderboard table
    lb_table_html = ""
    if leaderboard:
        if len(leaderboard) > 0:
            headers = list(leaderboard[0].keys())
            header_row = "".join(f"<th>{h}</th>" for h in headers)
            data_rows = ""
            for i, rec in enumerate(leaderboard[:10]):
                row_class = "best-row" if i == 0 else ""
                cells = "".join(f"<td>{rec.get(h, '')}</td>" for h in headers)
                data_rows += f"<tr class='{row_class}'>{cells}</tr>"
            lb_table_html = f"""
        <h2>🏆 AutoML Leaderboard — {problem_type}</h2>
        <table>
            <tr>{header_row}</tr>
            {data_rows}
        </table>"""

    # AI insights formatting
    insights_html = ""
    if ai_insights:
        insights_html = f"""
        <h2>🧠 AI-Generated Insights</h2>
        <div class="insights-box">
            {ai_insights.replace(chr(10), '<br>')}
        </div>"""

    model_insights_html = ""
    if model_insights:
        model_insights_html = f"""
        <h2>🤖 Model Explanation</h2>
        <div class="insights-box">
            {model_insights.replace(chr(10), '<br>')}
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Autonomous Data Analyst — Report: {filename}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Inter', sans-serif; background: #0e1117; color: #e0e0e0; padding: 40px; }}
  .header {{ text-align: center; padding: 40px 0; border-bottom: 1px solid #2d3139; margin-bottom: 40px; }}
  .header h1 {{ font-size: 2.5rem; font-weight: 800;
    background: linear-gradient(135deg, #FF6B6B, #4ECDC4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .header p {{ color: #8b949e; margin-top: 8px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 40px; }}
  .kpi-card {{ background: #1e2127; border-radius: 12px; padding: 24px; border: 1px solid #2d3139; text-align: center; }}
  .kpi-value {{ font-size: 2rem; font-weight: 800; color: #4ECDC4; }}
  .kpi-label {{ color: #8b949e; font-size: 0.85rem; text-transform: uppercase; margin-top: 6px; }}
  h2 {{ color: #4ECDC4; margin: 40px 0 16px; font-size: 1.3rem; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; background: #1e2127; border-radius: 8px; overflow: hidden; }}
  th {{ background: #2d3139; padding: 12px 16px; text-align: left; color: #8b949e; font-size: 0.85rem; text-transform: uppercase; }}
  td {{ padding: 10px 16px; border-bottom: 1px solid #2d3139; font-size: 0.9rem; }}
  .best-row td {{ color: #FFE66D; font-weight: 600; }}
  .insights-box {{ background: #1e2127; border-left: 4px solid #4ECDC4; border-radius: 8px; padding: 24px;
    line-height: 1.8; color: #c0c0c0; }}
  .footer {{ text-align: center; color: #555; margin-top: 60px; padding-top: 20px; border-top: 1px solid #2d3139; }}
  .badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;
    background: #4ECDC422; color: #4ECDC4; border: 1px solid #4ECDC455; margin: 4px; }}
</style>
</head>
<body>
<div class="header">
  <h1>🤖 Autonomous Data Analyst</h1>
  <p>AI-Powered Data Science Report | Generated: {timestamp}</p>
  <p style="margin-top:8px;"><span class="badge">📁 {filename}</span></p>
</div>

<h2>📋 Dataset Overview</h2>
<div class="kpi-grid">
  <div class="kpi-card"><div class="kpi-value">{schema.get('num_rows', 0):,}</div><div class="kpi-label">Total Rows</div></div>
  <div class="kpi-card"><div class="kpi-value">{schema.get('num_cols', 0)}</div><div class="kpi-label">Columns</div></div>
  <div class="kpi-card"><div class="kpi-value">{len(schema.get('numerical_cols', []))}</div><div class="kpi-label">Numerical</div></div>
  <div class="kpi-card"><div class="kpi-value">{schema.get('duplicate_rows', 0)}</div><div class="kpi-label">Duplicates</div></div>
</div>

{num_table_html}

{insights_html}

{lb_table_html}

{model_insights_html}

<div class="footer">
  <p>Generated by <strong>Autonomous Data Analyst</strong> — Powered by Gemini AI & PyCaret</p>
</div>
</body>
</html>"""

    return html


def save_html_report(html_content: str, output_dir: str, base_name: str = "report") -> str:
    """Saves the HTML report and returns the file path."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.html"
    file_path = os.path.join(output_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return file_path
