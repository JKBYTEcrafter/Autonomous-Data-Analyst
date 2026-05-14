---
title: Autonomous Data Analyst
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🤖 Autonomous Data Analyst

An AI-Powered, End-to-End Data Science Platform that automates data cleaning, exploratory data analysis, machine learning, and business insight generation — powered by LLMs (Gemini, OpenAI, Anthropic).

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌟 Features

| Feature | Description |
|---|---|
| 📥 **Data Ingestion** | Upload CSV, Excel, or JSON files via UI or API |
| 🧹 **Intelligent Cleaning** | Auto-handles missing values, outliers, encoding & scaling |
| 📊 **Automated EDA** | Rich Plotly dashboards, correlation matrices, distribution plots |
| 🤖 **AutoML Pipeline** | Auto-detects classification/regression, trains & compares multiple models |
| 💡 **AI Insights** | LLM-powered business insight generation (Gemini / OpenAI) |
| 💬 **Chat with Data** | Natural language querying via LangChain code-generation |
| 📄 **Report Generation** | Auto-generated PDF/HTML analysis reports |
| 🔌 **REST API** | Full FastAPI backend for programmatic access |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│              Streamlit Frontend (app.py)         │
│   Upload → EDA → AutoML → Insights → Chat       │
└──────────────────┬──────────────────────────────┘
                   │ HTTP
┌──────────────────▼──────────────────────────────┐
│           FastAPI Backend (app/backend/)         │
│  /upload  /eda  /automl  /insights  /chat       │
└──────┬──────────┬──────────┬───────────┬────────┘
       │          │          │           │
  app/utils  app/eda   app/automl  app/insights
  (cleaning) (plots)   (PyCaret)   (LangChain)
                                   (Gemini/GPT)
```

---

## 🚀 Quickstart

### Prerequisites
- Python 3.10+
- A Gemini API Key (free at [aistudio.google.com](https://aistudio.google.com)) or OpenAI API Key

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/autonomous-data-analyst.git
cd autonomous-data-analyst
```

### 2. Create a Virtual Environment
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
```bash
cp .env.example .env
```
Edit `.env` and add your API key:
```env
GEMINI_API_KEY=your-gemini-api-key
# Or use OpenAI:
# OPENAI_API_KEY=your-openai-api-key
```

### 5. Run the Application

**Option A — Streamlit UI only:**
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501)

**Option B — With FastAPI Backend:**
```bash
# Terminal 1
uvicorn app.backend.main:app --reload

# Terminal 2
streamlit run app.py
```
- UI: [http://localhost:8501](http://localhost:8501)
- API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 6. Run the Demo Pipeline
```bash
python demo_pipeline.py
```

---

## 🐳 Docker Compose
```bash
docker-compose up --build
```
- UI: `http://localhost:8501`
- API: `http://localhost:8000/docs`

---

## 📁 Project Structure

```
autonomous-data-analyst/
├── app.py                  # Streamlit frontend entrypoint
├── demo_pipeline.py        # End-to-end demo script
├── requirements.txt
├── docker-compose.yml
├── .env.example
│
├── app/
│   ├── backend/            # FastAPI routes & API logic
│   │   └── api/            # Endpoint modules (automl, eda, etc.)
│   ├── automl/             # AutoML engine (PyCaret wrapper)
│   ├── eda/                # EDA report generation
│   ├── insights/           # LLM insight generation (LangChain)
│   ├── llm/                # LLM provider abstraction (Gemini/OpenAI)
│   ├── models/             # Saved ML model artifacts
│   ├── reports/            # Report generation logic
│   ├── ui/                 # Streamlit UI components
│   ├── utils/              # Data ingestion, cleaning helpers
│   └── visualization/      # Plotly chart builders
│
├── configs/                # YAML configuration files
├── datasets/               # Sample datasets
├── notebooks/              # Exploration notebooks
└── tests/                  # Unit & integration tests
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **Backend** | FastAPI |
| **AutoML** | PyCaret, Scikit-learn |
| **LLM / AI** | LangChain, Google Gemini, OpenAI |
| **Visualizations** | Plotly, Matplotlib, Seaborn |
| **Data** | Pandas, NumPy |
| **Containerization** | Docker, Docker Compose |

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
"# Autonomous-Data-Analyst" 
