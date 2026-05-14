import os
import sys
import io
import traceback
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _BASE_DIR)

from app.utils.data_ingestion import load_dataset

load_dotenv()


def _get_gemini_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your-gemini-api-key":
        raise ValueError("Gemini API key not configured. Please set GEMINI_API_KEY in your .env file.")
    return api_key


def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
        google_api_key=_get_gemini_key()
    )


def generate_eda_insights(summary_stats: dict) -> str:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert AI Data Scientist. Analyze the following dataset summary statistics "
            "and provide 5 key business insights. Format your response with numbered points. "
            "Be concise, specific, and actionable."
        )),
        ("user", "Summary Statistics:\n{summary}")
    ])
    chain = prompt | llm
    response = chain.invoke({"summary": str(summary_stats)})
    return response.content


def generate_model_insights(leaderboard: list, problem_type: str) -> str:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert AI Data Scientist. Analyze the following AutoML leaderboard for a "
            "{problem_type} problem. Explain: 1) Which model performed best and why, 2) What the "
            "key metrics mean in business context, 3) Recommendations for deployment. "
            "Format with clear sections."
        )),
        ("user", "Leaderboard:\n{leaderboard}")
    ])
    chain = prompt | llm
    response = chain.invoke({"problem_type": problem_type, "leaderboard": str(leaderboard)})
    return response.content


def query_dataset(file_path: str, query: str, chat_history: list = None) -> str:
    """
    Natural language querying using Gemini code generation.
    Generates Python/pandas code to answer the query, executes it, and returns the result.
    No PandasAI dependency — works entirely with LangChain + Gemini.
    """
    api_key = _get_gemini_key()
    df = load_dataset(file_path)

    # Build dataset context
    col_info = []
    for col in df.columns:
        col_info.append(f"  - {col}: {df[col].dtype} (sample: {df[col].dropna().head(3).tolist()})")
    dataset_context = (
        f"DataFrame variable name: df\n"
        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
        f"Columns:\n" + "\n".join(col_info)
    )

    # Build conversation history string
    history_str = ""
    if chat_history:
        for msg in chat_history[-6:]:  # Last 3 exchanges
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"

    system_prompt = f"""You are an AI data analyst. You have access to a pandas DataFrame called `df`.
    
Dataset Information:
{dataset_context}

Your task:
1. Understand the user's question
2. Write Python code using pandas to answer it
3. The code must print the result or assign it to a variable called `result`
4. Return ONLY the Python code block, nothing else

Rules:
- Use only pandas, numpy operations on the `df` variable
- Always handle NaN values gracefully
- Keep the answer concise
- If you cannot answer with code, set result = "I cannot answer this question from the available data."

Previous conversation:
{history_str}"""

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.1,
        google_api_key=api_key
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Question: {query}\n\nWrite Python code to answer this:")
    ]

    response = llm.invoke(messages)
    code = response.content

    # Strip markdown code fences if present
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()

    # Execute the generated code safely
    local_vars = {"df": df.copy(), "pd": pd}
    try:
        import numpy as np
        local_vars["np"] = np

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        exec(code, {"__builtins__": __builtins__}, local_vars)

        printed = sys.stdout.getvalue()
        sys.stdout = old_stdout

        result = local_vars.get("result", None)

        if result is not None:
            if isinstance(result, pd.DataFrame):
                return result.head(20).to_markdown(index=False)
            elif isinstance(result, pd.Series):
                return result.head(20).to_string()
            else:
                return str(result)
        elif printed.strip():
            return printed.strip()
        else:
            return "Query executed successfully but returned no output."

    except Exception as e:
        sys.stdout = old_stdout
        # Ask Gemini to explain the error
        return f"I tried to answer your question but encountered an error: {str(e)}\n\nGenerated code:\n```python\n{code}\n```"
