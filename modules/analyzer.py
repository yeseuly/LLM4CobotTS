# modules/llm_analyzer.py
from __future__ import annotations

import os
from typing import Optional, List
import pandas as pd

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# LangChain OpenAI (chat-style)
from langchain_openai import ChatOpenAI

# OpenAI SDK (low-level)
from openai import OpenAI
from openai import RateLimitError, APIError, APITimeoutError

# ------------------------------------------------------------------------------
# Environment & Clients
# ------------------------------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is missing. Set it in your .env file or environment."
    )

# Ensure OpenAI SDK sees the key
os.environ["OPENAI_API_KEY"] = API_KEY

# Model names can be overridden by env vars
LC_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")     # for LangChain ChatOpenAI
OA_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")     # for OpenAI SDK

# LangChain chat LLM (good for quick invoke / tools)
llm = ChatOpenAI(
    model=LC_MODEL,
    temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),  # deterministic default
    top_p=float(os.getenv("OPENAI_TOP_P", "1.0")),
    max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2048")),
)

# OpenAI low-level client (good for fine-grained control & streaming later)
client = OpenAI(api_key=API_KEY)

def _df_preview(df: pd.DataFrame, rows: int = 8) -> str:
    """Small, safe preview of the dataframe (head & tail if large)."""
    if df.empty:
        return "[EMPTY DATAFRAME]"
    if len(df) <= rows:
        return df.to_string(index=False)
    head = df.head(rows // 2)
    tail = df.tail(rows - len(head))
    parts = [
        "[HEAD]\n" + head.to_string(index=False),
        "[TAIL]\n" + tail.to_string(index=False),
    ]
    return "\n\n".join(parts)

def _schema_summary(df: pd.DataFrame) -> str:
    """Column dtypes & basic stats (numeric only)."""
    lines: List[str] = []
    lines.append("Columns & dtypes:")
    for c, dt in df.dtypes.items():
        lines.append(f"- {c}: {dt}")
    try:
        desc = df.describe(include="all").fillna("").astype(str)
        lines.append("\nBasic describe():")
        lines.append(desc.to_string())
    except Exception:
        pass
    return "\n".join(lines)

def ping() -> str:
    """
    Quick health check using LangChain ChatOpenAI.
    Returns model echo like 'ping' or a short reply.
    """
    return llm.invoke("ping").content  # type: ignore[attr-defined]

@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=(
        retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(APITimeoutError)
        | retry_if_exception_type(APIError)
    ),
)
def _openai_chat(messages, model: Optional[str] = None, **kwargs) -> str:
    """
    Low-level OpenAI chat call with backoff retries.
    """
    resp = client.chat.completions.create(
        model=model or OA_MODEL,
        messages=messages,
        **kwargs,
    )
    return resp.choices[0].message.content or ""

def analyze_with_llm(
    df: pd.DataFrame,
    prompt: Optional[str] = None,
    rows: int = 8,
    use_langchain: bool = False,
) -> str:
    """
    Summarize cobot time-series and ask LLM for interpretation/anomalies.

    Args:
        df: normalized or raw time-series dataframe (must include timestamp or index).
        prompt: additional user instruction (optional).
        rows: preview rows to send.
        use_langchain: if True, call LangChain ChatOpenAI; else OpenAI SDK.

    Returns:
        LLM-generated analysis string.
    """
    preview = _df_preview(df, rows=rows)
    schema = _schema_summary(df)

    system = (
        "You are a senior robotics data analyst specializing in collaborative robots (cobots) "
        "and time-series interpretation. Provide crisp, technical insights with bullet points."
    )

    user = f"""
Cobot time-series snippet (preview only):
{preview}

Schema & stats:
{schema}

Guidelines:
- Identify patterns, regimes/phases, and any anomalies or outliers.
- Discuss correlations between joints/axes if evident.
- If the snippet suggests a potential fault (backlash, collision, encoder noise, saturation), explain briefly.
- Recommend next diagnostic steps (plots, thresholds, segmentation) concretely.

{f"Additional instruction: {prompt}" if prompt else ""}
"""

    if use_langchain:
        # LangChain path
        res = llm.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]  # type: ignore[arg-type]
        )
        return getattr(res, "content", str(res))

    # OpenAI SDK path (default)
    return _openai_chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
        top_p=float(os.getenv("OPENAI_TOP_P", "0.9")),
        max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "1200")),
    )

def quick_hello() -> str:
    """
    Simple hello using OpenAI SDK to verify credentials/models.
    """
    return _openai_chat(
        messages=[{"role": "user", "content": "Hello from LLM4CobotTS!"}],
        model=OA_MODEL,
        temperature=0.0,
        max_tokens=40,
    )
