from __future__ import annotations

import ast
import numpy as np
import pandas as pd
import re
from typing import List, Optional, Sequence

BRACKETED_PATTERN = re.compile(r"^\s*\[.*\]\s*$")
def _safe_literal_eval_list(s: str) -> Optional[List]:
    """ '[1,2,3]' 같은 문자열을 안전하게 리스트로 파싱. 실패 시 None """
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return v
        return None
    except Exception:
        return None
def expand_vector_columns(
    df: pd.DataFrame,
    candidates: Sequence[str],
    expected_len: Optional[int] = 6,
    drop_original: bool = False,
) -> pd.DataFrame:
    """
    벡터형 문자열('[...]') 컬럼을 *_0..*_k 형태로 확장.
    - concat으로 한 번에 붙여 조각화 방지
    - drop_original=True면 원본 문자열 컬럼 제거
    """
    out = df.copy()
    new_blocks: List[pd.DataFrame] = []
    cols_to_drop: List[str] = []

    for col in candidates:
        if col not in out.columns:
            continue

        s = out[col].astype(str)
        if (s.str.match(BRACKETED_PATTERN)).mean() < 0.6:
            continue

        parsed_lists = [_safe_literal_eval_list(v) for v in s]

        n = None
        for lst in parsed_lists:
            if isinstance(lst, list) and (expected_len is None or len(lst) == expected_len):
                n = len(lst)
                break
        if n is None:
            continue

        mat = np.empty((len(parsed_lists), n), dtype=object)
        mat[:] = np.nan
        for i, lst in enumerate(parsed_lists):
            if isinstance(lst, list) and len(lst) >= n:
                mat[i, :n] = lst[:n]

        block = pd.DataFrame(
            mat,
            columns=[f"{col}_{i}" for i in range(n)],
            index=out.index,
        ).apply(pd.to_numeric, errors="coerce")

        new_blocks.append(block)
        if drop_original:
            cols_to_drop.append(col)

    if new_blocks:
        out = pd.concat([out] + new_blocks, axis=1)
        if cols_to_drop:
            out = out.drop(columns=cols_to_drop)

    return out.copy()

# ------------------------------------------------------------------------------
# Type coercion
# ------------------------------------------------------------------------------

def coerce_bools_and_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """
    - 'True'/'False' 문자열 → int(1/0)
    - 숫자로 변환 가능한 문자열은 numeric으로 캐스팅
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "O":
            s = out[col].astype(str).str.strip()

            # bool
            mask_bool = s.isin(["True", "False", "true", "false"])
            if mask_bool.mean() > 0.8:
                out.loc[mask_bool, col] = s[mask_bool].str.lower().map({"true": 1, "false": 0})
                out[col] = pd.to_numeric(out[col], errors="coerce")
                continue

            # 숫자 문자열 (대괄호 제외)
            non_bracket = ~s.str.match(BRACKETED_PATTERN)
            if non_bracket.mean() > 0.5:
                coerced = pd.to_numeric(out[col], errors="coerce")
                if coerced.notna().mean() >= 0.6:
                    out[col] = coerced
    return out

# ------------------------------------------------------------------------------
# Normalization
# ------------------------------------------------------------------------------

def normalize(df: pd.DataFrame, exclude_cols=("timestamp",)) -> pd.DataFrame:
    """
    int/float 컬럼만 min-max 정규화 (bool 제외).
    """
    out = df.copy()
    for col in out.columns:
        if col in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(out[col]) and not pd.api.types.is_bool_dtype(out[col]):
            col_min = out[col].min()
            col_max = out[col].max()
            denom = col_max - col_min
            if pd.isna(col_min) or pd.isna(col_max):
                out[col] = 0.0
            elif denom == 0:
                out[col] = 0.0
            else:
                out[col] = (out[col] - col_min) / denom
    return out

# ------------------------------------------------------------------------------
# Full pipeline
# ------------------------------------------------------------------------------

def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    권장 파이프라인:
      1) 벡터 문자열 컬럼 확장(q, tau 등 → *_0..*_5)
      2) bool/숫자 캐스팅
      3) 숫자 컬럼만 정규화
    """
    VECTOR_COL_CANDIDATES = [
        "q","qdot","qddot","qdes","qdotdes","qddotdes",
        "p","pdot","pddot","pdes","pdotdes","pddotdes",
        "tau","tau_act","tau_ext",
        "status_codes","temperatures","voltages","currents",
        "servo_actives","brake_actives",
    ]
    x = expand_vector_columns(df, candidates=VECTOR_COL_CANDIDATES,
                              expected_len=6, drop_original=False)
    x = coerce_bools_and_numbers(x)
    x = normalize(x, exclude_cols=("timestamp",))
    return x
