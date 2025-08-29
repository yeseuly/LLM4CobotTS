# modules/data_loader.py
from __future__ import annotations
import re
from datetime import datetime
from typing import Iterable, Optional

import pandas as pd

def _parse_ts_msec_style(ts: str) -> Optional[pd.Timestamp]:
    """
    Parse strings like 'YYYY-MM-DD HH-MM-SS-mmm' where mmm is milliseconds.
    Example: '2025-04-08 12-27-10-512'
    """
    m = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2})\s+(\d{2})-(\d{2})-(\d{2})-(\d{3})", ts.strip()
    )
    if not m:
        return None
    date_part, hh, mm, ss, ms = m.groups()
    yyyy, MM, dd = map(int, date_part.split("-"))
    hh, mm, ss, ms = int(hh), int(mm), int(ss), int(ms)
    # convert milliseconds (3 digits) -> microseconds (6 digits)
    micro = ms * 1000
    return pd.Timestamp(datetime(yyyy, MM, dd, hh, mm, ss, micro))

def _try_formats(ts: str, fmts: Iterable[str]) -> Optional[pd.Timestamp]:
    for fmt in fmts:
        try:
            return pd.to_datetime(ts, format=fmt)
        except Exception:
            continue
    return None


def _safe_parse_timestamp(x) -> pd.Timestamp:
    """
    Robust timestamp parser that tries multiple patterns:
    - 'YYYY-MM-DD HH:MM:SS[.ffffff]'
    - 'YYYY-MM-DD HH:MM:SS-ffffff'
    - 'YYYY-MM-DD HH-MM-SS-fff' (milliseconds; custom)
    - common variants with '/' date separators
    """
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()

    # 1) custom "HH-MM-SS-mmm" (milliseconds)
    parsed = _parse_ts_msec_style(s)
    if parsed is not None:
        return parsed

    # 2) common explicit formats
    common_formats = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S-%f",
        "%Y-%m-%d %H-%M-%S-%f",  # some logs use dashes for time & microseconds
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
    ]
    parsed = _try_formats(s, common_formats)
    if parsed is not None:
        return parsed

    # 3) best-effort fallback (dateutil) with coercion
    return pd.to_datetime(s, errors="coerce")


def load_cobot_data(filepath: str) -> pd.DataFrame:
    """
    Load cobot time-series data from CSV and robustly parse 'timestamp' column.
    Expected at least: a 'timestamp' column and one or more numeric signal columns.
    """
    df = pd.read_csv(filepath)

    # If there's no 'timestamp' column, try to build it from index or other columns.
    if "timestamp" not in df.columns:
        # If index looks like time-like string, keep as is; otherwise create a range index.
        df = df.rename(columns={df.columns[0]: "timestamp"}) if df.columns.size > 0 else df
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.RangeIndex(start=0, stop=len(df), step=1)

    # Robust parse
    df["timestamp"] = df["timestamp"].apply(_safe_parse_timestamp)

    # Report unparsed values
    nat_count = df["timestamp"].isna().sum()
    if nat_count > 0:
        sample_bad = (
            df.loc[df["timestamp"].isna(), "timestamp"]
            .astype(str)
            .head(3)
            .tolist()
        )
        print(
            f"[load_cobot_data] Warning: {nat_count} timestamp(s) could not be parsed. "
            f"Sample: {sample_bad}"
        )

    # Sort by time if possible
    if df["timestamp"].notna().any():
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df
