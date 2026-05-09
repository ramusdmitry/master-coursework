from __future__ import annotations

import pandas as pd


def build_basic_features(df: pd.DataFrame, target_col: str, ma_windows: list[int], returns_window: int) -> pd.DataFrame:
    out = df.copy()
    out['ret'] = out[target_col].pct_change(returns_window)
    for w in ma_windows:
        out[f'ma_{w}'] = out[target_col].rolling(w).mean()
    return out.dropna().reset_index(drop=True)
