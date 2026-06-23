import numpy as np
import pandas as pd


def compute_state_mae(
    df,
    dataset="EXTERNAL20",
    k=2,
    states=("Fasting", "Random", "Postprandial"),
    base_col="L0_PPG_only",
    method_col="SA_PIF_adaptive_v2",
):
    sub = df[(df["dataset"] == dataset) & (df["k"] == k)].copy()

    rows = []

    for state in states:
        state_df = sub[sub["state"] == state]

        rows.append({
            "state": state,
            "n": len(state_df),
            "L0_MAE": np.mean(np.abs(state_df["true"] - state_df[base_col])),
            "SA_PIF_MAE": np.mean(np.abs(state_df["true"] - state_df[method_col])),
        })

    state_mae = pd.DataFrame(rows)

    overall = pd.DataFrame([{
        "dataset": dataset,
        "k": k,
        "n": len(sub),
        "L0_MAE": np.mean(np.abs(sub["true"] - sub[base_col])),
        "SA_PIF_MAE": np.mean(np.abs(sub["true"] - sub[method_col])),
    }])

    return state_mae, overall