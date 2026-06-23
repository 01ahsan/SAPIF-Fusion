import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if not np.any(mask):
        return np.nan

    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def format_p_value(p_value: float) -> str:
    if pd.isna(p_value):
        return "NA"

    if p_value < 1e-12:
        return "<1.0e-12"

    return f"{p_value:.3e}"


def subject_level_mae_pairs(
    df: pd.DataFrame,
    baseline_col: str,
    method_col: str,
) -> pd.DataFrame:
    rows = []

    for subject, group in df.groupby("subject"):
        rows.append(
            {
                "subject": subject,
                "baseline_MAE": mae(group["true"], group[baseline_col]),
                "method_MAE": mae(group["true"], group[method_col]),
            }
        )

    return pd.DataFrame(rows)


def signed_rank_test(diff: np.ndarray) -> tuple[float, int, str]:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    diff_nonzero = diff[np.abs(diff) > 1e-12]

    if len(diff_nonzero) <= 3:
        return np.nan, int(len(diff_nonzero)), "underpowered"

    _, p_value = wilcoxon(
        diff_nonzero,
        alternative="two-sided",
        zero_method="wilcox",
    )

    return float(p_value), int(len(diff_nonzero)), "ok"


def detect_iv_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in df.columns:
            return column

    raise KeyError(f"Corrected IV column not found. Expected one of: {candidates}")


def validate_columns(df: pd.DataFrame, required_cols: set[str]) -> None:
    missing = sorted(required_cols - set(df.columns))

    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def load_predictions(input_csv: Path, method_col: str, iv_candidates: list[str]) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(input_csv)
    iv_col = detect_iv_column(df, iv_candidates)

    required_cols = {
        "dataset",
        "subject",
        "k",
        "true",
        "L0_PPG_only",
        "E_local_only",
        method_col,
        "w_pop",
        "w_state",
        "w_local",
        iv_col,
    }
    validate_columns(df, required_cols)

    numeric_cols = [
        "k",
        "true",
        "L0_PPG_only",
        "E_local_only",
        method_col,
        "w_pop",
        "w_state",
        "w_local",
        iv_col,
    ]

    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["dataset"] = df["dataset"].astype(str)
    df["subject"] = df["subject"].astype(str)

    return df, iv_col


def build_weight_table(df: pd.DataFrame) -> pd.DataFrame:
    subject_weights = (
        df.groupby(["dataset", "k", "subject"], as_index=False)
        .agg(
            w_pop_subject=("w_pop", "mean"),
            w_state_subject=("w_state", "mean"),
            w_local_subject=("w_local", "mean"),
        )
    )

    table = (
        subject_weights.groupby(["dataset", "k"], as_index=False)
        .agg(
            n_subjects=("subject", "nunique"),
            population_weight=("w_pop_subject", "mean"),
            state_weight=("w_state_subject", "mean"),
            local_weight=("w_local_subject", "mean"),
            population_weight_sd=("w_pop_subject", "std"),
            state_weight_sd=("w_state_subject", "std"),
            local_weight_sd=("w_local_subject", "std"),
        )
    )

    return table.sort_values(["dataset", "k"]).reset_index(drop=True)


def plot_weight_table(table: pd.DataFrame, output_dir: Path) -> None:
    for dataset in sorted(table["dataset"].unique()):
        subset = table[table["dataset"] == dataset].sort_values("k")

        fig, ax = plt.subplots(figsize=(5.8, 4.2))

        ax.plot(subset["k"], subset["population_weight"], marker="o", linewidth=2, label="population")
        ax.plot(subset["k"], subset["state_weight"], marker="s", linewidth=2, label="state")
        ax.plot(subset["k"], subset["local_weight"], marker="^", linewidth=2, label="local")

        ax.set_title(f"Learned reliability weights: {dataset}")
        ax.set_xlabel("Calibration samples per subject")
        ax.set_ylabel("Mean subject-level reliability weight")
        ax.set_xticks(sorted(subset["k"].unique()))
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=True)

        plt.tight_layout()
        plt.savefig(output_dir / f"learned_reliability_weights_{dataset}.png", bbox_inches="tight", dpi=300)
        plt.savefig(output_dir / f"learned_reliability_weights_{dataset}.pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)


def parse_comparisons(values: list[str], iv_col: str) -> list[tuple[str, int, str, str]]:
    comparisons = []

    for value in values:
        parts = value.split(":", 3)

        if len(parts) != 4:
            raise ValueError("Each comparison must use dataset:k:label:baseline_col.")

        dataset, k_text, label, baseline_col = parts
        baseline_col = iv_col if baseline_col == "IV" else baseline_col
        comparisons.append((dataset, int(k_text), label, baseline_col))

    return comparisons


def build_wilcoxon_table(
    df: pd.DataFrame,
    comparisons: list[tuple[str, int, str, str]],
    method_col: str,
) -> pd.DataFrame:
    records = []

    for dataset, k, label, baseline_col in comparisons:
        subset = df[(df["dataset"] == dataset) & (df["k"] == k)].copy()

        if len(subset) == 0:
            continue

        pairs = subject_level_mae_pairs(
            df=subset,
            baseline_col=baseline_col,
            method_col=method_col,
        )
        pairs["diff_baseline_minus_method"] = pairs["baseline_MAE"] - pairs["method_MAE"]

        p_subject, n_nonzero, status = signed_rank_test(pairs["diff_baseline_minus_method"])

        sample_diff = (
            np.abs(subset["true"] - subset[baseline_col])
            - np.abs(subset["true"] - subset[method_col])
        )
        p_sample, _, _ = signed_rank_test(sample_diff)

        records.append(
            {
                "dataset": dataset,
                "k": int(k),
                "comparison": label,
                "baseline_column": baseline_col,
                "n_subjects": int(pairs["subject"].nunique()),
                "n_nonzero_subject_pairs": int(n_nonzero),
                "baseline_MAE_global": mae(subset["true"], subset[baseline_col]),
                "method_MAE_global": mae(subset["true"], subset[method_col]),
                "mean_delta_MAE_subject_level": float(pairs["diff_baseline_minus_method"].mean()),
                "median_delta_MAE_subject_level": float(pairs["diff_baseline_minus_method"].median()),
                "p_subject_level": p_subject,
                "p_subject_level_formatted": format_p_value(p_subject),
                "significant_subject_level": bool(p_subject < 0.05) if np.isfinite(p_subject) else False,
                "test_status": status,
                "p_sample_level_audit_only": p_sample,
                "p_sample_level_formatted_audit_only": format_p_value(p_sample),
            }
        )

    return pd.DataFrame(records)


def run_pipeline(
    input_csv: Path,
    output_dir: Path,
    method_col: str,
    iv_candidates: list[str],
    comparisons_raw: list[str],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    df, iv_col = load_predictions(
        input_csv=input_csv,
        method_col=method_col,
        iv_candidates=iv_candidates,
    )

    weight_table = build_weight_table(df)
    weight_table_path = output_dir / "learned_weights.csv"
    weight_table.to_csv(weight_table_path, index=False)

    plot_weight_table(weight_table, output_dir=output_dir)

    comparisons = parse_comparisons(comparisons_raw, iv_col=iv_col)

    wilcoxon_table = build_wilcoxon_table(
        df=df,
        comparisons=comparisons,
        method_col=method_col,
    )
    wilcoxon_path = output_dir / "subject_level_wilcoxon.csv"
    wilcoxon_table.to_csv(wilcoxon_path, index=False)

    return {
        "weights": str(weight_table_path),
        "wilcoxon": str(wilcoxon_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create learned-weight and subject-level Wilcoxon audit tables.")

    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)

    parser.add_argument("--method-col", default="SA_PIF_adaptive_v2")
    parser.add_argument(
        "--iv-candidates",
        nargs="+",
        default=[
            "SA_PIF_IV_C1_M1_C2",
            "corrected_iv_prediction",
            "SA_PIF_IV_corrected",
            "SA_PIF_inverse_variance_corrected",
        ],
    )

    parser.add_argument(
        "--comparisons",
        nargs="+",
        default=[
            "EXTERNAL20:1:SA-PIF vs PPG-only:L0_PPG_only",
            "EXTERNAL20:1:SA-PIF vs Local-only:E_local_only",
            "EXTERNAL20:1:SA-PIF vs Fixed SA-PIF:IV",
            "EXTERNAL20:2:SA-PIF vs PPG-only:L0_PPG_only",
            "EXTERNAL20:2:SA-PIF vs Local-only:E_local_only",
            "EXTERNAL20:2:SA-PIF vs Fixed SA-PIF:IV",
            "MUST:1:SA-PIF vs PPG-only:L0_PPG_only",
            "MUST:1:SA-PIF vs Local-only:E_local_only",
            "MUST:2:SA-PIF vs PPG-only:L0_PPG_only",
            "MUST:2:SA-PIF vs Fixed SA-PIF:IV",
        ],
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_pipeline(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        method_col=args.method_col,
        iv_candidates=args.iv_candidates,
        comparisons_raw=args.comparisons,
    )


if __name__ == "__main__":
    main()
