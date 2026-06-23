import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ALPHA = 0.05

DEFAULT_METHOD_COLUMNS = [
    "L0_base",
    "E_pop_only",
    "E_state_only",
    "E_local_only",
    "SA_PIF_inversevar",
    "SA_PIF_adaptive",
]


def holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    n_tests = len(p_values)

    if n_tests == 0:
        return np.asarray([], dtype=float)

    order = np.argsort(p_values)
    sorted_values = p_values[order]

    adjusted_sorted = np.empty(n_tests, dtype=float)

    for index, value in enumerate(sorted_values):
        adjusted_sorted[index] = (n_tests - index) * value

    adjusted_sorted = np.maximum.accumulate(adjusted_sorted)
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    adjusted = np.empty(n_tests, dtype=float)
    adjusted[order] = adjusted_sorted

    return adjusted


def paired_cliffs_delta(baseline: np.ndarray, method: np.ndarray) -> float:
    baseline = np.asarray(baseline, dtype=float)
    method = np.asarray(method, dtype=float)

    mask = np.isfinite(baseline) & np.isfinite(method)
    diff = baseline[mask] - method[mask]

    if len(diff) == 0:
        return np.nan

    return float((np.sum(diff > 0) - np.sum(diff < 0)) / len(diff))


def paired_effects(baseline: np.ndarray, method: np.ndarray) -> dict:
    baseline = np.asarray(baseline, dtype=float)
    method = np.asarray(method, dtype=float)

    mask = np.isfinite(baseline) & np.isfinite(method)
    baseline = baseline[mask]
    method = method[mask]

    diff = baseline - method

    if len(diff) == 0:
        return {
            "delta_MAE_mean": np.nan,
            "delta_MAE_median": np.nan,
            "relative_improvement_percent": np.nan,
            "cohen_dz": np.nan,
            "paired_cliffs_delta": np.nan,
            "n_improved": 0,
            "n_worse": 0,
            "n_tied": 0,
        }

    sd = np.std(diff, ddof=1)
    cohen_dz = np.nan if sd <= 1e-12 else float(np.mean(diff) / sd)

    baseline_mean = np.mean(baseline)
    relative_improvement = (
        float(np.mean(diff) / baseline_mean * 100.0)
        if np.isfinite(baseline_mean) and abs(baseline_mean) > 1e-12
        else np.nan
    )

    return {
        "delta_MAE_mean": float(np.mean(diff)),
        "delta_MAE_median": float(np.median(diff)),
        "relative_improvement_percent": relative_improvement,
        "cohen_dz": cohen_dz,
        "paired_cliffs_delta": paired_cliffs_delta(baseline, method),
        "n_improved": int(np.sum(diff > 0)),
        "n_worse": int(np.sum(diff < 0)),
        "n_tied": int(np.sum(np.isclose(diff, 0.0))),
    }


def safe_wilcoxon(baseline: np.ndarray, method: np.ndarray) -> tuple[float, float, str]:
    baseline = np.asarray(baseline, dtype=float)
    method = np.asarray(method, dtype=float)

    mask = np.isfinite(baseline) & np.isfinite(method)
    baseline = baseline[mask]
    method = method[mask]

    diff = baseline - method

    if len(diff) == 0:
        return np.nan, np.nan, "no_valid_pairs"

    if np.all(np.isclose(diff, 0.0)):
        return 0.0, 1.0, "all_differences_zero"

    try:
        statistic, p_value = wilcoxon(
            baseline,
            method,
            zero_method="wilcox",
            correction=False,
            alternative="greater",
            method="auto",
        )
    except TypeError:
        statistic, p_value = wilcoxon(
            baseline,
            method,
            zero_method="wilcox",
            correction=False,
            alternative="greater",
            mode="auto",
        )

    return float(statistic), float(p_value), "ok"


def validate_columns(df: pd.DataFrame, method_cols: list[str]) -> None:
    required = {"subject", "state", "k", "true", "repeat"} | set(method_cols)
    missing = sorted(required - set(df.columns))

    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def load_predictions(
    input_csv: Path,
    method_cols: list[str],
    k_values: list[int],
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    validate_columns(df, method_cols=method_cols)

    df = df.copy()
    df["subject"] = df["subject"].astype(str)
    df["state"] = df["state"].astype(str)
    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    df["repeat"] = pd.to_numeric(df["repeat"], errors="coerce").fillna(0).astype(int)
    df["true"] = pd.to_numeric(df["true"], errors="coerce")

    for column in method_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df[df["k"].isin(k_values) & df["true"].notna()].copy()
    df["k"] = df["k"].astype(int)

    if len(df) == 0:
        raise ValueError("No valid rows remain after filtering.")

    return df.reset_index(drop=True)


def wide_to_long(df: pd.DataFrame, method_cols: list[str]) -> pd.DataFrame:
    long_df = df.melt(
        id_vars=["subject", "state", "k", "true", "repeat"],
        value_vars=method_cols,
        var_name="method",
        value_name="prediction",
    )

    long_df = long_df[long_df["prediction"].notna()].copy()
    long_df["abs_error"] = np.abs(long_df["prediction"] - long_df["true"])
    long_df["sq_error"] = (long_df["prediction"] - long_df["true"]) ** 2

    return long_df.reset_index(drop=True)


def compute_subject_level_errors(long_df: pd.DataFrame) -> pd.DataFrame:
    return (
        long_df.groupby(["repeat", "k", "method", "subject"], as_index=False)
        .agg(
            subject_MAE=("abs_error", "mean"),
            subject_RMSE=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
            n_windows=("abs_error", "size"),
        )
        .reset_index(drop=True)
    )


def run_pairwise(
    subject_level: pd.DataFrame,
    k: int,
    method: str,
    comparator: str,
) -> list[dict]:
    rows = []

    for repeat in sorted(subject_level["repeat"].unique()):
        subset = subject_level[
            (subject_level["repeat"] == repeat)
            & (subject_level["k"] == k)
        ].copy()

        method_df = subset[subset["method"] == method][["subject", "subject_MAE"]].rename(
            columns={"subject_MAE": "mae_method"}
        )
        comparator_df = subset[subset["method"] == comparator][["subject", "subject_MAE"]].rename(
            columns={"subject_MAE": "mae_comparator"}
        )

        merged = pd.merge(method_df, comparator_df, on="subject", how="inner")

        if len(merged) == 0:
            continue

        method_mae = merged["mae_method"].to_numpy(dtype=float)
        comparator_mae = merged["mae_comparator"].to_numpy(dtype=float)

        statistic, p_value, status = safe_wilcoxon(comparator_mae, method_mae)
        effects = paired_effects(comparator_mae, method_mae)

        rows.append(
            {
                "repeat": int(repeat),
                "k": int(k),
                "comparison": f"{method} vs {comparator}",
                "method": method,
                "comparator": comparator,
                "n_subjects": int(len(merged)),
                "method_subject_MAE_mean": float(np.mean(method_mae)),
                "comparator_subject_MAE_mean": float(np.mean(comparator_mae)),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "status": status,
                **effects,
            }
        )

    if not rows:
        raise ValueError(f"No paired rows found for k={k}, method={method}, comparator={comparator}.")

    return rows


def run_tests(
    subject_level: pd.DataFrame,
    main_method: str,
    base_method: str,
    main_k_values: list[int],
    extra_comparisons: list[tuple[str, str]],
) -> pd.DataFrame:
    rows = []

    for k in main_k_values:
        rows.extend(
            run_pairwise(
                subject_level=subject_level,
                k=k,
                method=main_method,
                comparator=base_method,
            )
        )

    for method, comparator in extra_comparisons:
        for k in main_k_values:
            rows.extend(
                run_pairwise(
                    subject_level=subject_level,
                    k=k,
                    method=method,
                    comparator=comparator,
                )
            )

    return pd.DataFrame(rows)


def aggregate_results(
    per_repeat: pd.DataFrame,
    main_method: str,
    base_method: str,
    alpha: float,
) -> pd.DataFrame:
    agg = (
        per_repeat.groupby(["k", "comparison", "method", "comparator"], as_index=False)
        .agg(
            repeats=("repeat", "nunique"),
            n_subjects_mean=("n_subjects", "mean"),
            method_subject_MAE_mean=("method_subject_MAE_mean", "mean"),
            method_subject_MAE_std=("method_subject_MAE_mean", "std"),
            comparator_subject_MAE_mean=("comparator_subject_MAE_mean", "mean"),
            comparator_subject_MAE_std=("comparator_subject_MAE_mean", "std"),
            delta_MAE_mean=("delta_MAE_mean", "mean"),
            delta_MAE_std=("delta_MAE_mean", "std"),
            delta_MAE_median=("delta_MAE_median", "median"),
            relative_improvement_percent_mean=("relative_improvement_percent", "mean"),
            cohen_dz_mean=("cohen_dz", "mean"),
            paired_cliffs_delta_mean=("paired_cliffs_delta", "mean"),
            n_improved_mean=("n_improved", "mean"),
            n_worse_mean=("n_worse", "mean"),
            n_tied_mean=("n_tied", "mean"),
            p_value_median=("p_value", "median"),
            p_value_min=("p_value", "min"),
            p_value_max=("p_value", "max"),
        )
    )

    main_label = f"{main_method} vs {base_method}"
    main_mask = agg["comparison"] == main_label

    agg["p_value_holm_main"] = np.nan

    main_p_values = agg.loc[main_mask, "p_value_median"].to_numpy(dtype=float)

    if len(main_p_values) > 0:
        agg.loc[main_mask, "p_value_holm_main"] = holm_bonferroni(main_p_values)

    agg["significant_p05_median"] = agg["p_value_median"] < alpha
    agg["significant_p05_holm_main"] = agg["p_value_holm_main"] < alpha

    return agg.sort_values(["comparison", "k"]).reset_index(drop=True)


def build_main_summary(
    aggregate_df: pd.DataFrame,
    main_method: str,
    base_method: str,
) -> pd.DataFrame:
    main_label = f"{main_method} vs {base_method}"
    main_results = aggregate_df[aggregate_df["comparison"] == main_label].sort_values("k").copy()

    summary = main_results[
        [
            "k",
            "repeats",
            "n_subjects_mean",
            "comparator_subject_MAE_mean",
            "method_subject_MAE_mean",
            "delta_MAE_mean",
            "relative_improvement_percent_mean",
            "p_value_median",
            "p_value_holm_main",
            "cohen_dz_mean",
            "paired_cliffs_delta_mean",
            "n_improved_mean",
            "n_worse_mean",
        ]
    ].copy()

    return summary.rename(
        columns={
            "comparator_subject_MAE_mean": "base_subject_MAE",
            "method_subject_MAE_mean": "method_subject_MAE",
            "delta_MAE_mean": "delta_MAE",
            "relative_improvement_percent_mean": "improvement_percent",
            "p_value_median": "wilcoxon_p_median",
            "p_value_holm_main": "wilcoxon_p_holm",
            "cohen_dz_mean": "cohen_dz",
            "paired_cliffs_delta_mean": "paired_cliffs_delta",
        }
    )


def write_text_summary(
    summary_df: pd.DataFrame,
    output_path: Path,
    main_method: str,
    base_method: str,
) -> None:
    lines = [
        "Subject-level paired Wilcoxon signed-rank tests",
        "Statistical unit: subject-level MAE.",
        f"Alternative hypothesis: {base_method} subject MAE > {main_method} subject MAE.",
        "",
    ]

    for _, row in summary_df.iterrows():
        lines.append(
            f"k={int(row['k'])}: {main_method} reduced subject-level MAE from "
            f"{row['base_subject_MAE']:.3f} to {row['method_subject_MAE']:.3f} "
            f"(delta={row['delta_MAE']:.3f}; "
            f"{row['improvement_percent']:.1f}%). "
            f"Wilcoxon median p={row['wilcoxon_p_median']:.3e}, "
            f"Holm-adjusted p={row['wilcoxon_p_holm']:.3e}; "
            f"Cohen dz={row['cohen_dz']:.3f}; "
            f"paired Cliff's delta={row['paired_cliffs_delta']:.3f}."
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(
    input_csv: Path,
    output_dir: Path,
    method_cols: list[str],
    main_method: str,
    base_method: str,
    main_k_values: list[int],
    filter_k_values: list[int],
    extra_comparisons: list[tuple[str, str]],
    alpha: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    wide_df = load_predictions(
        input_csv=input_csv,
        method_cols=method_cols,
        k_values=filter_k_values,
    )

    long_df = wide_to_long(wide_df, method_cols=method_cols)
    long_path = output_dir / "predictions_long.csv"
    long_df.to_csv(long_path, index=False)

    subject_level = compute_subject_level_errors(long_df)
    subject_mae_path = output_dir / "subject_level_errors.csv"
    subject_level.to_csv(subject_mae_path, index=False)

    per_repeat = run_tests(
        subject_level=subject_level,
        main_method=main_method,
        base_method=base_method,
        main_k_values=main_k_values,
        extra_comparisons=extra_comparisons,
    )
    per_repeat_path = output_dir / "wilcoxon_per_repeat.csv"
    per_repeat.to_csv(per_repeat_path, index=False)

    aggregate_df = aggregate_results(
        per_repeat=per_repeat,
        main_method=main_method,
        base_method=base_method,
        alpha=alpha,
    )
    aggregate_path = output_dir / "wilcoxon_summary.csv"
    aggregate_df.to_csv(aggregate_path, index=False)

    main_summary = build_main_summary(
        aggregate_df=aggregate_df,
        main_method=main_method,
        base_method=base_method,
    )
    main_summary_path = output_dir / "main_comparison_summary.csv"
    main_summary.to_csv(main_summary_path, index=False)

    text_summary_path = output_dir / "summary.txt"
    write_text_summary(
        summary_df=main_summary,
        output_path=text_summary_path,
        main_method=main_method,
        base_method=base_method,
    )

    return {
        "predictions_long": str(long_path),
        "subject_level_errors": str(subject_mae_path),
        "wilcoxon_per_repeat": str(per_repeat_path),
        "wilcoxon_summary": str(aggregate_path),
        "main_comparison_summary": str(main_summary_path),
        "text_summary": str(text_summary_path),
    }


def parse_comparisons(values: list[str]) -> list[tuple[str, str]]:
    comparisons = []

    for value in values:
        if ":" not in value:
            raise ValueError("Each comparison must use the format method:comparator.")

        method, comparator = value.split(":", 1)
        comparisons.append((method.strip(), comparator.strip()))

    return comparisons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run subject-level paired Wilcoxon tests.")

    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)

    parser.add_argument("--main-method", default="SA_PIF_adaptive")
    parser.add_argument("--base-method", default="L0_base")

    parser.add_argument("--method-cols", nargs="+", default=DEFAULT_METHOD_COLUMNS)
    parser.add_argument("--main-k-values", nargs="+", type=int, default=[1, 2, 5])
    parser.add_argument("--filter-k-values", nargs="+", type=int, default=[0, 1, 2, 5])

    parser.add_argument(
        "--extra-comparisons",
        nargs="*",
        default=["SA_PIF_adaptive:E_local_only", "SA_PIF_adaptive:SA_PIF_inversevar"],
    )

    parser.add_argument("--alpha", type=float, default=ALPHA)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_pipeline(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        method_cols=args.method_cols,
        main_method=args.main_method,
        base_method=args.base_method,
        main_k_values=args.main_k_values,
        filter_k_values=args.filter_k_values,
        extra_comparisons=parse_comparisons(args.extra_comparisons),
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
