import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon


DEFAULT_K_VALUES = [0, 1, 2]

REQUIRED_METHODS = [
    "L0_PPG_only",
    "E_pop_only",
    "E_state_only",
    "E_local_only",
    "SA_PIF_full",
]

OUTPUT_METHODS = [
    "L0_PPG_only",
    "E_pop_only",
    "E_state_only",
    "E_local_only",
    "SA_PIF_full_v1",
    "SA_PIF_adaptive_v2",
]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 40, 400)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "n": 0,
            "MAE": np.nan,
            "MedianAE": np.nan,
            "RMSE": np.nan,
            "MARD": np.nan,
            "Bias": np.nan,
            "Pearson_r": np.nan,
        }

    abs_error = np.abs(y_pred - y_true)
    signed_error = y_pred - y_true

    if len(y_true) < 3 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        pearson_r = np.nan
    else:
        try:
            pearson_r = float(pearsonr(y_pred, y_true)[0])
        except Exception:
            pearson_r = np.nan

    return {
        "n": int(len(y_true)),
        "MAE": float(np.mean(abs_error)),
        "MedianAE": float(np.median(abs_error)),
        "RMSE": float(np.sqrt(np.mean(signed_error**2))),
        "MARD": float(np.mean(abs_error / np.maximum(y_true, 1e-9)) * 100.0),
        "Bias": float(np.mean(signed_error)),
        "Pearson_r": pearson_r,
    }


def clarke_zone(reference: float, prediction: float) -> str:
    reference = float(reference)
    prediction = float(prediction)

    if reference < 70:
        if prediction < 70:
            return "A"
        if prediction > 180:
            return "E"
        return "D"

    if reference <= 180:
        if abs(prediction - reference) / reference <= 0.20:
            return "A"
        if prediction > reference * 1.20 + 30:
            return "C"
        if prediction < reference * 0.80 - 10:
            return "C"
        return "B"

    if reference <= 290:
        if abs(prediction - reference) / reference <= 0.20:
            return "A"
        if prediction < 70:
            return "E"
        if prediction < 130:
            return "D"
        if prediction > reference * 1.20:
            return "C"
        return "B"

    if abs(prediction - reference) / reference <= 0.20:
        return "A"

    if prediction < 70:
        return "E"

    if prediction < 130:
        return "D"

    return "B"


def clarke_ab_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    zones = [clarke_zone(true, pred) for true, pred in zip(y_true, y_pred)]

    if len(zones) == 0:
        return np.nan

    return float(100.0 * sum(zone in {"A", "B"} for zone in zones) / len(zones))


def simplex_grid(step: float, allow_local: bool) -> list[tuple[float, float, float]]:
    values = np.arange(0, 1 + 1e-9, step)
    grid = []

    if allow_local:
        for weight_pop in values:
            for weight_state in values:
                weight_local = 1.0 - weight_pop - weight_state

                if weight_local < -1e-9:
                    continue

                if weight_local < 0:
                    weight_local = 0.0

                grid.append((float(weight_pop), float(weight_state), float(weight_local)))
    else:
        for weight_pop in values:
            weight_state = 1.0 - weight_pop
            grid.append((float(weight_pop), float(weight_state), 0.0))

    return grid


def predict_weighted(frame: pd.DataFrame, weights: tuple[float, float, float]) -> np.ndarray:
    weight_pop, weight_state, weight_local = weights

    predictions = (
        frame["base"].to_numpy(dtype=float)
        + weight_pop * frame["off_pop"].to_numpy(dtype=float)
        + weight_state * frame["off_state"].to_numpy(dtype=float)
        + weight_local * frame["off_local"].to_numpy(dtype=float)
    )

    return np.clip(predictions, 40, 400)


def choose_weights(
    train_frame: pd.DataFrame,
    k: int,
    grid_with_local: list[tuple[float, float, float]],
    grid_no_local: list[tuple[float, float, float]],
) -> dict:
    grid = grid_no_local if k == 0 else grid_with_local
    y_true = train_frame["true"].to_numpy(dtype=float)

    best = None

    for weights in grid:
        y_pred = predict_weighted(train_frame, weights)
        metrics = compute_metrics(y_true, y_pred)
        score = (metrics["MAE"], metrics["RMSE"])

        if best is None or score < best["score"]:
            best = {
                "weights": weights,
                "score": score,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
            }

    return best


def validate_columns(df: pd.DataFrame) -> None:
    required = {"dataset", "subject", "uid", "k", "method", "true", "base", "state", "pred"}
    missing = sorted(required - set(df.columns))

    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def load_predictions(paths: list[Path]) -> pd.DataFrame:
    frames = []

    for path in paths:
        frame = pd.read_csv(path)
        validate_columns(frame)
        frames.append(frame)

    if not frames:
        raise ValueError("No input files were provided.")

    return pd.concat(frames, ignore_index=True)


def build_pivot(df: pd.DataFrame, required_methods: list[str]) -> pd.DataFrame:
    subset = df[df["method"].isin(required_methods)].copy()

    pivot = (
        subset.pivot_table(
            index=["dataset", "subject", "uid", "k", "true", "base", "state"],
            columns="method",
            values="pred",
            aggfunc="first",
        )
        .reset_index()
    )

    pivot.columns.name = None

    for method in required_methods:
        if method not in pivot.columns:
            pivot[method] = np.nan

    pivot["E_local_only"] = pivot["E_local_only"].fillna(pivot["L0_PPG_only"])
    pivot["E_pop_only"] = pivot["E_pop_only"].fillna(pivot["L0_PPG_only"])
    pivot["E_state_only"] = pivot["E_state_only"].fillna(pivot["L0_PPG_only"])
    pivot["SA_PIF_full"] = pivot["SA_PIF_full"].fillna(pivot["L0_PPG_only"])

    pivot["off_pop"] = pivot["E_pop_only"] - pivot["base"]
    pivot["off_state"] = pivot["E_state_only"] - pivot["base"]
    pivot["off_local"] = pivot["E_local_only"] - pivot["base"]

    return pivot


def run_adaptive_fusion(
    pivot: pd.DataFrame,
    k_values: list[int],
    grid_step: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid_with_local = simplex_grid(grid_step, allow_local=True)
    grid_no_local = simplex_grid(grid_step, allow_local=False)

    prediction_rows = []
    weight_rows = []

    for dataset_name, dataset_df in pivot.groupby("dataset"):
        subjects = sorted(dataset_df["subject"].unique())

        for k in k_values:
            dataset_k = dataset_df[dataset_df["k"] == k].copy()

            if len(dataset_k) == 0:
                continue

            for test_subject in subjects:
                train_frame = dataset_k[dataset_k["subject"] != test_subject].copy()
                test_frame = dataset_k[dataset_k["subject"] == test_subject].copy()

                if len(train_frame) == 0 or len(test_frame) == 0:
                    continue

                best = choose_weights(
                    train_frame=train_frame,
                    k=k,
                    grid_with_local=grid_with_local,
                    grid_no_local=grid_no_local,
                )
                weight_pop, weight_state, weight_local = best["weights"]
                adaptive_predictions = predict_weighted(test_frame, best["weights"])

                for idx, (_, row) in enumerate(test_frame.iterrows()):
                    prediction_rows.append(
                        {
                            "dataset": dataset_name,
                            "subject": row["subject"],
                            "uid": row["uid"],
                            "state": row["state"],
                            "k": int(k),
                            "true": float(row["true"]),
                            "base": float(row["base"]),
                            "L0_PPG_only": float(row["L0_PPG_only"]),
                            "E_pop_only": float(row["E_pop_only"]),
                            "E_state_only": float(row["E_state_only"]),
                            "E_local_only": float(row["E_local_only"]),
                            "SA_PIF_full_v1": float(row["SA_PIF_full"]),
                            "SA_PIF_adaptive_v2": float(adaptive_predictions[idx]),
                            "w_pop": float(weight_pop),
                            "w_state": float(weight_state),
                            "w_local": float(weight_local),
                            "train_MAE_for_weights": float(best["MAE"]),
                            "train_RMSE_for_weights": float(best["RMSE"]),
                        }
                    )

                weight_rows.append(
                    {
                        "dataset": dataset_name,
                        "test_subject": test_subject,
                        "k": int(k),
                        "w_pop": float(weight_pop),
                        "w_state": float(weight_state),
                        "w_local": float(weight_local),
                        "train_MAE": float(best["MAE"]),
                        "train_RMSE": float(best["RMSE"]),
                        "n_train": int(len(train_frame)),
                        "n_test": int(len(test_frame)),
                    }
                )

    return pd.DataFrame(prediction_rows), pd.DataFrame(weight_rows)


def summarize_predictions(adaptive_df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    rows = []

    for (dataset_name, k), group in adaptive_df.groupby(["dataset", "k"]):
        y_true = group["true"].to_numpy(dtype=float)

        for method in methods:
            y_pred = group[method].to_numpy(dtype=float)
            metrics = compute_metrics(y_true, y_pred)
            ab_percent = clarke_ab_percent(y_true, y_pred)

            rows.append(
                {
                    "dataset": dataset_name,
                    "k": int(k),
                    "method": method,
                    "n": metrics["n"],
                    "MAE": metrics["MAE"],
                    "MedianAE": metrics["MedianAE"],
                    "RMSE": metrics["RMSE"],
                    "MARD": metrics["MARD"],
                    "Bias": metrics["Bias"],
                    "Pearson_r": metrics["Pearson_r"],
                    "Clarke_AB_percent": ab_percent,
                }
            )

    return pd.DataFrame(rows).sort_values(["dataset", "k", "MAE"])


def run_pairwise_tests(adaptive_df: pd.DataFrame, target_method: str, baselines: list[str]) -> pd.DataFrame:
    rows = []

    for (dataset_name, k), group in adaptive_df.groupby(["dataset", "k"]):
        y_true = group["true"].to_numpy(dtype=float)
        target_pred = group[target_method].to_numpy(dtype=float)
        target_error = np.abs(target_pred - y_true)

        for baseline in baselines:
            baseline_pred = group[baseline].to_numpy(dtype=float)
            baseline_error = np.abs(baseline_pred - y_true)
            diff = baseline_error - target_error
            nonzero = diff != 0

            if nonzero.sum() >= 3:
                try:
                    statistic, p_value = wilcoxon(diff[nonzero])
                except Exception:
                    statistic, p_value = np.nan, np.nan
            else:
                statistic, p_value = np.nan, np.nan

            rows.append(
                {
                    "dataset": dataset_name,
                    "k": int(k),
                    "comparison": f"{target_method} vs {baseline}",
                    "paired_n": int(len(group)),
                    "MAE_baseline": float(np.mean(baseline_error)),
                    "MAE_target": float(np.mean(target_error)),
                    "mean_MAE_reduction": float(np.mean(diff)),
                    "median_MAE_reduction": float(np.median(diff)),
                    "wilcoxon_stat": float(statistic) if np.isfinite(statistic) else np.nan,
                    "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def build_comparison_summary(
    summary_df: pd.DataFrame,
    adaptive_df: pd.DataFrame,
    methods: list[str],
    k_values: list[int],
) -> pd.DataFrame:
    rows = []

    for dataset_name in sorted(adaptive_df["dataset"].unique()):
        for k in k_values:
            subset = summary_df[(summary_df["dataset"] == dataset_name) & (summary_df["k"] == k)]

            baseline_row = subset[subset["method"] == "L0_PPG_only"]
            baseline_mae = float(baseline_row["MAE"].iloc[0]) if len(baseline_row) else np.nan

            for method in methods:
                method_row = subset[subset["method"] == method]

                if len(method_row) == 0:
                    continue

                row = method_row.iloc[0]

                rows.append(
                    {
                        "dataset": dataset_name,
                        "k": int(k),
                        "method": method,
                        "MAE": row["MAE"],
                        "RMSE": row["RMSE"],
                        "MARD": row["MARD"],
                        "Bias": row["Bias"],
                        "Pearson_r": row["Pearson_r"],
                        "Clarke_AB_percent": row["Clarke_AB_percent"],
                        "Delta_MAE_vs_L0": baseline_mae - row["MAE"]
                        if np.isfinite(baseline_mae)
                        else np.nan,
                        "Percent_Improvement_vs_L0": (
                            (baseline_mae - row["MAE"]) / baseline_mae * 100.0
                            if np.isfinite(baseline_mae) and baseline_mae > 0
                            else np.nan
                        ),
                    }
                )

    return pd.DataFrame(rows)


def plot_convergence(summary_df: pd.DataFrame, output_dir: Path, methods: list[str]) -> None:
    for dataset_name in sorted(summary_df["dataset"].unique()):
        plt.figure(figsize=(8.5, 5.2))

        for method in methods:
            subset = summary_df[
                (summary_df["dataset"] == dataset_name)
                & (summary_df["method"] == method)
            ].sort_values("k")

            if len(subset) == 0:
                continue

            plt.plot(subset["k"], subset["MAE"], marker="o", linewidth=2, label=method)

        plt.xlabel("Calibration samples k")
        plt.ylabel("MAE")
        plt.title(f"Adaptive evidence fusion: {dataset_name}")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()

        plt.savefig(output_dir / f"convergence_{dataset_name}.png", dpi=300, bbox_inches="tight")
        plt.savefig(output_dir / f"convergence_{dataset_name}.pdf", dpi=300, bbox_inches="tight")
        plt.close()


def plot_weight_summary(weight_summary: pd.DataFrame, output_dir: Path) -> None:
    for dataset_name in sorted(weight_summary["dataset"].unique()):
        subset = weight_summary[weight_summary["dataset"] == dataset_name].sort_values("k")

        plt.figure(figsize=(7.5, 4.8))

        for column in ["w_pop", "w_state", "w_local"]:
            plt.plot(
                subset["k"],
                subset[column],
                marker="o",
                linewidth=2,
                label=column.replace("w_", ""),
            )

        plt.xlabel("Calibration samples k")
        plt.ylabel("Mean learned fusion weight")
        plt.title(f"Learned fusion weights: {dataset_name}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.savefig(output_dir / f"weights_{dataset_name}.png", dpi=300, bbox_inches="tight")
        plt.savefig(output_dir / f"weights_{dataset_name}.pdf", dpi=300, bbox_inches="tight")
        plt.close()


def save_outputs(
    output_dir: Path,
    adaptive_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    tests_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    methods: list[str],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "adaptive_predictions.csv"
    weights_path = output_dir / "adaptive_weights.csv"
    summary_path = output_dir / "adaptive_summary.csv"
    tests_path = output_dir / "pairwise_tests.csv"
    comparison_path = output_dir / "method_comparison.csv"
    weight_summary_path = output_dir / "weight_summary.csv"

    adaptive_df.to_csv(predictions_path, index=False)
    weights_df.to_csv(weights_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    tests_df.to_csv(tests_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)

    weight_summary = (
        weights_df.groupby(["dataset", "k"])[["w_pop", "w_state", "w_local"]]
        .mean()
        .reset_index()
    )
    weight_summary.to_csv(weight_summary_path, index=False)

    plot_convergence(summary_df=summary_df, output_dir=output_dir, methods=methods)
    plot_weight_summary(weight_summary=weight_summary, output_dir=output_dir)

    return {
        "predictions": str(predictions_path),
        "weights": str(weights_path),
        "summary": str(summary_path),
        "pairwise_tests": str(tests_path),
        "comparison": str(comparison_path),
        "weight_summary": str(weight_summary_path),
    }


def run_pipeline(
    input_csvs: list[Path],
    output_dir: Path,
    k_values: list[int],
    grid_step: float,
) -> dict:
    raw_df = load_predictions(input_csvs)
    pivot = build_pivot(raw_df, required_methods=REQUIRED_METHODS)

    adaptive_df, weights_df = run_adaptive_fusion(
        pivot=pivot,
        k_values=k_values,
        grid_step=grid_step,
    )

    if len(adaptive_df) == 0:
        raise RuntimeError("No adaptive predictions were generated.")

    summary_df = summarize_predictions(adaptive_df, methods=OUTPUT_METHODS)

    tests_df = run_pairwise_tests(
        adaptive_df=adaptive_df,
        target_method="SA_PIF_adaptive_v2",
        baselines=["L0_PPG_only", "E_pop_only", "E_state_only", "E_local_only", "SA_PIF_full_v1"],
    )

    comparison_df = build_comparison_summary(
        summary_df=summary_df,
        adaptive_df=adaptive_df,
        methods=OUTPUT_METHODS,
        k_values=k_values,
    )

    return save_outputs(
        output_dir=output_dir,
        adaptive_df=adaptive_df,
        weights_df=weights_df,
        summary_df=summary_df,
        tests_df=tests_df,
        comparison_df=comparison_df,
        methods=OUTPUT_METHODS,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adaptive evidence fusion.")

    parser.add_argument("--input-csvs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--k-values", nargs="+", type=int, default=DEFAULT_K_VALUES)
    parser.add_argument("--grid-step", type=float, default=0.02)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_pipeline(
        input_csvs=args.input_csvs,
        output_dir=args.output_dir,
        k_values=args.k_values,
        grid_step=args.grid_step,
    )


if __name__ == "__main__":
    main()
