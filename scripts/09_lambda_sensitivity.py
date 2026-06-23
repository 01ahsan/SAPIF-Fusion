import argparse
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import kurtosis
except ImportError:
    kurtosis = None
    warnings.warn("scipy.stats.kurtosis is unavailable. Kurtosis inflation will be skipped.")


EPS = 1e-8


def robust_var(values: np.ndarray, eps: float = EPS) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) <= 1:
        return eps

    median = np.median(values)
    mad = np.median(np.abs(values - median))
    variance = (1.4826 * mad) ** 2

    if not np.isfinite(variance) or variance < eps:
        variance = np.var(values, ddof=1) if len(values) > 1 else eps

    if not np.isfinite(variance) or variance < eps:
        variance = eps

    return float(variance)


def excess_kurtosis(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 4 or kurtosis is None:
        return 0.0

    value = kurtosis(values, fisher=True, bias=False)

    if not np.isfinite(value):
        return 0.0

    return float(value)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if not np.any(mask):
        return np.nan

    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if not np.any(mask):
        return np.nan

    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mard(y_true: np.ndarray, y_pred: np.ndarray, eps: float = EPS) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if not np.any(mask):
        return np.nan

    return float(
        np.mean(np.abs((y_pred[mask] - y_true[mask]) / np.maximum(np.abs(y_true[mask]), eps))) * 100.0
    )


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if not np.any(mask):
        return np.nan

    return float(np.mean(y_pred[mask] - y_true[mask]))


def clarke_zone(reference: float, prediction: float) -> str:
    reference = float(reference)
    prediction = float(prediction)

    if (reference < 70 and prediction < 70) or (
        reference >= 70 and abs(prediction - reference) <= 0.20 * reference
    ):
        return "A"

    if (reference < 70 and prediction > 180) or (reference > 180 and prediction < 70):
        return "E"

    if 70 <= reference <= 290 and prediction >= reference + 110:
        return "C"

    if 130 <= reference <= 180 and prediction <= (7.0 / 5.0) * reference - 182:
        return "C"

    if reference >= 240 and 70 <= prediction <= 180:
        return "D"

    if reference <= 70 and 70 <= prediction <= 180:
        return "D"

    return "B"


def clarke_ab_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    zones = [clarke_zone(reference, prediction) for reference, prediction in zip(y_true, y_pred)]

    if len(zones) == 0:
        return np.nan

    return float(100.0 * pd.Series(zones).isin(["A", "B"]).mean())


def build_priors(train_df: pd.DataFrame, base_col: str, min_state_samples: int) -> dict:
    train_df = train_df.copy()
    train_df["residual"] = train_df["true"] - train_df[base_col]

    residuals = train_df["residual"].to_numpy(dtype=float)

    mu_population = float(np.median(residuals))
    sigma_w2 = robust_var(residuals)

    subject_medians = (
        train_df.groupby("subject")["residual"]
        .median()
        .to_numpy(dtype=float)
    )
    sigma_b2 = robust_var(subject_medians)
    sigma_p2 = sigma_b2 + sigma_w2

    state_priors = {}

    for state, group in train_df.groupby("state"):
        state_residuals = group["residual"].to_numpy(dtype=float)

        if len(state_residuals) >= min_state_samples:
            mu_state = float(np.median(state_residuals))
            kappa = excess_kurtosis(state_residuals)
            sigma_s2 = robust_var(state_residuals) + sigma_w2

            inflation = 1.0

            if kappa > 1.0:
                inflation = 1.0 + kappa / 2.0
                sigma_s2 *= inflation

            state_priors[str(state)] = {
                "mu_state": mu_state,
                "sigma_s2": float(max(sigma_s2, EPS)),
                "n_state_train": int(len(state_residuals)),
                "kurtosis": float(kappa),
                "inflation": float(inflation),
            }
        else:
            state_priors[str(state)] = {
                "mu_state": mu_population,
                "sigma_s2": float(max(10.0 * sigma_p2, EPS)),
                "n_state_train": int(len(state_residuals)),
                "kurtosis": 0.0,
                "inflation": 1.0,
            }

    return {
        "mu_population": mu_population,
        "sigma_w2": float(max(sigma_w2, EPS)),
        "sigma_b2": float(max(sigma_b2, EPS)),
        "sigma_p2": float(max(sigma_p2, EPS)),
        "state_priors": state_priors,
    }


def predict_for_lambda(
    row: pd.Series,
    priors: dict,
    lambda_value: float,
    base_col: str,
    local_col: str,
    min_prediction: float,
    max_prediction: float,
) -> dict:
    state = str(row["state"])
    k = int(row["k"])

    base_prediction = float(row[base_col])
    mu_population = float(priors["mu_population"])

    state_prior = priors["state_priors"].get(
        state,
        {
            "mu_state": mu_population,
            "sigma_s2": float(max(10.0 * priors["sigma_p2"], EPS)),
            "n_state_train": 0,
            "kurtosis": 0.0,
            "inflation": 1.0,
        },
    )

    mu_state = float(state_prior["mu_state"])
    sigma_s2 = float(max(state_prior["sigma_s2"], EPS))

    if k == 0:
        alpha_state = 1.0
        alpha_local = 0.0
        mu_local = np.nan
        sigma_l2 = np.nan
        delta = mu_state
    else:
        if pd.isna(row[local_col]):
            raise ValueError(f"{local_col} is missing for k > 0.")

        mu_local = float(row[local_col]) - base_prediction
        sigma_l2 = (priors["sigma_w2"] / max(k, 1)) + (lambda_value * priors["sigma_w2"])
        sigma_l2 = float(max(sigma_l2, EPS))

        tau_state = 1.0 / sigma_s2
        tau_local = 1.0 / sigma_l2
        denominator = tau_state + tau_local

        alpha_state = tau_state / denominator
        alpha_local = tau_local / denominator

        delta = (
            mu_population
            + alpha_state * (mu_state - mu_population)
            + alpha_local * (mu_local - mu_population)
        )

    prediction = float(np.clip(base_prediction + delta, min_prediction, max_prediction))

    return {
        "prediction": prediction,
        "delta": float(delta),
        "mu_population": float(mu_population),
        "mu_state": float(mu_state),
        "mu_local": float(mu_local) if np.isfinite(mu_local) else np.nan,
        "sigma_w2": float(priors["sigma_w2"]),
        "sigma_b2": float(priors["sigma_b2"]),
        "sigma_p2": float(priors["sigma_p2"]),
        "sigma_s2": float(sigma_s2),
        "sigma_l2": float(sigma_l2) if np.isfinite(sigma_l2) else np.nan,
        "alpha_state": float(alpha_state),
        "alpha_local": float(alpha_local),
        "state_kurtosis": float(state_prior["kurtosis"]),
        "state_inflation": float(state_prior["inflation"]),
        "n_state_train": int(state_prior["n_state_train"]),
    }


def validate_columns(df: pd.DataFrame, base_col: str, local_col: str) -> None:
    required_cols = {"dataset", "subject", "state", "k", "true", base_col, local_col}
    missing = sorted(required_cols - set(df.columns))

    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def load_predictions(
    input_csv: Path,
    base_col: str,
    local_col: str,
    datasets: list[str] | None,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    validate_columns(df, base_col=base_col, local_col=local_col)

    df = df.copy()
    df["dataset"] = df["dataset"].astype(str)
    df["subject"] = df["subject"].astype(str)
    df["state"] = df["state"].astype(str)
    df["k"] = pd.to_numeric(df["k"], errors="coerce").astype("Int64")

    for column in ["true", base_col, local_col]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["dataset", "subject", "state", "k", "true", base_col]).copy()
    df["k"] = df["k"].astype(int)

    if datasets:
        df = df[df["dataset"].isin(datasets)].copy()

    if len(df) == 0:
        raise ValueError("No valid rows found after loading and filtering input data.")

    return df.reset_index(drop=True)


def run_lambda_sensitivity(
    df: pd.DataFrame,
    lambda_grid: list[float],
    reference_lambda: float,
    threshold: float,
    base_col: str,
    local_col: str,
    min_state_samples: int,
    min_prediction: float,
    max_prediction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_records = []
    audit_records = []

    for dataset in sorted(df["dataset"].unique()):
        dataset_df = df[df["dataset"] == dataset].copy()

        for lambda_value in lambda_grid:
            for subject, subject_df in dataset_df.groupby("subject"):
                train_df = dataset_df[dataset_df["subject"] != subject].copy()
                test_df = subject_df.copy()

                if len(train_df) == 0:
                    continue

                priors = build_priors(
                    train_df=train_df,
                    base_col=base_col,
                    min_state_samples=min_state_samples,
                )

                for _, row in test_df.iterrows():
                    prediction_info = predict_for_lambda(
                        row=row,
                        priors=priors,
                        lambda_value=lambda_value,
                        base_col=base_col,
                        local_col=local_col,
                        min_prediction=min_prediction,
                        max_prediction=max_prediction,
                    )

                    prediction_records.append(
                        {
                            "dataset": dataset,
                            "subject": subject,
                            "uid": row["uid"] if "uid" in row.index else np.nan,
                            "state": row["state"],
                            "k": int(row["k"]),
                            "lambda": float(lambda_value),
                            "true": float(row["true"]),
                            "base_prediction": float(row[base_col]),
                            "local_prediction": float(row[local_col]) if not pd.isna(row[local_col]) else np.nan,
                            "prediction": prediction_info["prediction"],
                            "delta": prediction_info["delta"],
                            "mu_population": prediction_info["mu_population"],
                            "mu_state": prediction_info["mu_state"],
                            "mu_local": prediction_info["mu_local"],
                            "sigma_w2": prediction_info["sigma_w2"],
                            "sigma_b2": prediction_info["sigma_b2"],
                            "sigma_p2": prediction_info["sigma_p2"],
                            "sigma_s2": prediction_info["sigma_s2"],
                            "sigma_l2": prediction_info["sigma_l2"],
                            "alpha_state": prediction_info["alpha_state"],
                            "alpha_local": prediction_info["alpha_local"],
                            "state_kurtosis": prediction_info["state_kurtosis"],
                            "state_inflation": prediction_info["state_inflation"],
                            "n_state_train": prediction_info["n_state_train"],
                        }
                    )

            lambda_df = pd.DataFrame(
                [
                    record
                    for record in prediction_records
                    if record["dataset"] == dataset and record["lambda"] == lambda_value
                ]
            )

            for k, group in lambda_df.groupby("k"):
                y_true = group["true"].to_numpy(dtype=float)
                y_pred = group["prediction"].to_numpy(dtype=float)

                audit_records.append(
                    {
                        "dataset": dataset,
                        "k": int(k),
                        "lambda": float(lambda_value),
                        "n_rows": int(len(group)),
                        "n_subjects": int(group["subject"].nunique()),
                        "MAE": mae(y_true, y_pred),
                        "RMSE": rmse(y_true, y_pred),
                        "MARD": mard(y_true, y_pred),
                        "Bias": bias(y_true, y_pred),
                        "Clarke_AB_percent": clarke_ab_percent(y_true, y_pred),
                        "mean_alpha_state": float(np.nanmean(group["alpha_state"])),
                        "mean_alpha_local": float(np.nanmean(group["alpha_local"])),
                        "mean_sigma_s2": float(np.nanmean(group["sigma_s2"])),
                        "mean_sigma_l2": float(np.nanmean(group["sigma_l2"])),
                    }
                )

    predictions_df = pd.DataFrame(prediction_records)
    audit_df = pd.DataFrame(audit_records)

    reference = (
        audit_df[audit_df["lambda"] == reference_lambda][["dataset", "k", "MAE"]]
        .rename(columns={"MAE": "MAE_at_reference_lambda"})
    )

    audit_with_reference = audit_df.merge(reference, on=["dataset", "k"], how="left")
    audit_with_reference["delta_MAE_vs_reference"] = (
        audit_with_reference["MAE"] - audit_with_reference["MAE_at_reference_lambda"]
    )
    audit_with_reference["abs_delta_MAE_vs_reference"] = (
        audit_with_reference["delta_MAE_vs_reference"].abs()
    )

    summary_records = []

    for (dataset, k), group in audit_with_reference.groupby(["dataset", "k"]):
        group = group.sort_values("lambda")
        reference_mae = float(group["MAE_at_reference_lambda"].iloc[0])
        min_mae = float(group["MAE"].min())
        max_mae = float(group["MAE"].max())
        max_abs_delta = float(group["abs_delta_MAE_vs_reference"].max())

        summary_records.append(
            {
                "dataset": dataset,
                "k": int(k),
                "lambda_grid": ",".join(str(value) for value in lambda_grid),
                "reference_lambda": float(reference_lambda),
                "MAE_at_reference_lambda": reference_mae,
                "MAE_min": min_mae,
                "MAE_max": max_mae,
                "MAE_range": max_mae - min_mae,
                "max_abs_delta_MAE_vs_reference": max_abs_delta,
                "lambda_best": float(group.loc[group["MAE"].idxmin(), "lambda"]),
                "lambda_worst": float(group.loc[group["MAE"].idxmax(), "lambda"]),
                "stable_under_threshold": bool(max_abs_delta <= threshold),
                "n_subjects": int(group["n_subjects"].iloc[0]),
                "n_rows": int(group["n_rows"].iloc[0]),
            }
        )

    summary_df = pd.DataFrame(summary_records)

    return predictions_df, audit_with_reference, summary_df


def plot_sensitivity(
    audit_df: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    reference_lambda: float,
) -> None:
    if len(audit_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    for (dataset, k), group in audit_df.groupby(["dataset", "k"]):
        group = group.sort_values("lambda")
        ax.plot(
            group["lambda"],
            group["MAE"],
            marker="o",
            linewidth=2,
            markersize=5,
            label=f"{dataset}, k={k}",
        )

    ax.axvline(reference_lambda, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Local uncertainty coefficient")
    ax.set_ylabel("MAE")
    ax.set_title("Lambda sensitivity")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(output_png, bbox_inches="tight", dpi=300)
    plt.savefig(output_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)


def save_outputs(
    output_dir: Path,
    predictions_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    reference_lambda: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "lambda_sensitivity_predictions.csv"
    metrics_path = output_dir / "lambda_sensitivity_metrics.csv"
    summary_path = output_dir / "lambda_sensitivity_summary.csv"
    figure_png = output_dir / "lambda_sensitivity.png"
    figure_pdf = output_dir / "lambda_sensitivity.pdf"

    predictions_df.to_csv(predictions_path, index=False)
    audit_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    plot_sensitivity(
        audit_df=audit_df,
        output_png=figure_png,
        output_pdf=figure_pdf,
        reference_lambda=reference_lambda,
    )

    return {
        "predictions": str(predictions_path),
        "metrics": str(metrics_path),
        "summary": str(summary_path),
        "figure_png": str(figure_png),
        "figure_pdf": str(figure_pdf),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lambda sensitivity analysis.")

    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)

    parser.add_argument("--base-col", default="L0_PPG_only")
    parser.add_argument("--local-col", default="E_local_only")
    parser.add_argument("--datasets", nargs="*", default=None)

    parser.add_argument(
        "--lambda-grid",
        nargs="+",
        type=float,
        default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    )
    parser.add_argument("--reference-lambda", type=float, default=1.0)
    parser.add_argument("--stability-threshold", type=float, default=0.30)

    parser.add_argument("--min-state-samples", type=int, default=5)
    parser.add_argument("--min-prediction", type=float, default=40.0)
    parser.add_argument("--max-prediction", type=float, default=400.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_predictions(
        input_csv=args.input_csv,
        base_col=args.base_col,
        local_col=args.local_col,
        datasets=args.datasets,
    )

    predictions_df, audit_df, summary_df = run_lambda_sensitivity(
        df=df,
        lambda_grid=args.lambda_grid,
        reference_lambda=args.reference_lambda,
        threshold=args.stability_threshold,
        base_col=args.base_col,
        local_col=args.local_col,
        min_state_samples=args.min_state_samples,
        min_prediction=args.min_prediction,
        max_prediction=args.max_prediction,
    )

    save_outputs(
        output_dir=args.output_dir,
        predictions_df=predictions_df,
        audit_df=audit_df,
        summary_df=summary_df,
        reference_lambda=args.reference_lambda,
    )


if __name__ == "__main__":
    main()
