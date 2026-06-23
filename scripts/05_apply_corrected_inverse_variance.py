import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm


EPS = 1e-9
DEFAULT_ALPHAS = [0.50, 0.80, 0.90, 0.95]


def robust_scale_mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scale = 1.4826 * mad

    if not np.isfinite(scale) or scale <= EPS:
        scale = np.std(values, ddof=1) if len(values) > 1 else EPS

    return float(max(scale, EPS))


def robust_var(values: np.ndarray) -> float:
    return robust_scale_mad(values) ** 2


def excess_kurtosis(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 4:
        return 0.0

    value = float(kurtosis(values, fisher=True, bias=False))
    return 0.0 if not np.isfinite(value) else value


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


def mard(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if not np.any(mask):
        return np.nan

    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.maximum(y_true[mask], EPS)) * 100.0)


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if not np.any(mask):
        return np.nan

    return float(np.mean(y_pred[mask] - y_true[mask]))


def precision(variance: float) -> float:
    variance = float(variance)

    if not np.isfinite(variance) or variance <= EPS:
        variance = EPS

    return 1.0 / variance


def build_priors(train_df: pd.DataFrame) -> dict:
    residuals = train_df["base_residual"].to_numpy(dtype=float)

    mu_population = float(np.median(residuals))
    sigma_w = robust_scale_mad(residuals)

    subject_medians = (
        train_df.groupby("subject")["base_residual"]
        .median()
        .to_numpy(dtype=float)
    )
    sigma_b = robust_scale_mad(subject_medians)
    sigma2_population = sigma_b**2 + sigma_w**2

    state_priors = {}

    for state, group in train_df.groupby("state"):
        state_residuals = group["base_residual"].to_numpy(dtype=float)

        mu_state = float(np.median(state_residuals))
        sigma2_state_base = robust_var(state_residuals) + sigma_w**2

        kappa = excess_kurtosis(state_residuals)
        inflation = 1.0 + kappa / 2.0 if kappa > 1.0 else 1.0
        sigma2_state = sigma2_state_base * inflation

        state_priors[str(state)] = {
            "mu_state": mu_state,
            "sigma2_state": float(max(sigma2_state, EPS)),
            "kurtosis": float(kappa),
            "inflation": float(inflation),
            "n_state": int(len(state_residuals)),
        }

    return {
        "mu_population": mu_population,
        "sigma_w": float(sigma_w),
        "sigma_b": float(sigma_b),
        "sigma2_population": float(max(sigma2_population, EPS)),
        "state_priors": state_priors,
    }


def local_offset(row: pd.Series, base_col: str, local_col: str) -> float:
    return float(row[local_col]) - float(row[base_col])


def local_variance(sigma_w: float, k: int, lambda_local: float) -> float:
    if k <= 0:
        return np.inf

    return float((sigma_w**2) / k + lambda_local * sigma_w**2)


def predict_corrected_iv(
    test_df: pd.DataFrame,
    priors: dict,
    k: int,
    base_col: str,
    local_col: str,
    lambda_local: float,
) -> pd.DataFrame:
    rows = []

    mu_population = priors["mu_population"]
    sigma_w = priors["sigma_w"]
    sigma2_local = local_variance(sigma_w, k, lambda_local)

    for row_index, row in test_df.iterrows():
        base = float(row[base_col])
        state = str(row["state"])

        state_prior = priors["state_priors"].get(
            state,
            {
                "mu_state": mu_population,
                "sigma2_state": priors["sigma2_population"],
                "kurtosis": 0.0,
                "inflation": 1.0,
                "n_state": 0,
            },
        )

        mu_state = float(state_prior["mu_state"])
        tau_state = precision(state_prior["sigma2_state"])

        if k <= 0:
            mu_local = np.nan
            tau_local = 0.0
            alpha_state = 1.0
            alpha_local = 0.0
        else:
            mu_local = local_offset(row, base_col=base_col, local_col=local_col)
            tau_local = precision(sigma2_local)
            denominator = tau_state + tau_local
            alpha_state = tau_state / denominator
            alpha_local = tau_local / denominator

        local_term = mu_local if np.isfinite(mu_local) else mu_population

        delta = (
            mu_population
            + alpha_state * (mu_state - mu_population)
            + alpha_local * (local_term - mu_population)
        )
        prediction = base + delta
        sigma2_fused = 1.0 / max(tau_state + tau_local, EPS)

        rows.append(
            {
                "row_index": row_index,
                "corrected_iv_prediction": float(prediction),
                "corrected_iv_delta": float(delta),
                "mu_population": float(mu_population),
                "mu_state": float(mu_state),
                "mu_local": float(mu_local) if np.isfinite(mu_local) else np.nan,
                "alpha_state": float(alpha_state),
                "alpha_local": float(alpha_local),
                "sigma2_fused": float(sigma2_fused),
                "sigma_fused": float(np.sqrt(sigma2_fused)),
                "state_kurtosis": float(state_prior["kurtosis"]),
                "state_inflation": float(state_prior["inflation"]),
                "n_state_train": int(state_prior["n_state"]),
            }
        )

    return pd.DataFrame(rows)


def summarize_predictions(df: pd.DataFrame, prediction_cols: dict[str, str]) -> pd.DataFrame:
    rows = []

    for dataset in sorted(df["dataset"].unique()):
        dataset_df = df[df["dataset"] == dataset]

        for k in sorted(dataset_df["k"].dropna().astype(int).unique()):
            subset = dataset_df[dataset_df["k"] == k]

            for method, column in prediction_cols.items():
                if column not in subset.columns:
                    continue

                valid = subset.dropna(subset=["true", column])

                if len(valid) == 0:
                    continue

                rows.append(
                    {
                        "dataset": dataset,
                        "k": int(k),
                        "method": method,
                        "n": int(len(valid)),
                        "subjects": int(valid["subject"].nunique()),
                        "MAE": mae(valid["true"], valid[column]),
                        "RMSE": rmse(valid["true"], valid[column]),
                        "MARD": mard(valid["true"], valid[column]),
                        "Bias": bias(valid["true"], valid[column]),
                    }
                )

    return pd.DataFrame(rows).sort_values(["dataset", "k", "method"])


def compare_iv_methods(
    df: pd.DataFrame,
    old_iv_col: str,
    corrected_iv_col: str,
    adaptive_col: str,
) -> pd.DataFrame:
    rows = []

    for (dataset, k), group in df.groupby(["dataset", "k"]):
        rows.append(
            {
                "dataset": dataset,
                "k": int(k),
                "n": int(len(group)),
                "subjects": int(group["subject"].nunique()),
                "old_IV_MAE": mae(group["true"], group[old_iv_col]),
                "corrected_IV_MAE": mae(group["true"], group[corrected_iv_col]),
                "delta_corrected_minus_old": mae(group["true"], group[corrected_iv_col])
                - mae(group["true"], group[old_iv_col]),
                "adaptive_MAE": mae(group["true"], group[adaptive_col])
                if adaptive_col in group.columns
                else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values(["dataset", "k"])


def compute_interval_calibration(
    df: pd.DataFrame,
    datasets: list[str],
    k: int,
    alphas: list[float],
    prediction_col: str,
) -> pd.DataFrame:
    rows = []

    for dataset in datasets:
        subset = df[(df["dataset"] == dataset) & (df["k"] == k)].copy()

        if len(subset) == 0:
            continue

        for state, group in subset.groupby("state"):
            errors = group["true"].to_numpy(dtype=float) - group[prediction_col].to_numpy(dtype=float)
            sigmas = group["sigma_fused"].to_numpy(dtype=float)

            for alpha in alphas:
                z_value = norm.ppf((1.0 + alpha) / 2.0)
                covered = np.abs(errors) <= z_value * sigmas
                width = np.mean(2.0 * z_value * sigmas)

                rows.append(
                    {
                        "dataset": dataset,
                        "state": state,
                        "k": int(k),
                        "n": int(len(group)),
                        "nominal_alpha": float(alpha),
                        "empirical_coverage": float(np.mean(covered)),
                        "coverage_ratio": float(np.mean(covered) / alpha),
                        "mean_interval_width": float(width),
                        "mean_sigma_fused": float(np.mean(sigmas)),
                        "mean_alpha_state": float(group["alpha_state"].mean()),
                        "mean_alpha_local": float(group["alpha_local"].mean()),
                        "mean_state_kurtosis": float(group["state_kurtosis"].mean()),
                        "mean_state_inflation": float(group["state_inflation"].mean()),
                        "MAE": mae(group["true"], group[prediction_col]),
                    }
                )

    return pd.DataFrame(rows).sort_values(["dataset", "state", "nominal_alpha"])


def validate_columns(df: pd.DataFrame, required_cols: set[str]) -> None:
    missing = sorted(required_cols - set(df.columns))

    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def load_input(
    input_csv: Path,
    base_col: str,
    local_col: str,
    old_iv_col: str,
    adaptive_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    required_cols = {
        "dataset",
        "subject",
        "state",
        "k",
        "true",
        base_col,
        "E_pop_only",
        "E_state_only",
        local_col,
        old_iv_col,
        adaptive_col,
    }
    validate_columns(df, required_cols)

    for column in ["k", "true", base_col, "E_pop_only", "E_state_only", local_col, old_iv_col, adaptive_col]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["dataset"] = df["dataset"].astype(str)
    df["subject"] = df["subject"].astype(str)
    df["state"] = df["state"].astype(str)
    df["base_residual"] = df["true"] - df[base_col]

    return df


def run_correction(
    df: pd.DataFrame,
    base_col: str,
    local_col: str,
    lambda_local: float,
) -> pd.DataFrame:
    prediction_parts = []

    for dataset in sorted(df["dataset"].unique()):
        dataset_df = df[df["dataset"] == dataset].copy()

        for k in sorted(dataset_df["k"].dropna().astype(int).unique()):
            k_df = dataset_df[dataset_df["k"] == k].copy()

            for heldout_subject in sorted(k_df["subject"].unique()):
                train_df = k_df[k_df["subject"] != heldout_subject].copy()
                test_df = k_df[k_df["subject"] == heldout_subject].copy()

                if len(train_df) == 0 or len(test_df) == 0:
                    continue

                priors = build_priors(train_df)
                prediction_parts.append(
                    predict_corrected_iv(
                        test_df=test_df,
                        priors=priors,
                        k=k,
                        base_col=base_col,
                        local_col=local_col,
                        lambda_local=lambda_local,
                    )
                )

    if not prediction_parts:
        raise RuntimeError("No corrected IV predictions were generated.")

    predictions = pd.concat(prediction_parts, ignore_index=True).set_index("row_index")
    corrected = df.join(predictions, how="left")

    if corrected["corrected_iv_prediction"].isna().any():
        raise AssertionError("Missing corrected IV predictions.")

    return corrected


def validate_k0_identity(
    df: pd.DataFrame,
    corrected_col: str,
    state_col: str,
    tolerance: float,
) -> None:
    for dataset in sorted(df["dataset"].unique()):
        subset = df[(df["dataset"] == dataset) & (df["k"] == 0)]

        if len(subset) == 0:
            continue

        max_diff = float(np.max(np.abs(subset[corrected_col] - subset[state_col])))

        if max_diff > tolerance:
            raise AssertionError(f"k=0 identity check failed for {dataset}: {max_diff}")


def run_pipeline(
    input_csv: Path,
    output_dir: Path,
    lambda_local: float,
    alphas: list[float],
    calibration_k: int,
    calibration_datasets: list[str],
    base_col: str,
    local_col: str,
    old_iv_col: str,
    adaptive_col: str,
    k0_tolerance: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_input(
        input_csv=input_csv,
        base_col=base_col,
        local_col=local_col,
        old_iv_col=old_iv_col,
        adaptive_col=adaptive_col,
    )

    corrected = run_correction(
        df=df,
        base_col=base_col,
        local_col=local_col,
        lambda_local=lambda_local,
    )

    validate_k0_identity(
        df=corrected,
        corrected_col="corrected_iv_prediction",
        state_col="E_state_only",
        tolerance=k0_tolerance,
    )

    predictions_path = output_dir / "corrected_predictions.csv"
    corrected.to_csv(predictions_path, index=False)

    prediction_cols = {
        "L0": base_col,
        "population_only": "E_pop_only",
        "state_only": "E_state_only",
        "local_only": local_col,
        "corrected_iv": "corrected_iv_prediction",
        "adaptive": adaptive_col,
    }

    summary = summarize_predictions(corrected, prediction_cols)
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    comparison = compare_iv_methods(
        df=corrected,
        old_iv_col=old_iv_col,
        corrected_iv_col="corrected_iv_prediction",
        adaptive_col=adaptive_col,
    )
    comparison_path = output_dir / "iv_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    calibration = compute_interval_calibration(
        df=corrected,
        datasets=calibration_datasets,
        k=calibration_k,
        alphas=alphas,
        prediction_col="corrected_iv_prediction",
    )
    calibration_path = output_dir / "posterior_calibration.csv"
    calibration.to_csv(calibration_path, index=False)

    return {
        "predictions": str(predictions_path),
        "summary": str(summary_path),
        "iv_comparison": str(comparison_path),
        "posterior_calibration": str(calibration_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run corrected inverse-variance fusion.")

    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)

    parser.add_argument("--lambda-local", type=float, default=1.0)
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)

    parser.add_argument("--calibration-k", type=int, default=2)
    parser.add_argument("--calibration-datasets", nargs="+", default=["EXTERNAL20", "MUST"])

    parser.add_argument("--base-col", default="L0_PPG_only")
    parser.add_argument("--local-col", default="E_local_only")
    parser.add_argument("--old-iv-col", default="SA_PIF_full_v1")
    parser.add_argument("--adaptive-col", default="SA_PIF_adaptive_v2")

    parser.add_argument("--k0-tolerance", type=float, default=1e-8)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_pipeline(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        lambda_local=args.lambda_local,
        alphas=args.alphas,
        calibration_k=args.calibration_k,
        calibration_datasets=args.calibration_datasets,
        base_col=args.base_col,
        local_col=args.local_col,
        old_iv_col=args.old_iv_col,
        adaptive_col=args.adaptive_col,
        k0_tolerance=args.k0_tolerance,
    )


if __name__ == "__main__":
    main()
