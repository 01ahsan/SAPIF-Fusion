import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis


EPS = 1e-9


def robust_scale_mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scale = 1.4826 * mad

    if not np.isfinite(scale) or scale <= EPS:
        scale = np.std(values, ddof=1) if len(values) > 1 else 0.0

    return float(scale)


def robust_var_mad(values: np.ndarray) -> float:
    scale = robust_scale_mad(values)
    return float(max(scale * scale, EPS))


def excess_kurtosis(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 4:
        return 0.0

    value = float(kurtosis(values, fisher=True, bias=False))
    return value if np.isfinite(value) else 0.0


def kurtosis_inflated_variance(residuals: np.ndarray, base_variance: float) -> tuple[float, float, float]:
    kappa = excess_kurtosis(residuals)
    inflation = 1.0 + kappa / 2.0 if kappa > 1.0 else 1.0
    return float(max(base_variance * inflation, EPS)), float(kappa), float(inflation)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if not np.any(mask):
        return np.nan

    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def safe_precision(variance: float) -> float:
    variance = float(variance)

    if not np.isfinite(variance) or variance <= EPS:
        variance = EPS

    return 1.0 / variance


def build_prior(train_df: pd.DataFrame) -> dict:
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

        if len(state_residuals) == 0:
            mu_state = mu_population
            sigma2_state = sigma2_population
            kappa = 0.0
            inflation = 1.0
        else:
            mu_state = float(np.median(state_residuals))
            base_sigma2_state = robust_var_mad(state_residuals) + (sigma_w**2) / max(len(state_residuals), 1)
            sigma2_state, kappa, inflation = kurtosis_inflated_variance(
                residuals=state_residuals,
                base_variance=base_sigma2_state,
            )

        state_priors[str(state)] = {
            "mu_state": float(mu_state),
            "sigma2_state": float(max(sigma2_state, EPS)),
            "kurtosis": float(kappa),
            "inflation": float(inflation),
            "n_state": int(len(state_residuals)),
        }

    return {
        "mu_population": float(mu_population),
        "sigma_w": float(sigma_w),
        "sigma_b": float(sigma_b),
        "sigma2_population": float(max(sigma2_population, EPS)),
        "state_priors": state_priors,
    }


def predict_iv(
    test_df: pd.DataFrame,
    prior: dict,
    k: int,
    base_col: str,
    local_col: str,
    lambda_local: float,
) -> pd.DataFrame:
    rows = []

    mu_population = prior["mu_population"]
    sigma_w = prior["sigma_w"]

    if k <= 0:
        sigma2_local = np.inf
    else:
        sigma2_local = (sigma_w**2) / k + lambda_local * (sigma_w**2)

    tau_local = safe_precision(sigma2_local)

    for _, row in test_df.iterrows():
        base = float(row[base_col])
        state = str(row["state"])

        state_prior = prior["state_priors"].get(
            state,
            {
                "mu_state": mu_population,
                "sigma2_state": prior["sigma2_population"],
            },
        )

        mu_state = float(state_prior["mu_state"])
        sigma2_state = float(state_prior["sigma2_state"])
        tau_state = safe_precision(sigma2_state)

        if local_col in row.index and pd.notna(row[local_col]):
            mu_local = float(row[local_col]) - base
        else:
            mu_local = mu_population

        denominator = tau_state + tau_local

        if denominator <= EPS or not np.isfinite(denominator):
            alpha_state = 1.0
            alpha_local = 0.0
        else:
            alpha_state = tau_state / denominator
            alpha_local = tau_local / denominator

        delta = (
            mu_population
            + alpha_state * (mu_state - mu_population)
            + alpha_local * (mu_local - mu_population)
        )

        rows.append(
            {
                "true": float(row["true"]),
                "prediction": float(base + delta),
                "state": state,
                "subject": row["subject"],
            }
        )

    return pd.DataFrame(rows)


def validate_columns(df: pd.DataFrame, required_cols: set[str]) -> None:
    missing = sorted(required_cols - set(df.columns))

    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def prepare_data(
    input_csv: Path,
    datasets: list[str],
    k: int,
    base_col: str,
    local_col: str,
    method_cols: list[str],
) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(input_csv)

    required_cols = {"dataset", "subject", "state", "k", "true", base_col, local_col}
    required_cols.update(method_cols)
    validate_columns(df, required_cols)

    numeric_cols = ["k", "true", base_col, local_col] + method_cols

    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["dataset"] = df["dataset"].astype(str)
    df["subject"] = df["subject"].astype(str)
    df["state"] = df["state"].astype(str)
    df["base_residual"] = df["true"] - df[base_col]

    dataset_frames = {}

    for dataset in datasets:
        subset = df[(df["dataset"] == dataset) & (df["k"] == k)].copy()
        subset = subset.dropna(subset=["true", base_col, local_col, "base_residual"])

        if len(subset) == 0:
            continue

        dataset_frames[dataset] = subset.reset_index(drop=True)

    if not dataset_frames:
        raise ValueError("No valid dataset rows were found.")

    return dataset_frames


def bootstrap_prior_stability(
    data: pd.DataFrame,
    dataset: str,
    k: int,
    repeats: int,
    rng: np.random.Generator,
    base_col: str,
    local_col: str,
    lambda_local: float,
) -> tuple[dict, pd.DataFrame]:
    subjects = sorted(data["subject"].unique())
    expected_train_subjects = len(subjects) - 1

    sigma_b_values = []
    mae_values = []
    replicate_rows = []

    for bootstrap_id in range(repeats):
        fold_sigma_b = []
        prediction_parts = []

        for heldout_subject in subjects:
            train_subjects = [subject for subject in subjects if subject != heldout_subject]
            sampled_subjects = rng.choice(train_subjects, size=len(train_subjects), replace=True)

            train_parts = [data[data["subject"] == subject] for subject in sampled_subjects]
            train_df = pd.concat(train_parts, ignore_index=True)
            test_df = data[data["subject"] == heldout_subject].copy()

            prior = build_prior(train_df)
            fold_sigma_b.append(prior["sigma_b"])

            fold_predictions = predict_iv(
                test_df=test_df,
                prior=prior,
                k=k,
                base_col=base_col,
                local_col=local_col,
                lambda_local=lambda_local,
            )
            prediction_parts.append(fold_predictions)

        predictions = pd.concat(prediction_parts, ignore_index=True)
        cohort_mae = mae(predictions["true"], predictions["prediction"])

        mean_sigma_b = float(np.mean(fold_sigma_b))
        sd_sigma_b = float(np.std(fold_sigma_b, ddof=1))

        sigma_b_values.append(mean_sigma_b)
        mae_values.append(float(cohort_mae))

        replicate_rows.append(
            {
                "dataset": dataset,
                "bootstrap_id": int(bootstrap_id),
                "mean_sigma_b": mean_sigma_b,
                "sd_sigma_b": sd_sigma_b,
                "MAE": float(cohort_mae),
            }
        )

    original_sigma_b = []

    for heldout_subject in subjects:
        train_df = data[data["subject"] != heldout_subject].copy()
        prior = build_prior(train_df)
        original_sigma_b.append(prior["sigma_b"])

    sigma_b_values = np.asarray(sigma_b_values, dtype=float)
    mae_values = np.asarray(mae_values, dtype=float)

    summary = {
        "dataset": dataset,
        "subjects": int(len(subjects)),
        "training_subjects_per_fold": int(expected_train_subjects),
        "k": int(k),
        "sigma_b_original_mean": float(np.mean(original_sigma_b)),
        "sigma_b_bootstrap_mean": float(np.mean(sigma_b_values)),
        "sigma_b_bootstrap_ci90_low": float(np.percentile(sigma_b_values, 5)),
        "sigma_b_bootstrap_ci90_high": float(np.percentile(sigma_b_values, 95)),
        "sigma_b_rse_percent": float(np.std(sigma_b_values, ddof=1) / max(np.mean(sigma_b_values), EPS) * 100.0),
        "MAE_bootstrap_mean": float(np.mean(mae_values)),
        "MAE_bootstrap_ci90_low": float(np.percentile(mae_values, 5)),
        "MAE_bootstrap_ci90_high": float(np.percentile(mae_values, 95)),
        "MAE_sensitivity_sd": float(np.std(mae_values, ddof=1)),
    }

    return summary, pd.DataFrame(replicate_rows)


def run_pipeline(
    input_csv: Path,
    output_dir: Path,
    datasets: list[str],
    k: int,
    repeats: int,
    seed: int,
    base_col: str,
    local_col: str,
    method_cols: list[str],
    lambda_local: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    dataset_frames = prepare_data(
        input_csv=input_csv,
        datasets=datasets,
        k=k,
        base_col=base_col,
        local_col=local_col,
        method_cols=method_cols,
    )

    summary_rows = []
    replicate_parts = []

    for dataset, data in dataset_frames.items():
        summary, replicate_df = bootstrap_prior_stability(
            data=data,
            dataset=dataset,
            k=k,
            repeats=repeats,
            rng=rng,
            base_col=base_col,
            local_col=local_col,
            lambda_local=lambda_local,
        )
        summary_rows.append(summary)
        replicate_parts.append(replicate_df)

    summary_df = pd.DataFrame(summary_rows)
    replicate_df = pd.concat(replicate_parts, ignore_index=True)

    summary_path = output_dir / "prior_parameter_stability_summary.csv"
    replicate_path = output_dir / "prior_parameter_stability_replicates.csv"

    summary_df.to_csv(summary_path, index=False)
    replicate_df.to_csv(replicate_path, index=False)

    if summary_df["sigma_b_bootstrap_mean"].isna().any():
        raise AssertionError("Invalid sigma_b bootstrap mean.")

    if summary_df["MAE_sensitivity_sd"].isna().any():
        raise AssertionError("Invalid MAE sensitivity values.")

    if (summary_df["sigma_b_bootstrap_mean"] < 0).any():
        raise AssertionError("Negative sigma_b bootstrap mean.")

    if (summary_df["MAE_sensitivity_sd"] < 0).any():
        raise AssertionError("Negative MAE sensitivity.")

    return {
        "summary": str(summary_path),
        "replicates": str(replicate_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prior parameter bootstrap stability analysis.")

    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)

    parser.add_argument("--datasets", nargs="+", default=["EXTERNAL20", "MUST"])
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--base-col", default="L0_PPG_only")
    parser.add_argument("--local-col", default="E_local_only")
    parser.add_argument(
        "--method-cols",
        nargs="+",
        default=["SA_PIF_adaptive_v2", "SA_PIF_full_v1"],
    )
    parser.add_argument("--lambda-local", type=float, default=1.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_pipeline(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        datasets=args.datasets,
        k=args.k,
        repeats=args.repeats,
        seed=args.seed,
        base_col=args.base_col,
        local_col=args.local_col,
        method_cols=args.method_cols,
        lambda_local=args.lambda_local,
    )


if __name__ == "__main__":
    main()
