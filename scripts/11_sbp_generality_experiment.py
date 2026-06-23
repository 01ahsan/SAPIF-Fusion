import argparse
import json
import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


DEFAULT_K_VALUES = [0, 1, 2, 5]
DEFAULT_METHODS = [
    "L0_base",
    "E_pop_only",
    "E_state_only",
    "E_local_only",
    "SA_PIF_inversevar",
    "SA_PIF_adaptive",
]
PREDICTION_CLIP = (60.0, 260.0)
EPS = 1e-6


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, clip_range: tuple[float, float]) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), *clip_range)

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

    error = np.abs(y_pred - y_true)
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
        "MAE": float(np.mean(error)),
        "MedianAE": float(np.median(error)),
        "RMSE": float(np.sqrt(np.mean(signed_error**2))),
        "MARD": float(np.mean(error / np.maximum(y_true, 1e-9)) * 100.0),
        "Bias": float(np.mean(signed_error)),
        "Pearson_r": pearson_r,
    }


def robust_var(values: np.ndarray, fallback: float = 100.0, eps: float = EPS) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) <= 1:
        return float(max(fallback, eps))

    iqr = np.percentile(values, 75) - np.percentile(values, 25)

    if iqr > 0:
        variance = (iqr / 1.349) ** 2
    else:
        variance = np.var(values)

    if not np.isfinite(variance) or variance < eps:
        variance = max(np.var(values), fallback)

    return float(max(variance, eps))


def collect_parquet_files(processed_dir: Path) -> list[Path]:
    files = sorted(processed_dir.glob("p*.parquet"))

    if len(files) == 0:
        raise FileNotFoundError(f"No processed parquet files found in {processed_dir}")

    return files


def load_subject_data(
    processed_dir: Path,
    min_windows_per_subject: int,
    max_windows_per_subject: int,
    seed: int,
) -> pd.DataFrame:
    frames = []

    for path in collect_parquet_files(processed_dir):
        frame = pd.read_parquet(path)

        if len(frame) < min_windows_per_subject:
            continue

        if len(frame) > max_windows_per_subject:
            frame = frame.sample(n=max_windows_per_subject, random_state=seed).copy()

        frames.append(frame)

    if not frames:
        raise ValueError("No valid subject files remained after filtering.")

    return pd.concat(frames, ignore_index=True)


def select_feature_columns(df: pd.DataFrame, target_col: str, extra_drop_cols: list[str]) -> list[str]:
    drop_cols = {
        "subject_file",
        "subject_id",
        "case_id",
        "segment_id",
        "win_id",
        "dataset_subset",
        target_col,
        "sbp",
        "dbp",
        "mbp",
        "state",
        "gender",
        "subject",
    }
    drop_cols.update(extra_drop_cols)

    feature_cols = [
        column
        for column in df.columns
        if column not in drop_cols and pd.api.types.is_numeric_dtype(df[column])
    ]

    feature_cols = [column for column in feature_cols if not df[column].isna().all()]

    if not feature_cols:
        raise ValueError("No usable numeric feature columns were found.")

    return feature_cols


def prepare_dataset(
    combined: pd.DataFrame,
    target_col: str,
    min_windows_per_subject: int,
    target_range: tuple[float, float],
    dbp_range: tuple[float, float],
    extra_drop_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    df = combined.copy()

    df = df[df[target_col].between(*target_range)].copy()

    if "dbp" in df.columns:
        df = df[df["dbp"].between(*dbp_range)].copy()

    if "subject_file" not in df.columns:
        raise KeyError("Column 'subject_file' is required.")

    df["subject"] = df["subject_file"].astype(str)

    counts = df.groupby("subject").size()
    keep_subjects = counts[counts >= min_windows_per_subject].index
    df = df[df["subject"].isin(keep_subjects)].copy()

    if len(df) == 0:
        raise ValueError("No rows remained after cleaning and subject filtering.")

    feature_cols = select_feature_columns(
        df=df,
        target_col=target_col,
        extra_drop_cols=extra_drop_cols,
    )

    return df.reset_index(drop=True), feature_cols


def split_subjects(
    subjects: np.ndarray,
    base_train_frac: float,
    fusion_train_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    if not np.isclose(base_train_frac + fusion_train_frac + test_frac, 1.0):
        raise ValueError("Subject split fractions must sum to 1.")

    base_train_subjects, temp_subjects = train_test_split(
        subjects,
        train_size=base_train_frac,
        random_state=seed,
        shuffle=True,
    )

    fusion_train_subjects, test_subjects = train_test_split(
        temp_subjects,
        train_size=fusion_train_frac / (fusion_train_frac + test_frac),
        random_state=seed,
        shuffle=True,
    )

    return set(base_train_subjects), set(fusion_train_subjects), set(test_subjects)


def train_base_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seed: int,
) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="absolute_error",
                    max_iter=350,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    l2_regularization=0.01,
                    random_state=seed,
                ),
            ),
        ]
    )

    model.fit(train_df[feature_cols], train_df[target_col].astype(float))
    return model


def add_base_predictions(
    model: Pipeline,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    clip_range: tuple[float, float],
) -> pd.DataFrame:
    out = df.copy()
    out["base_pred"] = model.predict(out[feature_cols])
    out["base_pred"] = out["base_pred"].clip(*clip_range)
    out["resid"] = out[target_col].astype(float) - out["base_pred"]
    return out


def build_prior_stats(train_df: pd.DataFrame) -> dict:
    residuals = train_df["resid"].to_numpy(dtype=float)

    mu_population = float(np.median(residuals))
    sigma_w2 = robust_var(residuals, fallback=100.0)

    subject_offsets = (
        train_df.groupby("subject")["resid"]
        .median()
        .to_numpy(dtype=float)
    )
    sigma_b2 = robust_var(subject_offsets, fallback=sigma_w2)
    var_population = sigma_b2 + sigma_w2 / max(len(residuals), 1)

    state_mu = {}
    state_var = {}

    for state, group in train_df.groupby("state"):
        values = group["resid"].to_numpy(dtype=float)

        if len(values) >= 5:
            state_mu[state] = float(np.median(values))
            state_var[state] = float(robust_var(values, fallback=sigma_w2) + sigma_w2 / len(values))
        elif len(values) > 0:
            raw_mu = float(np.median(values))
            shrinkage = len(values) / (len(values) + 10.0)
            state_mu[state] = float(mu_population + shrinkage * (raw_mu - mu_population))
            state_var[state] = float(4.0 * sigma_w2)

    return {
        "mu_pop": mu_population,
        "var_pop": max(var_population, EPS),
        "sigma_w2": max(sigma_w2, EPS),
        "sigma_b2": max(sigma_b2, EPS),
        "state_mu": state_mu,
        "state_var": state_var,
    }


def select_calibration_rows(
    subject_df: pd.DataFrame,
    test_index: int,
    k: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if k <= 0:
        return subject_df.iloc[[]].copy()

    candidates = subject_df.drop(index=test_index, errors="ignore").copy()

    if len(candidates) == 0:
        return subject_df.iloc[[]].copy()

    k_effective = min(k, len(candidates))

    return candidates.sample(
        n=k_effective,
        replace=False,
        random_state=int(rng.integers(0, 10**9)),
    )


def local_evidence(calibration_df: pd.DataFrame, prior: dict, offset_clip: float) -> dict | None:
    if len(calibration_df) == 0:
        return None

    residuals = calibration_df["resid"].to_numpy(dtype=float)

    if len(residuals) == 0:
        return None

    if len(residuals) < 3:
        mu_local = float(np.mean(residuals))
    else:
        mu_local = float(np.median(residuals))

    if len(residuals) >= 2:
        var_empirical = robust_var(residuals, fallback=prior["sigma_w2"])
    else:
        var_empirical = prior["sigma_w2"]

    var_local = var_empirical / max(len(residuals), 1) + 0.10 * prior["sigma_w2"]
    clip_width = min(offset_clip, 3.0 * math.sqrt(prior["sigma_b2"] + prior["sigma_w2"]))
    mu_local = float(np.clip(mu_local, prior["mu_pop"] - clip_width, prior["mu_pop"] + clip_width))

    return {
        "mu": mu_local,
        "var": max(var_local, EPS),
    }


def inverse_variance_fuse(evidence_list: list[dict | None]) -> float:
    means = []
    precisions = []

    for evidence in evidence_list:
        if evidence is None:
            continue

        mean = float(evidence["mu"])
        variance = float(evidence["var"])

        if not np.isfinite(mean) or not np.isfinite(variance) or variance <= 0:
            continue

        means.append(mean)
        precisions.append(1.0 / max(variance, EPS))

    if len(means) == 0:
        return 0.0

    means = np.asarray(means, dtype=float)
    precisions = np.asarray(precisions, dtype=float)
    weights = precisions / np.sum(precisions)

    return float(np.sum(weights * means))


def simplex_grid(step: float, allow_local: bool) -> list[tuple[float, float, float]]:
    values = np.arange(0, 1 + 1e-9, step)
    grid = []

    if allow_local:
        for weight_pop in values:
            for weight_state in values:
                weight_local = 1.0 - weight_pop - weight_state

                if weight_local < -1e-9:
                    continue

                grid.append((float(weight_pop), float(weight_state), float(max(0.0, weight_local))))
    else:
        for weight_pop in values:
            weight_state = 1.0 - weight_pop
            grid.append((float(weight_pop), float(weight_state), 0.0))

    return grid


def predict_weighted(
    frame: pd.DataFrame,
    weights: tuple[float, float, float],
    clip_range: tuple[float, float],
) -> np.ndarray:
    weight_pop, weight_state, weight_local = weights

    prediction = (
        frame["base_pred"].to_numpy(dtype=float)
        + weight_pop * frame["off_pop"].to_numpy(dtype=float)
        + weight_state * frame["off_state"].to_numpy(dtype=float)
        + weight_local * frame["off_local"].to_numpy(dtype=float)
    )

    return np.clip(prediction, *clip_range)


def choose_adaptive_weights(
    train_evidence_df: pd.DataFrame,
    k: int,
    grid_with_local: list[tuple[float, float, float]],
    grid_no_local: list[tuple[float, float, float]],
    clip_range: tuple[float, float],
) -> dict:
    grid = grid_no_local if k == 0 else grid_with_local
    y_true = train_evidence_df["true"].to_numpy(dtype=float)

    best = None

    for weights in grid:
        prediction = predict_weighted(train_evidence_df, weights, clip_range=clip_range)
        metrics = compute_metrics(y_true, prediction, clip_range=clip_range)
        score = (metrics["MAE"], metrics["RMSE"])

        if best is None or score < best["score"]:
            best = {
                "weights": weights,
                "score": score,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
            }

    return best


def build_evidence_predictions(
    eval_df: pd.DataFrame,
    prior: dict,
    k: int,
    rng: np.random.Generator,
    target_col: str,
    clip_range: tuple[float, float],
    offset_clip: float,
) -> pd.DataFrame:
    rows = []

    for subject, subject_df in eval_df.groupby("subject"):
        subject_df = subject_df.copy()

        for index, row in subject_df.iterrows():
            state = row["state"]
            base_prediction = float(row["base_pred"])

            population_evidence = {
                "mu": prior["mu_pop"],
                "var": prior["var_pop"],
            }

            state_evidence = {
                "mu": prior["state_mu"].get(state, prior["mu_pop"]),
                "var": prior["state_var"].get(state, 4.0 * prior["sigma_w2"]),
            }

            calibration_df = select_calibration_rows(
                subject_df=subject_df,
                test_index=index,
                k=k,
                rng=rng,
            )
            local_ev = local_evidence(
                calibration_df=calibration_df,
                prior=prior,
                offset_clip=offset_clip,
            )

            offset_pop = population_evidence["mu"]
            offset_state = state_evidence["mu"]
            offset_local = local_ev["mu"] if local_ev is not None else 0.0
            offset_inverse_variance = inverse_variance_fuse(
                [population_evidence, state_evidence, local_ev]
            )

            rows.append(
                {
                    "subject": subject,
                    "state": state,
                    "k": int(k),
                    "true": float(row[target_col]),
                    "base_pred": base_prediction,
                    "off_pop": offset_pop,
                    "off_state": offset_state,
                    "off_local": offset_local,
                    "L0_base": float(np.clip(base_prediction, *clip_range)),
                    "E_pop_only": float(np.clip(base_prediction + offset_pop, *clip_range)),
                    "E_state_only": float(np.clip(base_prediction + offset_state, *clip_range)),
                    "E_local_only": float(np.clip(base_prediction + offset_local, *clip_range)),
                    "SA_PIF_inversevar": float(np.clip(base_prediction + offset_inverse_variance, *clip_range)),
                }
            )

    return pd.DataFrame(rows)


def learn_adaptive_weights(
    fusion_train_df: pd.DataFrame,
    prior: dict,
    k_values: list[int],
    grid_step: float,
    seed: int,
    target_col: str,
    clip_range: tuple[float, float],
    offset_clip: float,
) -> pd.DataFrame:
    grid_with_local = simplex_grid(grid_step, allow_local=True)
    grid_no_local = simplex_grid(grid_step, allow_local=False)

    rows = []

    for k in k_values:
        rng = np.random.default_rng(seed + k)

        train_evidence = build_evidence_predictions(
            eval_df=fusion_train_df,
            prior=prior,
            k=k,
            rng=rng,
            target_col=target_col,
            clip_range=clip_range,
            offset_clip=offset_clip,
        )

        best = choose_adaptive_weights(
            train_evidence_df=train_evidence,
            k=k,
            grid_with_local=grid_with_local,
            grid_no_local=grid_no_local,
            clip_range=clip_range,
        )

        rows.append(
            {
                "k": int(k),
                "w_pop": float(best["weights"][0]),
                "w_state": float(best["weights"][1]),
                "w_local": float(best["weights"][2]),
                "fusion_train_MAE": float(best["MAE"]),
                "fusion_train_RMSE": float(best["RMSE"]),
            }
        )

    return pd.DataFrame(rows)


def evaluate_test_set(
    test_df: pd.DataFrame,
    prior: dict,
    weights_df: pd.DataFrame,
    k_values: list[int],
    n_repeats: int,
    seed: int,
    target_col: str,
    clip_range: tuple[float, float],
    offset_clip: float,
) -> pd.DataFrame:
    parts = []

    for repeat in range(n_repeats):
        rng = np.random.default_rng(seed + 1000 + repeat)

        for k in k_values:
            evidence = build_evidence_predictions(
                eval_df=test_df,
                prior=prior,
                k=k,
                rng=rng,
                target_col=target_col,
                clip_range=clip_range,
                offset_clip=offset_clip,
            )

            weights = (
                weights_df[weights_df["k"] == k][["w_pop", "w_state", "w_local"]]
                .iloc[0]
                .to_numpy(dtype=float)
            )

            evidence["SA_PIF_adaptive"] = predict_weighted(
                evidence,
                tuple(weights),
                clip_range=clip_range,
            )
            evidence["repeat"] = int(repeat)
            parts.append(evidence)

    return pd.concat(parts, ignore_index=True)


def summarize_repeats(
    predictions: pd.DataFrame,
    methods: list[str],
    clip_range: tuple[float, float],
) -> pd.DataFrame:
    rows = []

    for (repeat, k), group in predictions.groupby(["repeat", "k"]):
        for method in methods:
            metrics = compute_metrics(group["true"], group[method], clip_range=clip_range)

            rows.append(
                {
                    "repeat": int(repeat),
                    "k": int(k),
                    "method": method,
                    "n": metrics["n"],
                    "MAE": metrics["MAE"],
                    "MedianAE": metrics["MedianAE"],
                    "RMSE": metrics["RMSE"],
                    "MARD": metrics["MARD"],
                    "Bias": metrics["Bias"],
                    "Pearson_r": metrics["Pearson_r"],
                }
            )

    return pd.DataFrame(rows)


def aggregate_summary(
    repeat_summary: pd.DataFrame,
    dataset_name: str,
    task_name: str,
) -> pd.DataFrame:
    rows = []

    for (k, method), group in repeat_summary.groupby(["k", "method"]):
        baseline = repeat_summary[
            (repeat_summary["k"] == k)
            & (repeat_summary["method"] == "L0_base")
        ][["repeat", "MAE"]].rename(columns={"MAE": "baseline_MAE"})

        merged = pd.merge(group, baseline, on="repeat")
        improvement = (merged["baseline_MAE"] - merged["MAE"]) / merged["baseline_MAE"] * 100.0

        rows.append(
            {
                "dataset": dataset_name,
                "task": task_name,
                "k": int(k),
                "method": method,
                "n_test_samples_mean": float(merged["n"].mean()),
                "MAE_mean": float(merged["MAE"].mean()),
                "MAE_std": float(merged["MAE"].std()),
                "MedianAE_mean": float(merged["MedianAE"].mean()),
                "RMSE_mean": float(merged["RMSE"].mean()),
                "MARD_mean": float(merged["MARD"].mean()),
                "Bias_mean": float(merged["Bias"].mean()),
                "Pearson_r_mean": float(merged["Pearson_r"].mean()),
                "Improvement_vs_L0_mean_percent": float(improvement.mean()),
                "Improvement_vs_L0_std_percent": float(improvement.std()),
            }
        )

    return pd.DataFrame(rows).sort_values(["k", "MAE_mean"])


def pairwise_tests(repeat_summary: pd.DataFrame, k_values: list[int]) -> pd.DataFrame:
    rows = []

    for k in k_values:
        target = repeat_summary[
            (repeat_summary["k"] == k)
            & (repeat_summary["method"] == "SA_PIF_adaptive")
        ][["repeat", "MAE"]].rename(columns={"MAE": "MAE_target"})

        for baseline_method in ["L0_base", "E_local_only", "SA_PIF_inversevar"]:
            baseline = repeat_summary[
                (repeat_summary["k"] == k)
                & (repeat_summary["method"] == baseline_method)
            ][["repeat", "MAE"]].rename(columns={"MAE": "MAE_baseline"})

            merged = pd.merge(target, baseline, on="repeat")
            diff = merged["MAE_baseline"].to_numpy(dtype=float) - merged["MAE_target"].to_numpy(dtype=float)

            if len(diff) < 2 or np.allclose(diff, 0):
                statistic, p_value = np.nan, np.nan
            else:
                try:
                    statistic, p_value = wilcoxon(diff)
                except Exception:
                    statistic, p_value = np.nan, np.nan

            rows.append(
                {
                    "k": int(k),
                    "comparison": f"SA_PIF_adaptive vs {baseline_method}",
                    "repeats": int(len(merged)),
                    "MAE_baseline_mean": float(merged["MAE_baseline"].mean()),
                    "MAE_target_mean": float(merged["MAE_target"].mean()),
                    "delta_MAE_mean": float(diff.mean()),
                    "delta_MAE_median": float(np.median(diff)),
                    "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                    "significant_p05": bool(p_value < 0.05) if np.isfinite(p_value) else False,
                }
            )

    return pd.DataFrame(rows)


def statewise_summary(
    predictions: pd.DataFrame,
    clip_range: tuple[float, float],
) -> pd.DataFrame:
    rows = []

    methods = ["L0_base", "E_state_only", "E_local_only", "SA_PIF_adaptive"]

    for (repeat, k, state), group in predictions.groupby(["repeat", "k", "state"]):
        for method in methods:
            metrics = compute_metrics(group["true"], group[method], clip_range=clip_range)

            rows.append(
                {
                    "repeat": int(repeat),
                    "k": int(k),
                    "state": state,
                    "method": method,
                    "n": metrics["n"],
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "MARD": metrics["MARD"],
                    "Bias": metrics["Bias"],
                    "Pearson_r": metrics["Pearson_r"],
                }
            )

    state_repeat = pd.DataFrame(rows)

    return (
        state_repeat.groupby(["k", "state", "method"])
        .agg(
            n_mean=("n", "mean"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            MARD_mean=("MARD", "mean"),
            Bias_mean=("Bias", "mean"),
            Pearson_r_mean=("Pearson_r", "mean"),
        )
        .reset_index()
    )


def plot_calibration_curve(aggregate_df: pd.DataFrame, output_dir: Path) -> None:
    methods = ["L0_base", "E_local_only", "SA_PIF_inversevar", "SA_PIF_adaptive"]
    labels = {
        "L0_base": "Base model",
        "E_local_only": "Local calibration only",
        "SA_PIF_inversevar": "Inverse-variance fusion",
        "SA_PIF_adaptive": "Adaptive fusion",
    }

    plt.figure(figsize=(8.5, 5.2))

    for method in methods:
        subset = aggregate_df[aggregate_df["method"] == method].sort_values("k")

        if len(subset) == 0:
            continue

        plt.errorbar(
            subset["k"],
            subset["MAE_mean"],
            yerr=subset["MAE_std"],
            marker="o",
            linewidth=2,
            capsize=4,
            label=labels[method],
        )

    plt.xlabel("Calibration samples per subject")
    plt.ylabel("MAE")
    plt.title("Sparse calibration performance")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig(output_dir / "calibration_curve.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "calibration_curve.pdf", bbox_inches="tight")
    plt.close()


def save_metadata(
    output_dir: Path,
    dataset_name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    base_train_subjects: set[str],
    fusion_train_subjects: set[str],
    test_subjects: set[str],
) -> None:
    split_info = {
        "dataset": dataset_name,
        "rows_total": int(len(df)),
        "subjects_total": int(df["subject"].nunique()),
        "base_train_subjects": int(len(base_train_subjects)),
        "fusion_train_subjects": int(len(fusion_train_subjects)),
        "test_subjects": int(len(test_subjects)),
        "feature_cols": feature_cols,
    }

    with open(output_dir / "split_info.json", "w", encoding="utf-8") as file:
        json.dump(split_info, file, indent=2)


def save_prior_stats(output_dir: Path, prior: dict) -> None:
    prior_out = {
        "mu_pop": prior["mu_pop"],
        "var_pop": prior["var_pop"],
        "sigma_w2": prior["sigma_w2"],
        "sigma_b2": prior["sigma_b2"],
        "state_mu": prior["state_mu"],
        "state_var": prior["state_var"],
    }

    with open(output_dir / "prior_stats.json", "w", encoding="utf-8") as file:
        json.dump(prior_out, file, indent=2)


def run_pipeline(
    processed_dir: Path,
    output_dir: Path,
    dataset_name: str,
    target_col: str,
    task_name: str,
    max_windows_per_subject: int,
    min_windows_per_subject: int,
    base_train_frac: float,
    fusion_train_frac: float,
    test_frac: float,
    seed: int,
    k_values: list[int],
    n_repeats: int,
    grid_step: float,
    prediction_clip: tuple[float, float],
    target_range: tuple[float, float],
    dbp_range: tuple[float, float],
    offset_clip: float,
    extra_drop_cols: list[str],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = load_subject_data(
        processed_dir=processed_dir,
        min_windows_per_subject=min_windows_per_subject,
        max_windows_per_subject=max_windows_per_subject,
        seed=seed,
    )

    combined.to_parquet(output_dir / "balanced_dataset.parquet", index=False)

    df, feature_cols = prepare_dataset(
        combined=combined,
        target_col=target_col,
        min_windows_per_subject=min_windows_per_subject,
        target_range=target_range,
        dbp_range=dbp_range,
        extra_drop_cols=extra_drop_cols,
    )

    subjects = np.asarray(sorted(df["subject"].unique()))

    base_subjects, fusion_subjects, test_subjects = split_subjects(
        subjects=subjects,
        base_train_frac=base_train_frac,
        fusion_train_frac=fusion_train_frac,
        test_frac=test_frac,
        seed=seed,
    )

    base_train_df = df[df["subject"].isin(base_subjects)].copy()
    fusion_train_df = df[df["subject"].isin(fusion_subjects)].copy()
    test_df = df[df["subject"].isin(test_subjects)].copy()

    save_metadata(
        output_dir=output_dir,
        dataset_name=dataset_name,
        df=df,
        feature_cols=feature_cols,
        base_train_subjects=base_subjects,
        fusion_train_subjects=fusion_subjects,
        test_subjects=test_subjects,
    )

    base_model = train_base_model(
        train_df=base_train_df,
        feature_cols=feature_cols,
        target_col=target_col,
        seed=seed,
    )

    fusion_train_df = add_base_predictions(
        model=base_model,
        df=fusion_train_df,
        feature_cols=feature_cols,
        target_col=target_col,
        clip_range=prediction_clip,
    )
    test_df = add_base_predictions(
        model=base_model,
        df=test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        clip_range=prediction_clip,
    )

    fusion_train_df.to_parquet(output_dir / "fusion_train_predictions.parquet", index=False)
    test_df.to_parquet(output_dir / "test_predictions.parquet", index=False)

    prior = build_prior_stats(fusion_train_df)
    save_prior_stats(output_dir=output_dir, prior=prior)

    weights_df = learn_adaptive_weights(
        fusion_train_df=fusion_train_df,
        prior=prior,
        k_values=k_values,
        grid_step=grid_step,
        seed=seed,
        target_col=target_col,
        clip_range=prediction_clip,
        offset_clip=offset_clip,
    )

    weights_path = output_dir / "adaptive_weights.csv"
    weights_df.to_csv(weights_path, index=False)

    predictions = evaluate_test_set(
        test_df=test_df,
        prior=prior,
        weights_df=weights_df,
        k_values=k_values,
        n_repeats=n_repeats,
        seed=seed,
        target_col=target_col,
        clip_range=prediction_clip,
        offset_clip=offset_clip,
    )

    predictions_path = output_dir / "test_predictions_all_repeats.csv"
    predictions.to_csv(predictions_path, index=False)

    repeat_summary = summarize_repeats(
        predictions=predictions,
        methods=DEFAULT_METHODS,
        clip_range=prediction_clip,
    )
    repeat_summary_path = output_dir / "summary_by_repeat.csv"
    repeat_summary.to_csv(repeat_summary_path, index=False)

    aggregate = aggregate_summary(
        repeat_summary=repeat_summary,
        dataset_name=dataset_name,
        task_name=task_name,
    )
    aggregate_path = output_dir / "generality_summary.csv"
    aggregate.to_csv(aggregate_path, index=False)

    tests = pairwise_tests(repeat_summary=repeat_summary, k_values=k_values)
    tests_path = output_dir / "pairwise_tests.csv"
    tests.to_csv(tests_path, index=False)

    state_summary = statewise_summary(predictions=predictions, clip_range=prediction_clip)
    state_path = output_dir / "statewise_summary.csv"
    state_summary.to_csv(state_path, index=False)

    plot_calibration_curve(aggregate_df=aggregate, output_dir=output_dir)

    return {
        "balanced_dataset": str(output_dir / "balanced_dataset.parquet"),
        "fusion_train_predictions": str(output_dir / "fusion_train_predictions.parquet"),
        "test_predictions": str(output_dir / "test_predictions.parquet"),
        "adaptive_weights": str(weights_path),
        "test_predictions_all_repeats": str(predictions_path),
        "summary_by_repeat": str(repeat_summary_path),
        "generality_summary": str(aggregate_path),
        "pairwise_tests": str(tests_path),
        "statewise_summary": str(state_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sparse-calibration residual fusion experiment.")

    parser.add_argument("--processed-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)

    parser.add_argument("--dataset-name", default="PulseDB_Vital")
    parser.add_argument("--target-col", default="sbp")
    parser.add_argument("--task-name", default="PPG-to-SBP")

    parser.add_argument("--max-windows-per-subject", type=int, default=80)
    parser.add_argument("--min-windows-per-subject", type=int, default=10)

    parser.add_argument("--base-train-frac", type=float, default=0.60)
    parser.add_argument("--fusion-train-frac", type=float, default=0.20)
    parser.add_argument("--test-frac", type=float, default=0.20)

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--k-values", nargs="+", type=int, default=DEFAULT_K_VALUES)
    parser.add_argument("--n-repeats", type=int, default=50)
    parser.add_argument("--grid-step", type=float, default=0.02)

    parser.add_argument("--min-prediction", type=float, default=PREDICTION_CLIP[0])
    parser.add_argument("--max-prediction", type=float, default=PREDICTION_CLIP[1])
    parser.add_argument("--min-target", type=float, default=60.0)
    parser.add_argument("--max-target", type=float, default=260.0)
    parser.add_argument("--min-dbp", type=float, default=30.0)
    parser.add_argument("--max-dbp", type=float, default=160.0)
    parser.add_argument("--offset-clip", type=float, default=80.0)

    parser.add_argument("--extra-drop-cols", nargs="*", default=[])

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_pipeline(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        target_col=args.target_col,
        task_name=args.task_name,
        max_windows_per_subject=args.max_windows_per_subject,
        min_windows_per_subject=args.min_windows_per_subject,
        base_train_frac=args.base_train_frac,
        fusion_train_frac=args.fusion_train_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        k_values=args.k_values,
        n_repeats=args.n_repeats,
        grid_step=args.grid_step,
        prediction_clip=(args.min_prediction, args.max_prediction),
        target_range=(args.min_target, args.max_target),
        dbp_range=(args.min_dbp, args.max_dbp),
        offset_clip=args.offset_clip,
        extra_drop_cols=args.extra_drop_cols,
    )


if __name__ == "__main__":
    main()
