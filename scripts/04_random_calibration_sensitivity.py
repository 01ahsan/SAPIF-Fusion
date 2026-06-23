import argparse
from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon


DEFAULT_K_VALUES = [0, 1, 2]
DEFAULT_METHODS = [
    "L0_PPG_only",
    "E_pop_only",
    "E_state_only",
    "E_local_only",
    "SA_PIF_full_v1",
    "SA_PIF_adaptive_v2",
]
EPS_VAR = 1e-6


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


def robust_var(values: np.ndarray, fallback: float = 100.0, eps: float = EPS_VAR) -> float:
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


def train_stats(train_df: pd.DataFrame) -> dict:
    train_df = train_df.copy()
    train_df["residual"] = train_df["true"] - train_df["base"]

    residuals = train_df["residual"].to_numpy(dtype=float)
    mu_population = float(np.median(residuals))
    sigma_w2 = robust_var(residuals, fallback=100.0)

    subject_offsets = (
        train_df.groupby("subject")["residual"]
        .median()
        .to_numpy(dtype=float)
    )
    sigma_b2 = robust_var(subject_offsets, fallback=sigma_w2)
    var_population = float(sigma_b2 + sigma_w2 / max(len(residuals), 1))

    mu_state = {}
    var_state = {}

    for state, group in train_df.groupby("state"):
        values = group["residual"].to_numpy(dtype=float)

        if len(values) >= 3:
            mu_state[state] = float(np.median(values))
            var_state[state] = float(robust_var(values, fallback=sigma_w2) + sigma_w2 / len(values))
        elif len(values) > 0:
            raw_mu = float(np.median(values))
            shrink = len(values) / (len(values) + 5.0)
            mu_state[state] = float(mu_population + shrink * (raw_mu - mu_population))
            var_state[state] = float(4.0 * sigma_w2)

    return {
        "mu_pop": mu_population,
        "var_pop": max(var_population, EPS_VAR),
        "mu_state": mu_state,
        "var_state": var_state,
        "sigma_w2": max(sigma_w2, EPS_VAR),
        "sigma_b2": max(sigma_b2, EPS_VAR),
    }


def local_evidence(
    subject_df: pd.DataFrame,
    test_uid: str,
    k: int,
    stats: dict,
    rng: np.random.Generator,
    offset_clip: float,
) -> dict | None:
    if k <= 0:
        return None

    candidates = subject_df[subject_df["uid"] != test_uid].copy()

    if len(candidates) == 0:
        return None

    k_effective = min(k, len(candidates))
    calibration = candidates.sample(
        n=k_effective,
        replace=False,
        random_state=int(rng.integers(0, 10**9)),
    )

    residuals = (calibration["true"] - calibration["base"]).to_numpy(dtype=float)
    residuals = residuals[np.isfinite(residuals)]

    if len(residuals) == 0:
        return None

    if len(residuals) >= 3:
        mu_local = float(np.median(residuals))
    else:
        mu_local = float(np.mean(residuals))

    if len(residuals) >= 2:
        empirical_var = robust_var(residuals, fallback=stats["sigma_w2"])
    else:
        empirical_var = stats["sigma_w2"]

    var_local = float(empirical_var / max(len(residuals), 1) + 0.10 * stats["sigma_w2"])
    mu_population = stats["mu_pop"]
    clip_width = min(offset_clip, 3.0 * math.sqrt(stats["sigma_b2"] + stats["sigma_w2"]))
    mu_local = float(np.clip(mu_local, mu_population - clip_width, mu_population + clip_width))

    return {
        "mu": mu_local,
        "var": max(var_local, EPS_VAR),
        "k": int(len(residuals)),
    }


def fuse_inverse_variance(evidence: dict) -> float:
    means = []
    precisions = []

    for ev in evidence.values():
        if ev is None:
            continue

        mean = float(ev["mu"])
        variance = float(ev["var"])

        if not np.isfinite(mean) or not np.isfinite(variance) or variance <= 0:
            continue

        means.append(mean)
        precisions.append(1.0 / max(variance, EPS_VAR))

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

                weight_local = max(0.0, weight_local)
                grid.append((float(weight_pop), float(weight_state), float(weight_local)))
    else:
        for weight_pop in values:
            weight_state = 1.0 - weight_pop
            grid.append((float(weight_pop), float(weight_state), 0.0))

    return grid


def choose_adaptive_weights(
    train_pred_df: pd.DataFrame,
    k: int,
    grid_with_local: list[tuple[float, float, float]],
    grid_no_local: list[tuple[float, float, float]],
) -> dict:
    grid = grid_no_local if k == 0 else grid_with_local
    y_true = train_pred_df["true"].to_numpy(dtype=float)

    best = None

    for weight_pop, weight_state, weight_local in grid:
        y_pred = (
            train_pred_df["base"].to_numpy(dtype=float)
            + weight_pop * train_pred_df["off_pop"].to_numpy(dtype=float)
            + weight_state * train_pred_df["off_state"].to_numpy(dtype=float)
            + weight_local * train_pred_df["off_local"].to_numpy(dtype=float)
        )

        y_pred = np.clip(y_pred, 40, 400)
        metrics = compute_metrics(y_true, y_pred)
        score = (metrics["MAE"], metrics["RMSE"])

        if best is None or score < best["score"]:
            best = {
                "weights": (weight_pop, weight_state, weight_local),
                "score": score,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
            }

    return best


def validate_columns(df: pd.DataFrame) -> None:
    required = {"dataset", "subject", "uid", "state", "true", "base", "method"}
    missing = sorted(required - set(df.columns))

    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def load_base_records(input_csvs: list[Path]) -> pd.DataFrame:
    frames = []

    for path in input_csvs:
        frame = pd.read_csv(path)
        validate_columns(frame)
        frames.append(frame)

    if not frames:
        raise ValueError("No input files were provided.")

    df_all = pd.concat(frames, ignore_index=True)
    base_df = df_all[df_all["method"] == "L0_PPG_only"].copy()

    base_df = base_df.sort_values(["dataset", "subject", "uid"])
    base_df = base_df.drop_duplicates(subset=["dataset", "subject", "uid"], keep="first")
    base_df = base_df[["dataset", "subject", "uid", "state", "true", "base"]].copy()

    base_df["dataset"] = base_df["dataset"].astype(str)
    base_df["subject"] = base_df["subject"].astype(str)
    base_df["uid"] = base_df["uid"].astype(str)
    base_df["state"] = base_df["state"].astype(str)
    base_df["true"] = pd.to_numeric(base_df["true"], errors="coerce")
    base_df["base"] = pd.to_numeric(base_df["base"], errors="coerce")

    base_df = base_df.dropna(subset=["true", "base"]).reset_index(drop=True)

    if len(base_df) == 0:
        raise ValueError("No valid base records were found.")

    return base_df


def run_one_repeat(
    base_df: pd.DataFrame,
    seed: int,
    k_values: list[int],
    grid_with_local: list[tuple[float, float, float]],
    grid_no_local: list[tuple[float, float, float]],
    offset_clip: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    prediction_parts = []
    weight_rows = []

    for dataset_name, dataset_df in base_df.groupby("dataset"):
        subjects = sorted(dataset_df["subject"].unique())

        for k in k_values:
            evidence_rows = []

            for test_subject in subjects:
                train_df = dataset_df[dataset_df["subject"] != test_subject].copy()
                test_df = dataset_df[dataset_df["subject"] == test_subject].copy()

                if len(train_df) == 0 or len(test_df) == 0:
                    continue

                stats = train_stats(train_df)
                subject_df = test_df.copy()

                for _, row in test_df.iterrows():
                    base = float(row["base"])
                    true = float(row["true"])
                    state = row["state"]
                    uid = row["uid"]

                    population_evidence = {
                        "mu": stats["mu_pop"],
                        "var": stats["var_pop"],
                    }
                    state_evidence = {
                        "mu": stats["mu_state"].get(state, stats["mu_pop"]),
                        "var": stats["var_state"].get(state, 4.0 * stats["sigma_w2"]),
                    }
                    local_ev = local_evidence(
                        subject_df=subject_df,
                        test_uid=uid,
                        k=k,
                        stats=stats,
                        rng=rng,
                        offset_clip=offset_clip,
                    )

                    offset_pop = population_evidence["mu"]
                    offset_state = state_evidence["mu"]
                    offset_local = local_ev["mu"] if local_ev is not None else 0.0
                    offset_full = fuse_inverse_variance(
                        {
                            "pop": population_evidence,
                            "state": state_evidence,
                            "local": local_ev,
                        }
                    )

                    evidence_rows.append(
                        {
                            "dataset": dataset_name,
                            "subject": test_subject,
                            "uid": uid,
                            "state": state,
                            "k": int(k),
                            "true": true,
                            "base": base,
                            "off_pop": offset_pop,
                            "off_state": offset_state,
                            "off_local": offset_local,
                            "L0_PPG_only": float(np.clip(base, 40, 400)),
                            "E_pop_only": float(np.clip(base + offset_pop, 40, 400)),
                            "E_state_only": float(np.clip(base + offset_state, 40, 400)),
                            "E_local_only": float(np.clip(base + offset_local, 40, 400)),
                            "SA_PIF_full_v1": float(np.clip(base + offset_full, 40, 400)),
                        }
                    )

            evidence_df = pd.DataFrame(evidence_rows)

            for test_subject in subjects:
                train_evidence = evidence_df[evidence_df["subject"] != test_subject].copy()
                test_evidence = evidence_df[evidence_df["subject"] == test_subject].copy()

                if len(train_evidence) == 0 or len(test_evidence) == 0:
                    continue

                best = choose_adaptive_weights(
                    train_pred_df=train_evidence,
                    k=k,
                    grid_with_local=grid_with_local,
                    grid_no_local=grid_no_local,
                )
                weight_pop, weight_state, weight_local = best["weights"]

                adaptive_pred = (
                    test_evidence["base"].to_numpy(dtype=float)
                    + weight_pop * test_evidence["off_pop"].to_numpy(dtype=float)
                    + weight_state * test_evidence["off_state"].to_numpy(dtype=float)
                    + weight_local * test_evidence["off_local"].to_numpy(dtype=float)
                )
                adaptive_pred = np.clip(adaptive_pred, 40, 400)

                test_evidence = test_evidence.copy()
                test_evidence["SA_PIF_adaptive_v2"] = adaptive_pred
                test_evidence["repeat_seed"] = seed
                test_evidence["w_pop"] = weight_pop
                test_evidence["w_state"] = weight_state
                test_evidence["w_local"] = weight_local

                prediction_parts.append(test_evidence)

                weight_rows.append(
                    {
                        "repeat_seed": seed,
                        "dataset": dataset_name,
                        "k": int(k),
                        "test_subject": test_subject,
                        "w_pop": weight_pop,
                        "w_state": weight_state,
                        "w_local": weight_local,
                    }
                )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    weights = pd.DataFrame(weight_rows)

    return predictions, weights


def summarize_by_repeat(predictions: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    rows = []

    for (seed, dataset, k), group in predictions.groupby(["repeat_seed", "dataset", "k"]):
        y_true = group["true"].to_numpy(dtype=float)

        for method in methods:
            y_pred = group[method].to_numpy(dtype=float)
            metrics = compute_metrics(y_true, y_pred)
            ab_percent = clarke_ab_percent(y_true, y_pred)

            rows.append(
                {
                    "repeat_seed": int(seed),
                    "dataset": dataset,
                    "k": int(k),
                    "method": method,
                    "n": metrics["n"],
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "MARD": metrics["MARD"],
                    "Bias": metrics["Bias"],
                    "Clarke_AB_percent": ab_percent,
                }
            )

    return pd.DataFrame(rows)


def aggregate_summary(repeat_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (dataset, k, method), group in repeat_summary.groupby(["dataset", "k", "method"]):
        baseline = repeat_summary[
            (repeat_summary["dataset"] == dataset)
            & (repeat_summary["k"] == k)
            & (repeat_summary["method"] == "L0_PPG_only")
        ][["repeat_seed", "MAE"]].rename(columns={"MAE": "baseline_MAE"})

        current = group[["repeat_seed", "MAE", "RMSE", "MARD", "Bias", "Clarke_AB_percent"]]
        merged = pd.merge(current, baseline, on="repeat_seed")

        improvement = (merged["baseline_MAE"] - merged["MAE"]) / merged["baseline_MAE"] * 100.0

        rows.append(
            {
                "dataset": dataset,
                "k": int(k),
                "method": method,
                "MAE_mean": float(merged["MAE"].mean()),
                "MAE_std": float(merged["MAE"].std()),
                "MAE_median": float(merged["MAE"].median()),
                "MAE_IQR25": float(merged["MAE"].quantile(0.25)),
                "MAE_IQR75": float(merged["MAE"].quantile(0.75)),
                "RMSE_mean": float(merged["RMSE"].mean()),
                "MARD_mean": float(merged["MARD"].mean()),
                "Bias_mean": float(merged["Bias"].mean()),
                "Clarke_AB_mean": float(merged["Clarke_AB_percent"].mean()),
                "Improvement_vs_L0_mean_percent": float(improvement.mean()),
                "Improvement_vs_L0_std_percent": float(improvement.std()),
            }
        )

    return pd.DataFrame(rows).sort_values(["dataset", "k", "MAE_mean"])


def repeat_level_tests(
    repeat_summary: pd.DataFrame,
    k_values: list[int],
    target_method: str,
    baselines: list[str],
) -> pd.DataFrame:
    rows = []

    for dataset in sorted(repeat_summary["dataset"].unique()):
        for k in k_values:
            for baseline in baselines:
                base_df = repeat_summary[
                    (repeat_summary["dataset"] == dataset)
                    & (repeat_summary["k"] == k)
                    & (repeat_summary["method"] == baseline)
                ][["repeat_seed", "MAE"]].rename(columns={"MAE": "MAE_base"})

                target_df = repeat_summary[
                    (repeat_summary["dataset"] == dataset)
                    & (repeat_summary["k"] == k)
                    & (repeat_summary["method"] == target_method)
                ][["repeat_seed", "MAE"]].rename(columns={"MAE": "MAE_target"})

                merged = pd.merge(base_df, target_df, on="repeat_seed")
                diff = merged["MAE_base"].to_numpy(dtype=float) - merged["MAE_target"].to_numpy(dtype=float)

                try:
                    stat, p_value = wilcoxon(diff)
                except Exception:
                    stat, p_value = np.nan, np.nan

                rows.append(
                    {
                        "dataset": dataset,
                        "k": int(k),
                        "comparison": f"{target_method} vs {baseline}",
                        "repeats": int(len(merged)),
                        "mean_MAE_base": float(merged["MAE_base"].mean()),
                        "mean_MAE_target": float(merged["MAE_target"].mean()),
                        "mean_delta_MAE": float(diff.mean()),
                        "median_delta_MAE": float(np.median(diff)),
                        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                        "significant_p05": bool(p_value < 0.05) if np.isfinite(p_value) else False,
                    }
                )

    return pd.DataFrame(rows)


def plot_summary(aggregate_df: pd.DataFrame, output_dir: Path) -> None:
    for dataset in sorted(aggregate_df["dataset"].unique()):
        plt.figure(figsize=(8.5, 5.2))

        for method in ["L0_PPG_only", "E_local_only", "SA_PIF_full_v1", "SA_PIF_adaptive_v2"]:
            subset = aggregate_df[
                (aggregate_df["dataset"] == dataset)
                & (aggregate_df["method"] == method)
            ].sort_values("k")

            if len(subset) == 0:
                continue

            plt.errorbar(
                subset["k"],
                subset["MAE_mean"],
                yerr=subset["MAE_std"],
                marker="o",
                linewidth=2,
                capsize=4,
                label=method,
            )

        plt.xlabel("Calibration samples per subject")
        plt.ylabel("MAE")
        plt.title(f"Random calibration sensitivity: {dataset}")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()

        plt.savefig(output_dir / f"random_sensitivity_{dataset}.png", dpi=300, bbox_inches="tight")
        plt.savefig(output_dir / f"random_sensitivity_{dataset}.pdf", bbox_inches="tight")
        plt.close()


def run_pipeline(
    input_csvs: list[Path],
    output_dir: Path,
    k_values: list[int],
    n_repeats: int,
    grid_step: float,
    base_seed: int,
    offset_clip: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    base_df = load_base_records(input_csvs)
    grid_with_local = simplex_grid(grid_step, allow_local=True)
    grid_no_local = simplex_grid(grid_step, allow_local=False)

    prediction_parts = []
    weight_parts = []

    for repeat_idx in range(n_repeats):
        seed = base_seed + repeat_idx

        predictions, weights = run_one_repeat(
            base_df=base_df,
            seed=seed,
            k_values=k_values,
            grid_with_local=grid_with_local,
            grid_no_local=grid_no_local,
            offset_clip=offset_clip,
        )

        prediction_parts.append(predictions)
        weight_parts.append(weights)

    all_predictions = pd.concat(prediction_parts, ignore_index=True)
    all_weights = pd.concat(weight_parts, ignore_index=True)

    prediction_path = output_dir / "random_sensitivity_predictions.csv"
    weight_path = output_dir / "random_sensitivity_weights.csv"

    all_predictions.to_csv(prediction_path, index=False)
    all_weights.to_csv(weight_path, index=False)

    repeat_summary = summarize_by_repeat(all_predictions, methods=DEFAULT_METHODS)
    repeat_summary_path = output_dir / "random_sensitivity_repeat_summary.csv"
    repeat_summary.to_csv(repeat_summary_path, index=False)

    aggregate_df = aggregate_summary(repeat_summary)
    aggregate_path = output_dir / "random_sensitivity_summary.csv"
    aggregate_df.to_csv(aggregate_path, index=False)

    tests_df = repeat_level_tests(
        repeat_summary=repeat_summary,
        k_values=k_values,
        target_method="SA_PIF_adaptive_v2",
        baselines=["L0_PPG_only", "E_local_only", "SA_PIF_full_v1"],
    )
    tests_path = output_dir / "random_sensitivity_tests.csv"
    tests_df.to_csv(tests_path, index=False)

    weight_summary = (
        all_weights.groupby(["dataset", "k"])[["w_pop", "w_state", "w_local"]]
        .agg(["mean", "std"])
    )
    weight_summary.columns = ["_".join(column) for column in weight_summary.columns]
    weight_summary = weight_summary.reset_index()

    weight_summary_path = output_dir / "random_sensitivity_weight_summary.csv"
    weight_summary.to_csv(weight_summary_path, index=False)

    plot_summary(aggregate_df, output_dir=output_dir)

    return {
        "predictions": str(prediction_path),
        "weights": str(weight_path),
        "repeat_summary": str(repeat_summary_path),
        "summary": str(aggregate_path),
        "tests": str(tests_path),
        "weight_summary": str(weight_summary_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run random calibration sensitivity analysis.")

    parser.add_argument("--input-csvs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)

    parser.add_argument("--k-values", nargs="+", type=int, default=DEFAULT_K_VALUES)
    parser.add_argument("--n-repeats", type=int, default=100)
    parser.add_argument("--grid-step", type=float, default=0.02)
    parser.add_argument("--offset-clip", type=float, default=120.0)
    parser.add_argument("--base-seed", type=int, default=2026)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_pipeline(
        input_csvs=args.input_csvs,
        output_dir=args.output_dir,
        k_values=args.k_values,
        n_repeats=args.n_repeats,
        grid_step=args.grid_step,
        base_seed=args.base_seed,
        offset_clip=args.offset_clip,
    )


if __name__ == "__main__":
    main()
