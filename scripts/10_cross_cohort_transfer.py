import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


EPS_VAR = 1e-6
DEFAULT_K_VALUES = [0, 1, 2]
DEFAULT_TRANSFER_PAIRS = ["EXTERNAL20:MUST", "MUST:EXTERNAL20"]


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
        "RMSE": float(np.sqrt(np.mean(signed_error**2))),
        "MARD": float(np.mean(error / np.maximum(y_true, 1e-9)) * 100.0),
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


def robust_var(values: np.ndarray, fallback: float = 100.0) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) <= 1:
        return float(max(fallback, EPS_VAR))

    iqr = np.percentile(values, 75) - np.percentile(values, 25)
    variance = (iqr / 1.349) ** 2 if iqr > 0 else np.var(values)

    if not np.isfinite(variance) or variance < EPS_VAR:
        variance = max(np.var(values), fallback)

    return float(max(variance, EPS_VAR))


def build_source_stats(source_df: pd.DataFrame) -> dict:
    source_df = source_df.copy()
    source_df["residual"] = source_df["true"] - source_df["base"]

    residuals = source_df["residual"].to_numpy(dtype=float)
    residuals = residuals[np.isfinite(residuals)]

    if len(residuals) == 0:
        raise ValueError("No residuals available for source prior construction.")

    mu_population = float(np.median(residuals))
    sigma_w2 = robust_var(residuals, fallback=100.0)

    subject_offsets = (
        source_df.groupby("subject")["residual"]
        .median()
        .to_numpy(dtype=float)
    )
    sigma_b2 = robust_var(subject_offsets, fallback=sigma_w2)

    var_population = float(sigma_b2 + sigma_w2 / max(len(residuals), 1))

    mu_state = {}
    var_state = {}

    for state, group in source_df.groupby("state"):
        state_residuals = (group["true"] - group["base"]).to_numpy(dtype=float)
        state_residuals = state_residuals[np.isfinite(state_residuals)]

        if len(state_residuals) >= 3:
            mu_state[state] = float(np.median(state_residuals))
            var_state[state] = float(
                robust_var(state_residuals, fallback=sigma_w2)
                + sigma_w2 / len(state_residuals)
            )
        elif len(state_residuals) > 0:
            raw_mu = float(np.median(state_residuals))
            shrink = len(state_residuals) / (len(state_residuals) + 5.0)
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

    mu_local = float(np.mean(residuals)) if len(residuals) < 3 else float(np.median(residuals))

    if len(residuals) >= 2:
        empirical_var = robust_var(residuals, fallback=stats["sigma_w2"])
    else:
        empirical_var = stats["sigma_w2"]

    var_local = float(empirical_var / max(len(residuals), 1) + 0.10 * stats["sigma_w2"])
    clip_width = min(offset_clip, 3.0 * math.sqrt(stats["sigma_b2"] + stats["sigma_w2"]))
    mu_local = float(np.clip(mu_local, stats["mu_pop"] - clip_width, stats["mu_pop"] + clip_width))

    return {
        "mu": mu_local,
        "var": max(var_local, EPS_VAR),
        "k_effective": int(len(residuals)),
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
        precisions.append(1.0 / max(variance, EPS_VAR))

    if len(means) == 0:
        return 0.0

    means = np.asarray(means, dtype=float)
    precisions = np.asarray(precisions, dtype=float)
    weights = precisions / precisions.sum()

    return float(np.sum(weights * means))


def validate_columns(df: pd.DataFrame) -> None:
    required = {"dataset", "subject", "uid", "state", "k", "true", "L0_PPG_only"}
    missing = sorted(required - set(df.columns))

    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def load_predictions(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    validate_columns(df)

    df = df.copy()
    df["dataset"] = df["dataset"].astype(str)
    df["subject"] = df["subject"].astype(str)
    df["uid"] = df["uid"].astype(str)
    df["state"] = df["state"].astype(str)
    df["k"] = pd.to_numeric(df["k"], errors="coerce").astype("Int64")
    df["true"] = pd.to_numeric(df["true"], errors="coerce")
    df["L0_PPG_only"] = pd.to_numeric(df["L0_PPG_only"], errors="coerce")
    df["base"] = df["L0_PPG_only"]

    df = df.dropna(subset=["k", "true", "base"]).copy()
    df["k"] = df["k"].astype(int)

    if len(df) == 0:
        raise ValueError("No valid rows found after loading and filtering.")

    return df.reset_index(drop=True)


def run_cross_transfer(
    df: pd.DataFrame,
    source_dataset: str,
    target_dataset: str,
    k_values: list[int],
    n_repeats: int,
    base_seed: int,
    offset_clip: float,
) -> pd.DataFrame:
    source_df = df[(df["dataset"] == source_dataset) & (df["k"] == 0)].copy()

    if len(source_df) == 0:
        raise ValueError(f"No source rows found for {source_dataset}.")

    stats = build_source_stats(source_df)
    rows = []

    for repeat in range(n_repeats):
        rng = np.random.default_rng(base_seed + repeat)

        for k in k_values:
            target_df = df[(df["dataset"] == target_dataset) & (df["k"] == k)].copy()

            if len(target_df) == 0:
                continue

            for subject, subject_df in target_df.groupby("subject"):
                subject_df = subject_df.copy()

                for _, row in subject_df.iterrows():
                    base = float(row["base"])
                    true = float(row["true"])
                    state = str(row["state"])
                    uid = str(row["uid"])

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
                    offset_full = inverse_variance_fuse([population_evidence, state_evidence, local_ev])

                    rows.append(
                        {
                            "repeat": int(repeat),
                            "source_prior": source_dataset,
                            "target_eval": target_dataset,
                            "k": int(k),
                            "subject": subject,
                            "uid": uid,
                            "state": state,
                            "true": true,
                            "base": base,
                            "L0_PPG_only": float(np.clip(base, 40, 400)),
                            "population_transfer": float(np.clip(base + offset_pop, 40, 400)),
                            "state_transfer": float(np.clip(base + offset_state, 40, 400)),
                            "local_target": float(np.clip(base + offset_local, 40, 400)),
                            "cross_inverse_variance": float(np.clip(base + offset_full, 40, 400)),
                            "local_k_effective": local_ev["k_effective"] if local_ev is not None else 0,
                        }
                    )

    return pd.DataFrame(rows)


def summarize_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    methods = [
        "L0_PPG_only",
        "population_transfer",
        "state_transfer",
        "local_target",
        "cross_inverse_variance",
    ]

    rows = []

    for (source, target, k, repeat), group in predictions.groupby(["source_prior", "target_eval", "k", "repeat"]):
        for method in methods:
            y_true = group["true"].to_numpy(dtype=float)
            y_pred = group[method].to_numpy(dtype=float)
            metrics = compute_metrics(y_true, y_pred)

            rows.append(
                {
                    "source_prior": source,
                    "target_eval": target,
                    "k": int(k),
                    "repeat": int(repeat),
                    "method": method,
                    "n": metrics["n"],
                    "subjects": int(group["subject"].nunique()),
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "MARD": metrics["MARD"],
                    "Bias": metrics["Bias"],
                    "Clarke_AB": clarke_ab_percent(y_true, y_pred),
                }
            )

    repeat_summary = pd.DataFrame(rows)
    aggregate_rows = []

    for (source, target, k, method), group in repeat_summary.groupby(["source_prior", "target_eval", "k", "method"]):
        baseline = repeat_summary[
            (repeat_summary["source_prior"] == source)
            & (repeat_summary["target_eval"] == target)
            & (repeat_summary["k"] == k)
            & (repeat_summary["method"] == "L0_PPG_only")
        ][["repeat", "MAE"]].rename(columns={"MAE": "L0_MAE"})

        merged = pd.merge(group, baseline, on="repeat", how="inner")

        if len(merged) == 0:
            continue

        improvement = (merged["L0_MAE"] - merged["MAE"]) / merged["L0_MAE"] * 100.0

        aggregate_rows.append(
            {
                "source_prior": source,
                "target_eval": target,
                "k": int(k),
                "method": method,
                "n": int(merged["n"].iloc[0]),
                "subjects": int(merged["subjects"].iloc[0]),
                "target_L0_MAE": float(merged["L0_MAE"].mean()),
                "MAE_mean": float(merged["MAE"].mean()),
                "MAE_sd": float(merged["MAE"].std()),
                "RMSE_mean": float(merged["RMSE"].mean()),
                "MARD_mean": float(merged["MARD"].mean()),
                "Bias_mean": float(merged["Bias"].mean()),
                "Clarke_AB_mean": float(merged["Clarke_AB"].mean()),
                "improvement_vs_L0_mean_percent": float(improvement.mean()),
            }
        )

    aggregate = pd.DataFrame(aggregate_rows).sort_values(["source_prior", "target_eval", "k", "MAE_mean"])

    return repeat_summary, aggregate


def build_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    table = aggregate[aggregate["method"] == "cross_inverse_variance"].copy()

    table = table[
        [
            "source_prior",
            "target_eval",
            "k",
            "n",
            "subjects",
            "target_L0_MAE",
            "MAE_mean",
            "improvement_vs_L0_mean_percent",
        ]
    ].rename(
        columns={
            "source_prior": "source_cohort",
            "target_eval": "target_cohort",
            "MAE_mean": "cross_cohort_MAE",
            "improvement_vs_L0_mean_percent": "improvement_vs_L0_percent",
        }
    )

    return table.sort_values(["source_cohort", "target_cohort", "k"]).reset_index(drop=True)


def parse_transfer_pairs(values: list[str]) -> list[tuple[str, str]]:
    pairs = []

    for value in values:
        if ":" not in value:
            raise ValueError("Each transfer pair must use the format source:target.")

        source, target = value.split(":", 1)
        pairs.append((source.strip(), target.strip()))

    return pairs


def run_pipeline(
    input_csv: Path,
    output_dir: Path,
    transfer_pairs: list[tuple[str, str]],
    k_values: list[int],
    n_repeats: int,
    base_seed: int,
    offset_clip: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_predictions(input_csv)

    prediction_parts = [
        run_cross_transfer(
            df=df,
            source_dataset=source,
            target_dataset=target,
            k_values=k_values,
            n_repeats=n_repeats,
            base_seed=base_seed,
            offset_clip=offset_clip,
        )
        for source, target in transfer_pairs
    ]

    predictions = pd.concat(prediction_parts, ignore_index=True)
    repeat_summary, aggregate = summarize_predictions(predictions)
    table = build_table(aggregate)

    predictions_path = output_dir / "cross_cohort_transfer_predictions.csv"
    repeat_summary_path = output_dir / "cross_cohort_transfer_repeat_summary.csv"
    summary_path = output_dir / "cross_cohort_transfer_summary.csv"
    table_path = output_dir / "cross_cohort_transfer_table.csv"

    predictions.to_csv(predictions_path, index=False)
    repeat_summary.to_csv(repeat_summary_path, index=False)
    aggregate.to_csv(summary_path, index=False)
    table.to_csv(table_path, index=False)

    return {
        "predictions": str(predictions_path),
        "repeat_summary": str(repeat_summary_path),
        "summary": str(summary_path),
        "table": str(table_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cross-cohort prior transfer.")

    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)

    parser.add_argument("--transfer-pairs", nargs="+", default=DEFAULT_TRANSFER_PAIRS)
    parser.add_argument("--k-values", nargs="+", type=int, default=DEFAULT_K_VALUES)
    parser.add_argument("--n-repeats", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--offset-clip", type=float, default=120.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_pipeline(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        transfer_pairs=parse_transfer_pairs(args.transfer_pairs),
        k_values=args.k_values,
        n_repeats=args.n_repeats,
        base_seed=args.base_seed,
        offset_clip=args.offset_clip,
    )


if __name__ == "__main__":
    main()
