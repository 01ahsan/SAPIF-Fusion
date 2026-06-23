import argparse
import json
import math
import os
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import find_peaks, resample
from scipy.stats import pearsonr, wilcoxon
from tqdm import tqdm

warnings.filterwarnings("ignore")


@dataclass
class ExperimentConfig:
    dataset: str
    data_path: str
    tflite_path: str
    scaler_path: str
    output_dir: str
    subject_key: str
    glucose_keys: List[str]
    uid_keys: List[str]
    condition_default: str
    adapt_ppg: bool
    seed: int = 42
    fs: float = 100.0
    target_len: int = 1000
    n_features: int = 12
    k_values: Tuple[int, ...] = (0, 1, 2)
    cal_selection_mode: str = "diverse"
    offset_clip: float = 120.0
    eps_var: float = 1e-6
    show_progress: bool = True


class TFLiteGlucoseModel:
    def __init__(self, model_path: str, scaler_path: str, config: ExperimentConfig):
        self.config = config
        self.scaler = np.load(scaler_path, allow_pickle=True).item()
        self.sc_mean = np.asarray(self.scaler["mean"], dtype=np.float32)
        self.sc_scale = np.asarray(self.scaler["scale"], dtype=np.float32)
        self.lbl_mean = float(self.scaler.get("lbl_mean", 0.0))
        self.lbl_std = float(self.scaler.get("lbl_std", 1.0))

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]

    def standardize_features(self, features: np.ndarray) -> np.ndarray:
        return (features - self.sc_mean) / (self.sc_scale + 1e-9)

    def predict(self, window_raw: Any) -> float:
        if window_raw is None:
            return np.nan

        if self.config.adapt_ppg:
            signal = adapt_ppg(window_raw, target_len=self.config.target_len)
        else:
            signal = np.asarray(window_raw, dtype=np.float32).reshape(-1)
            if len(signal) != self.config.target_len:
                signal = adapt_ppg(signal, target_len=self.config.target_len)

        if signal is None or len(signal) != self.config.target_len:
            return np.nan

        features = extract_features(signal, fs=self.config.fs)
        features_scaled = self.standardize_features(features)
        model_input = np.concatenate([signal, features_scaled]).astype(np.float32).reshape(1, -1)

        try:
            self.interpreter.set_tensor(self.input_detail["index"], model_input)
            self.interpreter.invoke()
            raw = float(self.interpreter.get_tensor(self.output_detail["index"]).squeeze())
            pred = raw * self.lbl_std + self.lbl_mean
            return float(np.clip(pred, 40, 400))
        except Exception:
            return np.nan

    def count_params(self) -> int:
        total = 0
        for tensor in self.interpreter.get_tensor_details():
            shape = tensor.get("shape", [])
            dtype = tensor.get("dtype", None)
            if len(shape) > 0 and dtype == np.float32:
                total += int(np.prod(shape))
        return total


def parse_list(value: str, cast_type=str) -> Tuple:
    if value is None or value == "":
        return tuple()
    return tuple(cast_type(v.strip()) for v in value.split(",") if v.strip() != "")


def validate_paths(config: ExperimentConfig) -> None:
    for path, name in [
        (config.data_path, "data_path"),
        (config.tflite_path, "tflite_path"),
        (config.scaler_path, "scaler_path"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")
    os.makedirs(config.output_dir, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def adapt_ppg(ppg_raw: Any, target_len: int = 1000) -> Optional[np.ndarray]:
    signal = np.asarray(ppg_raw, dtype=np.float64).reshape(-1)

    if len(signal) == 0:
        return None

    if len(signal) >= 2:
        signal = signal - np.linspace(signal[0], signal[-1], len(signal))

    if len(signal) == target_len:
        output = signal
    elif len(signal) > target_len:
        start = (len(signal) - target_len) // 2
        output = signal[start : start + target_len]
    else:
        output = resample(signal, target_len)

    std = float(np.std(output))
    if std < 1e-9:
        return None

    return ((output - np.mean(output)) / std).astype(np.float32)


def extract_features(ppg_adapted: Any, fs: float = 100.0) -> np.ndarray:
    ppg = np.asarray(ppg_adapted, dtype=np.float64).reshape(-1)
    features = np.zeros(12, dtype=np.float64)

    try:
        min_dist = int(0.4 * fs)
        prominence = max(0.1, 0.3 * (ppg.max() - ppg.min()))
        peaks, _ = find_peaks(ppg, distance=min_dist, prominence=prominence)

        valleys = []
        for i in range(len(peaks) - 1):
            segment = ppg[peaks[i] : peaks[i + 1]]
            if len(segment) > 0:
                valleys.append(peaks[i] + np.argmin(segment))
        valleys = np.asarray(valleys, dtype=int)

        if len(peaks) >= 2:
            ibi = np.diff(peaks) / fs
            features[0] = 60.0 / (np.mean(ibi) + 1e-9)
            features[1] = np.std(ibi) * 1000.0
        else:
            features[0] = 75.0

        if len(peaks) > 0 and len(valleys) > 0:
            n = min(len(peaks), len(valleys))
            amplitude = ppg[peaks[:n]] - ppg[valleys[:n]]
            features[2] = np.mean(amplitude)
            features[3] = np.std(amplitude) if len(amplitude) > 1 else 0.0
        else:
            features[2] = ppg.max() - ppg.min()

        rise_times = []
        decay_times = []
        for peak in peaks:
            previous_valleys = valleys[valleys < peak]
            next_valleys = valleys[valleys > peak]
            if len(previous_valleys):
                rise_times.append((peak - previous_valleys[-1]) / fs)
            if len(next_valleys):
                decay_times.append((next_valleys[0] - peak) / fs)

        features[4] = np.mean(rise_times) if rise_times else 0.0
        features[5] = np.mean(decay_times) if decay_times else 0.0

        pulse_widths = []
        for peak in peaks:
            previous_valleys = valleys[valleys < peak]
            next_valleys = valleys[valleys > peak]
            if not len(previous_valleys) or not len(next_valleys):
                continue
            foot = previous_valleys[-1]
            next_valley = next_valleys[0]
            threshold = ppg[foot] + 0.5 * (ppg[peak] - ppg[foot])
            above = np.where(ppg[foot : next_valley + 1] >= threshold)[0]
            if len(above) >= 2:
                pulse_widths.append((above[-1] - above[0]) / fs)

        features[6] = np.mean(pulse_widths) if pulse_widths else 0.0

        auc_values = []
        for peak in peaks:
            previous_valleys = valleys[valleys < peak]
            next_valleys = valleys[valleys > peak]
            if not len(previous_valleys) or not len(next_valleys):
                continue
            foot = previous_valleys[-1]
            next_valley = next_valleys[0]
            auc_values.append(np.trapezoid(ppg[foot : next_valley + 1] - ppg[foot]))

        features[7] = np.mean(auc_values) if auc_values else 0.0
        features[8] = np.mean(ppg)
        features[9] = np.std(ppg)

        vpg = np.diff(ppg)
        features[10] = np.mean(np.abs(vpg)) if len(vpg) else 0.0

        apg = np.diff(vpg)
        if len(apg) > 5 and np.std(apg) > 0:
            positive_peaks, _ = find_peaks(apg, prominence=np.std(apg) * 0.3)
            negative_peaks, _ = find_peaks(-apg, prominence=np.std(apg) * 0.3)
            if len(positive_peaks) and len(negative_peaks):
                a_peak = apg[positive_peaks[0]]
                b_candidates = negative_peaks[negative_peaks > positive_peaks[0]]
                if len(b_candidates):
                    features[11] = apg[b_candidates[0]] / (a_peak + 1e-9)

    except Exception:
        pass

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def get_first_existing(record: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def load_records(config: ExperimentConfig) -> List[Dict[str, Any]]:
    with open(config.data_path, "r", encoding="utf-8") as file:
        raw_records = json.load(file)

    valid_records = []

    for record in raw_records:
        subject = record.get(config.subject_key)
        glucose = get_first_existing(record, config.glucose_keys)

        if subject is None or glucose is None:
            continue

        window_data = record.get("window_data", [])
        if isinstance(window_data, str) or window_data is None:
            window_data = []
        if not isinstance(window_data, list):
            try:
                window_data = list(window_data)
            except Exception:
                window_data = []

        if len(window_data) == 0:
            continue

        condition = str(record.get("condition", config.condition_default))
        uid = get_first_existing(
            record,
            config.uid_keys,
            default=f"{subject}_{len(valid_records)}",
        )

        valid_records.append(
            {
                "subject": subject,
                "uid": uid,
                "condition": condition,
                "glucose": float(glucose),
                "window_data": window_data,
            }
        )

    if len(valid_records) == 0:
        raise RuntimeError("No valid records found. Check subject, glucose, and window fields.")

    return valid_records


def group_by_subject(records: List[Dict[str, Any]]) -> Tuple[Dict[Any, List[Dict[str, Any]]], List[str]]:
    state_names = sorted(set(record["condition"] for record in records))
    condition_to_state = {condition: index for index, condition in enumerate(state_names)}
    subject_data = defaultdict(list)

    for record in records:
        subject_data[record["subject"]].append(
            {
                "uid": record["uid"],
                "state": condition_to_state[record["condition"]],
                "condition": record["condition"],
                "true": float(record["glucose"]),
                "window": record["window_data"],
            }
        )

    return subject_data, state_names


def cache_base_predictions(
    subject_data: Dict[Any, List[Dict[str, Any]]],
    model: TFLiteGlucoseModel,
    show_progress: bool = True,
) -> Dict[Tuple[Any, int], float]:
    base_cache = {}
    iterator = subject_data.items()

    if show_progress:
        iterator = tqdm(iterator, desc="Base inference")

    for subject, samples in iterator:
        for index, sample in enumerate(samples):
            base_cache[(subject, index)] = model.predict(sample["window"])

    return base_cache


def compute_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

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

    absolute_error = np.abs(y_pred - y_true)
    signed_error = y_pred - y_true

    if len(y_true) < 3 or np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        pearson_r = np.nan
    else:
        try:
            pearson_r = float(pearsonr(y_pred, y_true)[0])
        except Exception:
            pearson_r = np.nan

    return {
        "n": int(len(y_true)),
        "MAE": float(np.mean(absolute_error)),
        "MedianAE": float(np.median(absolute_error)),
        "RMSE": float(np.sqrt(np.mean(signed_error**2))),
        "MARD": float(np.mean(absolute_error / np.maximum(y_true, 1e-9)) * 100),
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


def clarke_counts(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    zones = [clarke_zone(true, pred) for true, pred in zip(y_true, y_pred)]
    counts = {zone: zones.count(zone) for zone in "ABCDE"}
    counts["n"] = len(zones)
    counts["AB"] = counts["A"] + counts["B"]
    counts["AB_percent"] = 100.0 * counts["AB"] / len(zones) if zones else np.nan
    return counts


def robust_var(x: Sequence[float], eps_var: float, fallback: float = 100.0) -> float:
    values = np.asarray(x, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) <= 1:
        return float(max(fallback, eps_var))

    iqr = np.percentile(values, 75) - np.percentile(values, 25)
    variance = (iqr / 1.349) ** 2 if iqr > 0 else np.var(values)

    if not np.isfinite(variance) or variance < eps_var:
        variance = max(np.var(values), fallback)

    return float(max(variance, eps_var))


def create_accessors(subject_data: Dict[Any, List[Dict[str, Any]]], base_cache: Dict[Tuple[Any, int], float]):
    def base(subject, index):
        return base_cache[(subject, index)]

    def true(subject, index):
        return float(subject_data[subject][index]["true"])

    def state(subject, index):
        return int(subject_data[subject][index]["state"])

    def condition(subject, index):
        return str(subject_data[subject][index]["condition"])

    def uid(subject, index):
        return str(subject_data[subject][index]["uid"])

    def n_samples(subject):
        return len(subject_data[subject])

    return base, true, state, condition, uid, n_samples


def train_stats(
    train_subjects: Sequence[Any],
    state_names: Sequence[str],
    accessors,
    config: ExperimentConfig,
) -> Dict[str, Any]:
    base, true, _, condition, _, n_samples = accessors
    all_residuals = []
    state_residuals = defaultdict(list)
    subject_offsets = []

    for subject in train_subjects:
        subject_residuals = []
        for index in range(n_samples(subject)):
            b = base(subject, index)
            t = true(subject, index)

            if not np.isfinite(b) or not np.isfinite(t):
                continue

            residual = float(t - b)
            state_name = condition(subject, index)
            all_residuals.append(residual)
            state_residuals[state_name].append(residual)
            subject_residuals.append(residual)

        if len(subject_residuals) > 0:
            subject_offsets.append(float(np.median(subject_residuals)))

    if len(all_residuals) == 0:
        mu_pop = 0.0
        sigma_w2 = 100.0
        sigma_b2 = 100.0
    else:
        mu_pop = float(np.median(all_residuals))
        sigma_w2 = robust_var(all_residuals, eps_var=config.eps_var, fallback=100.0)
        sigma_b2 = robust_var(subject_offsets, eps_var=config.eps_var, fallback=sigma_w2)

    var_pop = float(sigma_b2 + sigma_w2 / max(len(all_residuals), 1))
    mu_state = {}
    var_state = {}
    n_state = {}

    for state_name in state_names:
        values = np.asarray(state_residuals.get(state_name, []), dtype=float)
        values = values[np.isfinite(values)]
        n_state[state_name] = int(len(values))

        if len(values) >= 3:
            mu_state[state_name] = float(np.median(values))
            var_state[state_name] = float(
                robust_var(values, eps_var=config.eps_var, fallback=sigma_w2) + sigma_w2 / len(values)
            )
        elif len(values) > 0:
            raw_mu = float(np.median(values))
            shrinkage = len(values) / (len(values) + 5.0)
            mu_state[state_name] = float(mu_pop + shrinkage * (raw_mu - mu_pop))
            var_state[state_name] = float(4.0 * sigma_w2)
        else:
            mu_state[state_name] = float(mu_pop)
            var_state[state_name] = float(4.0 * sigma_w2)

    return {
        "mu_pop": float(mu_pop),
        "var_pop": float(max(var_pop, config.eps_var)),
        "mu_state": mu_state,
        "var_state": var_state,
        "sigma_w2": float(max(sigma_w2, config.eps_var)),
        "sigma_b2": float(max(sigma_b2, config.eps_var)),
        "n_train_resids": int(len(all_residuals)),
        "n_state": n_state,
    }


def select_calibration_indices(
    subject: Any,
    k: int,
    exclude: Optional[Sequence[int]],
    mode: str,
    accessors,
    seed: int,
) -> List[int]:
    base, true, _, condition, _, n_samples = accessors
    exclude_set = set(exclude or [])
    valid = []

    for index in range(n_samples(subject)):
        if index in exclude_set:
            continue
        b = base(subject, index)
        t = true(subject, index)
        if np.isfinite(b) and np.isfinite(t):
            valid.append(index)

    if k <= 0:
        return []

    if len(valid) <= k:
        return valid

    if mode == "random":
        rng = np.random.default_rng(seed + abs(hash(str(subject))) % 100000)
        return list(rng.choice(valid, size=k, replace=False))

    selected = []
    glucose_values = np.asarray([true(subject, index) for index in valid], dtype=float)
    median_glucose = np.median(glucose_values)
    seed_index = valid[int(np.argmin(np.abs(glucose_values - median_glucose)))]
    selected.append(seed_index)

    while len(selected) < k:
        remaining = [index for index in valid if index not in selected]
        if not remaining:
            break

        selected_glucose = np.asarray([true(subject, index) for index in selected], dtype=float)
        selected_states = set(condition(subject, index) for index in selected)
        scores = []

        for index in remaining:
            glucose = true(subject, index)
            state_name = condition(subject, index)
            glucose_spread = float(np.min(np.abs(selected_glucose - glucose)))
            state_bonus = 20.0 if state_name not in selected_states else 0.0
            scores.append(glucose_spread + state_bonus)

        selected.append(remaining[int(np.argmax(scores))])

    return selected


def local_evidence(subject: Any, cal_indices: Sequence[int], stats: Dict[str, Any], accessors, config: ExperimentConfig):
    if len(cal_indices) == 0:
        return None

    base, true, _, _, _, _ = accessors
    residuals = []

    for index in cal_indices:
        b = base(subject, index)
        t = true(subject, index)
        if np.isfinite(b) and np.isfinite(t):
            residuals.append(float(t - b))

    if len(residuals) == 0:
        return None

    mu_local = float(np.median(residuals)) if len(residuals) >= 3 else float(np.mean(residuals))
    k = len(residuals)

    if k >= 2:
        empirical_var = robust_var(residuals, eps_var=config.eps_var, fallback=stats["sigma_w2"])
    else:
        empirical_var = stats["sigma_w2"]

    var_local = float(empirical_var / max(k, 1) + 0.10 * stats["sigma_w2"])
    mu_pop = stats["mu_pop"]
    clip_width = min(config.offset_clip, 3.0 * math.sqrt(stats["sigma_b2"] + stats["sigma_w2"]))
    mu_local = float(np.clip(mu_local, mu_pop - clip_width, mu_pop + clip_width))

    return {"mu": mu_local, "var": float(max(var_local, config.eps_var)), "k": int(k)}


def fuse_offsets(evidence_dict: Dict[str, Optional[Dict[str, float]]], eps_var: float) -> Tuple[float, Dict[str, float]]:
    names = []
    mus = []
    precisions = []

    for name, evidence in evidence_dict.items():
        if evidence is None:
            continue

        mu = float(evidence["mu"])
        var = float(evidence["var"])

        if not np.isfinite(mu) or not np.isfinite(var) or var <= 0:
            continue

        names.append(name)
        mus.append(mu)
        precisions.append(1.0 / max(var, eps_var))

    if len(mus) == 0:
        return 0.0, {}

    mus = np.asarray(mus, dtype=float)
    precisions = np.asarray(precisions, dtype=float)
    weights = precisions / np.sum(precisions)
    fused = float(np.sum(weights * mus))
    weight_map = {name: float(weight) for name, weight in zip(names, weights)}

    return fused, weight_map


def predict_with_evidence(base_pred: float, evidence_dict: Dict[str, Optional[Dict[str, float]]], eps_var: float):
    offset, weight_map = fuse_offsets(evidence_dict, eps_var=eps_var)
    prediction = float(np.clip(base_pred + offset, 40, 400))
    return prediction, offset, weight_map


def run_loso_experiment(
    subjects: Sequence[Any],
    subject_data: Dict[Any, List[Dict[str, Any]]],
    state_names: Sequence[str],
    base_cache: Dict[Tuple[Any, int], float],
    config: ExperimentConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    accessors = create_accessors(subject_data, base_cache)
    base, true, _, condition, uid, n_samples = accessors
    prediction_rows = []
    weight_rows = []
    iterator = subjects

    if config.show_progress:
        iterator = tqdm(subjects, desc="SA-PIF LOSO")

    for test_subject in iterator:
        train_subjects = [subject for subject in subjects if subject != test_subject]
        if len(train_subjects) == 0:
            continue

        stats = train_stats(train_subjects, state_names, accessors, config)

        for k in config.k_values:
            for test_index in range(n_samples(test_subject)):
                base_pred = base(test_subject, test_index)
                true_value = true(test_subject, test_index)

                if not np.isfinite(base_pred) or not np.isfinite(true_value):
                    continue

                cal_indices = select_calibration_indices(
                    subject=test_subject,
                    k=k,
                    exclude=[test_index],
                    mode=config.cal_selection_mode,
                    accessors=accessors,
                    seed=config.seed,
                )

                if k > 0 and len(cal_indices) == 0:
                    continue

                state_name = condition(test_subject, test_index)
                pop_ev = {"mu": stats["mu_pop"], "var": stats["var_pop"]}
                state_ev = {
                    "mu": stats["mu_state"].get(state_name, stats["mu_pop"]),
                    "var": stats["var_state"].get(state_name, 4.0 * stats["sigma_w2"]),
                }
                local_ev = local_evidence(test_subject, cal_indices, stats, accessors, config)

                method_evidence = {
                    "L0_PPG_only": {},
                    "E_pop_only": {"pop": pop_ev},
                    "E_state_only": {"state": state_ev},
                    "E_local_only": {"local": local_ev},
                    "E_pop_state": {"pop": pop_ev, "state": state_ev},
                    "E_pop_local": {"pop": pop_ev, "local": local_ev},
                    "E_state_local": {"state": state_ev, "local": local_ev},
                    "SA_PIF_full": {"pop": pop_ev, "state": state_ev, "local": local_ev},
                    "Missing_state": {"pop": pop_ev, "local": local_ev},
                    "Missing_local": {"pop": pop_ev, "state": state_ev},
                    "Missing_population": {"state": state_ev, "local": local_ev},
                    "Uncertain_state_x9var": {
                        "pop": pop_ev,
                        "state": {"mu": state_ev["mu"], "var": state_ev["var"] * 9.0},
                        "local": local_ev,
                    },
                }

                for method, evidence in method_evidence.items():
                    if method == "L0_PPG_only":
                        prediction = float(np.clip(base_pred, 40, 400))
                        offset = 0.0
                        weight_map = {}
                    else:
                        prediction, offset, weight_map = predict_with_evidence(
                            base_pred,
                            evidence,
                            eps_var=config.eps_var,
                        )

                    prediction_rows.append(
                        {
                            "dataset": config.dataset,
                            "subject": str(test_subject),
                            "uid": uid(test_subject, test_index),
                            "test_idx": int(test_index),
                            "state": state_name,
                            "k": int(k),
                            "method": method,
                            "true": float(true_value),
                            "base": float(base_pred),
                            "pred": float(prediction),
                            "offset": float(offset),
                            "n_cal": int(len(cal_indices)),
                            "cal_indices": ",".join(map(str, cal_indices)),
                            "mu_pop": float(stats["mu_pop"]),
                            "var_pop": float(stats["var_pop"]),
                            "mu_state": float(state_ev["mu"]),
                            "var_state": float(state_ev["var"]),
                            "sigma_w2": float(stats["sigma_w2"]),
                            "sigma_b2": float(stats["sigma_b2"]),
                        }
                    )

                    if method == "SA_PIF_full":
                        row = {
                            "dataset": config.dataset,
                            "subject": str(test_subject),
                            "uid": uid(test_subject, test_index),
                            "state": state_name,
                            "k": int(k),
                            "n_cal": int(len(cal_indices)),
                        }
                        for key in ["pop", "state", "local"]:
                            row[f"w_{key}"] = float(weight_map.get(key, 0.0))
                        weight_rows.append(row)

    return pd.DataFrame(prediction_rows), pd.DataFrame(weight_rows)


def create_summary(pred_df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    summary_rows = []

    for (method, k), group in pred_df.groupby(["method", "k"]):
        metrics = compute_metrics(group["true"].to_numpy(), group["pred"].to_numpy())
        clarke = clarke_counts(group["true"].to_numpy(), group["pred"].to_numpy())
        summary_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "k": int(k),
                "n": metrics["n"],
                "MAE": metrics["MAE"],
                "MedianAE": metrics["MedianAE"],
                "RMSE": metrics["RMSE"],
                "MARD": metrics["MARD"],
                "Bias": metrics["Bias"],
                "Pearson_r": metrics["Pearson_r"],
                "Clarke_A": clarke["A"],
                "Clarke_B": clarke["B"],
                "Clarke_AB": clarke["AB"],
                "Clarke_AB_percent": clarke["AB_percent"],
            }
        )

    return pd.DataFrame(summary_rows).sort_values(["k", "MAE"])


def run_pairwise_tests(pred_df: pd.DataFrame, dataset: str, k_values: Sequence[int]) -> pd.DataFrame:
    test_rows = []

    for k in k_values:
        sub = pred_df[pred_df["k"] == k]

        for method in sorted(sub["method"].unique()):
            if method == "L0_PPG_only":
                continue

            for baseline in ["L0_PPG_only", "E_pop_only", "E_state_only", "E_local_only"]:
                if baseline == method:
                    continue

                baseline_df = sub[sub["method"] == baseline][["subject", "uid", "true", "pred"]].rename(
                    columns={"pred": "pred_base"}
                )
                method_df = sub[sub["method"] == method][["subject", "uid", "true", "pred"]].rename(
                    columns={"pred": "pred_method"}
                )
                merged = pd.merge(baseline_df, method_df, on=["subject", "uid", "true"])

                if len(merged) < 5:
                    continue

                err_base = np.abs(merged["pred_base"].to_numpy() - merged["true"].to_numpy())
                err_method = np.abs(merged["pred_method"].to_numpy() - merged["true"].to_numpy())
                diff = err_base - err_method
                nonzero = diff != 0

                if nonzero.sum() >= 3:
                    try:
                        stat, p_value = wilcoxon(diff[nonzero])
                    except Exception:
                        stat, p_value = np.nan, np.nan
                else:
                    stat, p_value = np.nan, np.nan

                test_rows.append(
                    {
                        "dataset": dataset,
                        "k": int(k),
                        "method": method,
                        "baseline": baseline,
                        "paired_n": int(len(merged)),
                        "MAE_baseline": float(np.mean(err_base)),
                        "MAE_method": float(np.mean(err_method)),
                        "mean_MAE_reduction": float(np.mean(diff)),
                        "median_MAE_reduction": float(np.median(diff)),
                        "wilcoxon_stat": float(stat) if np.isfinite(stat) else np.nan,
                        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                    }
                )

    if not test_rows:
        return pd.DataFrame()

    return pd.DataFrame(test_rows).sort_values(
        ["k", "baseline", "mean_MAE_reduction"],
        ascending=[True, True, False],
    )


def create_evidence_ablation_table(summary_df: pd.DataFrame, dataset: str, k_values: Sequence[int]) -> pd.DataFrame:
    main_methods = [
        "L0_PPG_only",
        "E_pop_only",
        "E_state_only",
        "E_local_only",
        "E_pop_state",
        "E_pop_local",
        "E_state_local",
        "SA_PIF_full",
    ]
    evidence_desc = {
        "L0_PPG_only": "PPG neural estimate only",
        "E_pop_only": "PPG + population residual prior",
        "E_state_only": "PPG + state-conditioned residual prior",
        "E_local_only": "PPG + sparse local calibration",
        "E_pop_state": "PPG + population + state",
        "E_pop_local": "PPG + population + local",
        "E_state_local": "PPG + state + local",
        "SA_PIF_full": "PPG + population + state + local",
    }
    rows = []

    for k in k_values:
        l0_row = summary_df[(summary_df["method"] == "L0_PPG_only") & (summary_df["k"] == k)]
        l0_mae = float(l0_row["MAE"].iloc[0]) if len(l0_row) else np.nan

        for method in main_methods:
            sub = summary_df[(summary_df["method"] == method) & (summary_df["k"] == k)]
            if len(sub) == 0:
                continue

            row = sub.iloc[0]
            rows.append(
                {
                    "Dataset": dataset,
                    "k": int(k),
                    "Method": method,
                    "Evidence": evidence_desc[method],
                    "MAE": row["MAE"],
                    "RMSE": row["RMSE"],
                    "MARD": row["MARD"],
                    "Bias": row["Bias"],
                    "Pearson_r": row["Pearson_r"],
                    "Clarke_AB_percent": row["Clarke_AB_percent"],
                    "Delta_MAE_vs_L0": l0_mae - row["MAE"] if np.isfinite(l0_mae) else np.nan,
                    "Percent_Improvement_vs_L0": ((l0_mae - row["MAE"]) / l0_mae * 100)
                    if np.isfinite(l0_mae) and l0_mae > 0
                    else np.nan,
                }
            )

    return pd.DataFrame(rows)


def create_missing_evidence_table(summary_df: pd.DataFrame, dataset: str, k_values: Sequence[int]) -> pd.DataFrame:
    robust_methods = [
        "SA_PIF_full",
        "Missing_state",
        "Missing_local",
        "Missing_population",
        "Uncertain_state_x9var",
    ]
    robust_desc = {
        "SA_PIF_full": "All evidence sources available",
        "Missing_state": "State evidence removed",
        "Missing_local": "Local calibration evidence removed",
        "Missing_population": "Population prior removed",
        "Uncertain_state_x9var": "State evidence uncertainty inflated 9x",
    }
    rows = []

    for k in k_values:
        full_row = summary_df[(summary_df["method"] == "SA_PIF_full") & (summary_df["k"] == k)]
        full_mae = float(full_row["MAE"].iloc[0]) if len(full_row) else np.nan

        for method in robust_methods:
            sub = summary_df[(summary_df["method"] == method) & (summary_df["k"] == k)]
            if len(sub) == 0:
                continue

            row = sub.iloc[0]
            rows.append(
                {
                    "Dataset": dataset,
                    "k": int(k),
                    "Scenario": method,
                    "Description": robust_desc[method],
                    "MAE": row["MAE"],
                    "RMSE": row["RMSE"],
                    "MARD": row["MARD"],
                    "Bias": row["Bias"],
                    "Clarke_AB_percent": row["Clarke_AB_percent"],
                    "Delta_MAE_vs_Full": row["MAE"] - full_mae if np.isfinite(full_mae) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def summarize_weights(weight_df: pd.DataFrame) -> pd.DataFrame:
    if len(weight_df) == 0:
        return pd.DataFrame()
    weight_cols = [column for column in weight_df.columns if column.startswith("w_")]
    if not weight_cols:
        return pd.DataFrame()
    return weight_df.groupby("k")[weight_cols].mean().reset_index()


def plot_weight_behavior(weight_summary: pd.DataFrame, output_dir: str) -> None:
    if len(weight_summary) == 0:
        return

    weight_cols = [column for column in weight_summary.columns if column.startswith("w_")]
    if not weight_cols:
        return

    plt.figure(figsize=(7.5, 4.8))
    for column in weight_cols:
        plt.plot(
            weight_summary["k"],
            weight_summary[column],
            marker="o",
            linewidth=2,
            label=column.replace("w_", ""),
        )

    plt.xlabel("Calibration samples k")
    plt.ylabel("Mean fusion weight")
    plt.title("SA-PIF Reliability-Aware Fusion Weight Behavior")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_sa_pif_weight_behavior.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "fig_sa_pif_weight_behavior.pdf"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_convergence(summary_df: pd.DataFrame, dataset: str, output_dir: str) -> None:
    plot_methods = ["L0_PPG_only", "E_pop_only", "E_local_only", "SA_PIF_full"]
    plt.figure(figsize=(8.5, 5.2))

    for method in plot_methods:
        sub = summary_df[summary_df["method"] == method].sort_values("k")
        if len(sub) == 0:
            continue
        plt.plot(sub["k"], sub["MAE"], marker="o", linewidth=2, label=method)

    plt.xlabel("Calibration samples k")
    plt.ylabel("MAE (mg/dL)")
    plt.title(f"SA-PIF Sparse-Calibration Evidence Fusion - {dataset}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_sa_pif_sparse_calibration_convergence.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "fig_sa_pif_sparse_calibration_convergence.pdf"), dpi=300, bbox_inches="tight")
    plt.close()


def save_outputs(
    pred_df: pd.DataFrame,
    weight_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    tests_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
    weight_summary: pd.DataFrame,
    output_dir: str,
) -> Dict[str, str]:
    output_files = {
        "predictions": os.path.join(output_dir, "sa_pif_predictions_long.csv"),
        "weights": os.path.join(output_dir, "sa_pif_weight_records.csv"),
        "summary": os.path.join(output_dir, "sa_pif_summary_by_method_k.csv"),
        "pairwise_tests": os.path.join(output_dir, "sa_pif_pairwise_tests.csv"),
        "evidence_ablation": os.path.join(output_dir, "paper_ready_evidence_ablation_table.csv"),
        "missing_evidence": os.path.join(output_dir, "paper_ready_missing_evidence_robustness_table.csv"),
        "weight_behavior": os.path.join(output_dir, "paper_ready_fusion_weight_behavior.csv"),
    }

    pred_df.to_csv(output_files["predictions"], index=False)
    weight_df.to_csv(output_files["weights"], index=False)
    summary_df.to_csv(output_files["summary"], index=False)
    tests_df.to_csv(output_files["pairwise_tests"], index=False)
    evidence_df.to_csv(output_files["evidence_ablation"], index=False)
    robustness_df.to_csv(output_files["missing_evidence"], index=False)

    if len(weight_summary) > 0:
        weight_summary.to_csv(output_files["weight_behavior"], index=False)

    return output_files


def create_claim_candidates(summary_df: pd.DataFrame, k_values: Sequence[int]) -> pd.DataFrame:
    rows = []

    for k in k_values:
        l0 = summary_df[(summary_df["method"] == "L0_PPG_only") & (summary_df["k"] == k)]
        full = summary_df[(summary_df["method"] == "SA_PIF_full") & (summary_df["k"] == k)]

        if len(l0) == 0 or len(full) == 0:
            continue

        l0_row = l0.iloc[0]
        full_row = full.iloc[0]
        delta = l0_row["MAE"] - full_row["MAE"]
        percent = delta / l0_row["MAE"] * 100 if l0_row["MAE"] > 0 else np.nan

        rows.append(
            {
                "k": int(k),
                "SA_PIF_MAE": full_row["MAE"],
                "L0_MAE": l0_row["MAE"],
                "MAE_delta": delta,
                "MAE_percent_change": percent,
                "SA_PIF_RMSE": full_row["RMSE"],
                "SA_PIF_MARD": full_row["MARD"],
                "SA_PIF_Clarke_AB_percent": full_row["Clarke_AB_percent"],
            }
        )

    return pd.DataFrame(rows)


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    validate_paths(config)
    set_seed(config.seed)

    records = load_records(config)
    subject_data, state_names = group_by_subject(records)
    subjects = sorted(subject_data.keys())

    model = TFLiteGlucoseModel(config.tflite_path, config.scaler_path, config)
    base_cache = cache_base_predictions(subject_data, model, show_progress=config.show_progress)

    pred_df, weight_df = run_loso_experiment(
        subjects=subjects,
        subject_data=subject_data,
        state_names=state_names,
        base_cache=base_cache,
        config=config,
    )

    summary_df = create_summary(pred_df, config.dataset)
    tests_df = run_pairwise_tests(pred_df, config.dataset, config.k_values)
    evidence_df = create_evidence_ablation_table(summary_df, config.dataset, config.k_values)
    robustness_df = create_missing_evidence_table(summary_df, config.dataset, config.k_values)
    weight_summary = summarize_weights(weight_df)
    claim_df = create_claim_candidates(summary_df, config.k_values)

    output_files = save_outputs(
        pred_df=pred_df,
        weight_df=weight_df,
        summary_df=summary_df,
        tests_df=tests_df,
        evidence_df=evidence_df,
        robustness_df=robustness_df,
        weight_summary=weight_summary,
        output_dir=config.output_dir,
    )

    claim_path = os.path.join(config.output_dir, "paper_ready_claim_candidates.csv")
    claim_df.to_csv(claim_path, index=False)
    output_files["claim_candidates"] = claim_path

    plot_weight_behavior(weight_summary, config.output_dir)
    plot_convergence(summary_df, config.dataset, config.output_dir)

    return {
        "dataset": config.dataset,
        "subjects": len(subjects),
        "records": len(records),
        "states": state_names,
        "outputs": output_files,
        "summary": summary_df,
        "claims": claim_df,
    }


def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    dataset = args.dataset.upper()

    if dataset == "EXTERNAL20":
        subject_key = args.subject_key or "subject_id"
        glucose_keys = list(parse_list(args.glucose_keys or "bgl_mgdl,glucose", str))
        uid_keys = list(parse_list(args.uid_keys or "id,uid", str))
        condition_default = args.condition_default or "Unknown"
        adapt_ppg = args.adapt_ppg if args.adapt_ppg is not None else True
    elif dataset == "MUST":
        subject_key = args.subject_key or "caseid"
        glucose_keys = list(parse_list(args.glucose_keys or "glucose,bgl_mgdl", str))
        uid_keys = list(parse_list(args.uid_keys or "uid,id", str))
        condition_default = args.condition_default or "fasting"
        adapt_ppg = args.adapt_ppg if args.adapt_ppg is not None else False
    else:
        subject_key = args.subject_key
        glucose_keys = list(parse_list(args.glucose_keys, str))
        uid_keys = list(parse_list(args.uid_keys or "uid,id", str))
        condition_default = args.condition_default or "Unknown"
        adapt_ppg = args.adapt_ppg if args.adapt_ppg is not None else False

    if not subject_key:
        raise ValueError("--subject-key is required for custom datasets.")
    if not glucose_keys:
        raise ValueError("--glucose-keys is required.")

    return ExperimentConfig(
        dataset=dataset,
        data_path=args.data_path,
        tflite_path=args.tflite_path,
        scaler_path=args.scaler_path,
        output_dir=args.output_dir,
        subject_key=subject_key,
        glucose_keys=glucose_keys,
        uid_keys=uid_keys,
        condition_default=condition_default,
        adapt_ppg=adapt_ppg,
        seed=args.seed,
        fs=args.fs,
        target_len=args.target_len,
        n_features=args.n_features,
        k_values=parse_list(args.k_values, int),
        cal_selection_mode=args.cal_selection_mode,
        offset_clip=args.offset_clip,
        eps_var=args.eps_var,
        show_progress=not args.no_progress,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SA-PIF information fusion experiments.")
    parser.add_argument("--dataset", default="MUST", help="Dataset name: MUST, EXTERNAL20, or custom.")
    parser.add_argument("--data-path", required=True, help="Input JSON dataset path.")
    parser.add_argument("--tflite-path", required=True, help="TFLite model path.")
    parser.add_argument("--scaler-path", required=True, help="Feature scaler .npy path.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--subject-key", default=None, help="Subject identifier key in the JSON records.")
    parser.add_argument("--glucose-keys", default=None, help="Comma-separated glucose field names.")
    parser.add_argument("--uid-keys", default=None, help="Comma-separated UID field names.")
    parser.add_argument("--condition-default", default=None, help="Default condition/state label.")
    parser.add_argument("--adapt-ppg", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fs", type=float, default=100.0)
    parser.add_argument("--target-len", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=12)
    parser.add_argument("--k-values", default="0,1,2", help="Comma-separated calibration k values.")
    parser.add_argument("--cal-selection-mode", choices=["diverse", "random"], default="diverse")
    parser.add_argument("--offset-clip", type=float, default=120.0)
    parser.add_argument("--eps-var", type=float, default=1e-6)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config_from_args(args)
    result = run_experiment(config)

    print("SA-PIF experiment completed.")
    print(f"Dataset: {result['dataset']}")
    print(f"Subjects: {result['subjects']}")
    print(f"Records: {result['records']}")
    print(f"Output directory: {config.output_dir}")


if __name__ == "__main__":
    main()
