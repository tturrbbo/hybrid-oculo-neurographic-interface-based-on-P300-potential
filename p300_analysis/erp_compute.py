"""Усреднение эпох и подготовка данных для отображения победителя."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from p300_analysis.constants import SAFE_MIN_EPOCHS_TO_DECIDE
from p300_analysis.marker_parsing import stim_key_sort_key, stim_key_to_tile_digit
from p300_analysis.signal_processing import (
    baseline_correction,
    integrated_cumsum,
    normalize_channels,
    time_window_to_indices,
)
from p300_analysis.winner_selection import (
    WINNER_MODE_AUC,
    WINNER_MODE_SIGNED_MEAN,
    WINNER_MODE_TEMPLATE_CORR,
)


def artifact_reject_epochs(
    epochs: List[np.ndarray],
    threshold_uv: float,
) -> Tuple[List[np.ndarray], int]:
    """Отбрасывает эпохи, в которых амплитуда превышает порог.

    epochs: список массивов (epoch_len,) или (epoch_len, n_ch).
    np.max(np.abs) работает для обоих вариантов.
    """
    if threshold_uv <= 0:
        return epochs, 0
    clean: List[np.ndarray] = []
    rejected = 0
    for ep in epochs:
        if np.max(np.abs(ep)) <= threshold_uv:
            clean.append(ep)
        else:
            rejected += 1
    return clean, rejected


def build_averaged_erp(
    epochs_data: Dict[str, List[np.ndarray]],
    epoch_len: int,
    artifact_threshold_uv: Optional[float] = None,
) -> Tuple[List[str], np.ndarray, Dict[str, int]]:
    """Усредняет эпохи по каждому стимулу.

    Поддерживает два формата эпох:
    - 1D (epoch_len,)           — legacy / single-channel
    - 2D (epoch_len, n_ch)      — per-channel (новый формат)

    Для 2D каждая эпоха нормализуется по каналу (÷ std), затем каналы
    усредняются → выравнивает влияние шумных каналов.

    Возвращает (stim_keys, raw_averaged, rejected_counts),
    где raw_averaged.shape = (n_stim, epoch_len) — совместимо с downstream.
    """
    stim_keys = [k for k, v in epochs_data.items() if v]
    stim_keys.sort(key=stim_key_sort_key)
    n_stim = len(stim_keys)
    raw_averaged = np.zeros((n_stim, epoch_len), dtype=np.float64)
    rejected_counts: Dict[str, int] = {}

    for i, key in enumerate(stim_keys):
        epochs = epochs_data.get(key, [])
        if not epochs:
            continue
        if artifact_threshold_uv is not None and artifact_threshold_uv > 0:
            epochs, n_rej = artifact_reject_epochs(epochs, artifact_threshold_uv)
            rejected_counts[key] = n_rej
        if not epochs:
            rejected_counts[key] = rejected_counts.get(key, 0)
            continue

        if epochs[0].ndim == 2:
            # Per-channel path: normalize each epoch by channel std, then average
            normed = [normalize_channels(ep[:epoch_len]) for ep in epochs]
            stack = np.stack(normed, axis=0)          # (n_ep, epoch_len, n_ch)
            mean_ch_erp = np.mean(stack, axis=0)      # (epoch_len, n_ch)
            raw_averaged[i, :] = np.mean(mean_ch_erp, axis=-1)  # (epoch_len,)
        else:
            stack = np.stack([e[:epoch_len] for e in epochs], axis=0)
            raw_averaged[i, :] = np.mean(stack, axis=0)

    return stim_keys, raw_averaged, rejected_counts


def compute_corrected_and_integrated(
    raw_averaged: np.ndarray,
    time_ms: np.ndarray,
    baseline_ms: int,
    window_x_ms: int,
    window_y_ms: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    wy = window_y_ms if window_y_ms > window_x_ms else window_x_ms + 1
    corrected = baseline_correction(raw_averaged, time_ms, baseline_ms=baseline_ms)
    integrated, time_crop = integrated_cumsum(
        corrected,
        time_ms,
        window_x_ms=window_x_ms,
        window_y_ms=wy,
    )
    return corrected, integrated, time_crop, window_x_ms, wy


def stim_epoch_count_stats(
    stim_keys: List[str],
    epochs_data: Dict[str, List[np.ndarray]],
) -> Dict[str, Any]:
    """Per-class epoch counts at decision time (all accumulated epochs are used downstream)."""
    counts = {k: len(epochs_data.get(k, [])) for k in stim_keys}
    vals = list(counts.values())
    if not vals:
        return {
            "min_per_class": 0,
            "max_per_class": 0,
            "total_epochs": 0,
            "epochs_used_for_decision": {},
            "epoch_count_at_decision": {},
        }
    return {
        "min_per_class": int(min(vals)),
        "max_per_class": int(max(vals)),
        "total_epochs": int(sum(vals)),
        "epochs_used_for_decision": dict(counts),
        "epoch_count_at_decision": dict(counts),
    }


def check_can_decide(stim_keys: List[str], epochs_data: Dict[str, List[np.ndarray]]) -> Tuple[bool, int]:
    """Allow decision when every active class has at least SAFE_MIN epochs (use all stored epochs)."""
    if not stim_keys:
        return False, 0
    stats = stim_epoch_count_stats(stim_keys, epochs_data)
    min_n = int(stats["min_per_class"])
    can = min_n >= int(SAFE_MIN_EPOCHS_TO_DECIDE)
    return can, min_n


def compute_winner_metrics(
    stim_keys: List[str],
    raw_averaged: np.ndarray,
    corrected: np.ndarray,
    time_ms: np.ndarray,
    window_x_ms: int,
    window_y_ms: int,
    winner_mode: str = WINNER_MODE_AUC,
    *,
    template_window: Optional[np.ndarray] = None,
) -> Tuple[int, str, Dict[str, Any]]:
    """Победитель по Main ERP.

    Новая логика специально связана с графиком Main ERP: берётся тот же массив
    raw_averaged, который затем рисуется в plot_raw / Main ERP. В выбранном
    временном окне [window_x_ms, window_y_ms] для каждого стимула находится
    самый маленький минимум ERP. Чем минимум ниже, тем сильнее ответ.

    score = -min(Main ERP window)
    winner = argmax(score)

    Вероятность считается по softmax от score, чтобы её можно было показать в Qt.
    """
    n_stim = int(raw_averaged.shape[0]) if raw_averaged.ndim == 2 else 0
    if n_stim <= 0:
        return 0, "main_erp_min", {
            "winner_rule": "main_erp_min",
            "reason": "empty_raw_averaged",
            "margin": 0.0,
            "main_erp_probability": 0.0,
        }

    xi0, xi1 = time_window_to_indices(time_ms, window_x_ms, window_y_ms)
    main_win = np.asarray(raw_averaged[:, xi0:xi1], dtype=np.float64)
    if main_win.size == 0 or main_win.shape[1] == 0:
        main_win = np.asarray(raw_averaged, dtype=np.float64)
        xi0, xi1 = 0, int(main_win.shape[1]) if main_win.ndim == 2 else 0

    # Минимум Main ERP: пользователь просил "наибольшая амплитуда = минимум самый маленький".
    erp_min_values = np.min(main_win, axis=1) if main_win.size else np.zeros(n_stim)
    erp_min_abs_scores = -erp_min_values

    # Для отладки также сохраняем максимум и peak-to-peak, но выбор идёт именно по минимуму.
    erp_max_values = np.max(main_win, axis=1) if main_win.size else np.zeros(n_stim)
    erp_ptp_values = erp_max_values - erp_min_values

    winner_idx = int(np.argmax(erp_min_abs_scores))

    # Softmax-вероятности. Температура по std делает шкалу устойчивее на разных амплитудах.
    scores = np.asarray(erp_min_abs_scores, dtype=np.float64)
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size and float(np.std(finite_scores)) > 1e-12:
        temperature = float(np.std(finite_scores))
    else:
        temperature = 1.0
    z = (scores - float(np.max(scores))) / max(temperature, 1e-9)
    exp_z = np.exp(np.clip(z, -50, 50))
    probs = exp_z / max(float(np.sum(exp_z)), 1e-12)
    winner_prob = float(probs[winner_idx]) if probs.size else 0.0

    order = np.argsort(scores)[::-1]
    top1 = float(scores[order[0]]) if order.size else 0.0
    top2 = float(scores[order[1]]) if order.size > 1 else 0.0
    if abs(top1) > 1e-12:
        margin = max(0.0, min(1.0, (top1 - top2) / (abs(top1) + 1e-12)))
    else:
        margin = 0.0

    # Legacy поля оставляем, чтобы старые экспорты/логи не ломались.
    corr_win = corrected[:, xi0:xi1] if corrected.ndim == 2 and corrected.size else np.zeros((n_stim, 0))
    abs_auc_values = np.sum(np.abs(corr_win), axis=1) if corr_win.size else np.zeros(n_stim)
    signed_mean_values = np.mean(corr_win, axis=1) if corr_win.size else np.zeros(n_stim)
    positive_peak_values = np.max(corr_win, axis=1) if corr_win.size else np.zeros(n_stim)

    dbg: Dict[str, Any] = {
        "winner_rule": "main_erp_min",
        "chosen_winner_idx": int(winner_idx),
        "chosen_winner_key": stim_keys[winner_idx] if 0 <= winner_idx < len(stim_keys) else None,
        "stim_keys": list(stim_keys),
        "window_index": [int(xi0), int(xi1)],
        "window_ms": [int(window_x_ms), int(window_y_ms)],
        "final_metric_values": [float(x) for x in scores],
        "main_erp_min_values": [float(x) for x in erp_min_values],
        "main_erp_min_abs_scores": [float(x) for x in erp_min_abs_scores],
        "main_erp_max_values": [float(x) for x in erp_max_values],
        "main_erp_ptp_values": [float(x) for x in erp_ptp_values],
        "main_erp_probabilities": [float(x) for x in probs],
        "main_erp_probability": float(winner_prob),
        "winner_min_value": float(erp_min_values[winner_idx]),
        "winner_score": float(scores[winner_idx]),
        "second_score": float(top2),
        "margin": float(winner_prob),
        "main_erp_margin_ratio": float(margin),
        "signed_mean_final": [float(x) for x in signed_mean_values],
        "abs_auc_values": [float(x) for x in abs_auc_values],
        "positive_peak_values": [float(x) for x in positive_peak_values],
    }
    return int(winner_idx), "main_erp_min", dbg


def winner_display_lines(
    winner_key: str,
    mode_short: str,
    lsl_cue_target_id: Optional[int],
    margin: Optional[float] = None,
) -> Tuple[List[str], int, bool]:
    win_digit = stim_key_to_tile_digit(winner_key)
    lines = ["РЕЗУЛЬТАТ:", f"ПЛИТКА {win_digit}", f"режим: {mode_short}"]
    if margin is not None:
        pct = int(round(margin * 100))
        confidence = "высокая" if pct >= 30 else ("средняя" if pct >= 12 else "низкая ⚠")
        lines.append(f"уверенность: {pct}% ({confidence})")
    if lsl_cue_target_id is not None:
        lines.append(f"цель LSL: {lsl_cue_target_id}")
    match_lsl = lsl_cue_target_id is None or win_digit == lsl_cue_target_id
    return lines, win_digit, match_lsl
