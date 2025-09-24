"""Utility helpers for preparing custom datasets for the HSPGNN models.

These helpers allow loading a pair of CSV files (time-series values and an
adjacency matrix) and converting them into the tensor dictionary structure
expected by :mod:`model.HSPGCN` without touching the original repository
logic.  The utilities intentionally mirror the layout created by
``lib.data_preparation.read_and_generate_dataset`` but remove the dataset
specific assumptions so that arbitrary graph signals can be consumed.

Typical usage::

    from custom_dataset_utils import prepare_custom_dataset

    dataset, stats = prepare_custom_dataset(
        timeseries_path="my_values.csv",
        adjacency_path="my_adj.csv",
        week_len=12,
        day_len=12,
        recent_len=36,
        target_len=6,
    )

The returned ``dataset`` dictionary contains ``train``/``val``/``test``
sub-dictionaries with NumPy arrays that can be wrapped into ``TensorDataset``
instances directly.  The validation split mirrors the testing split so that
workflows expecting a ``val`` key continue to operate when only a train/test
division is desired.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


ArrayDict = Dict[str, np.ndarray]
DatasetDict = Dict[str, ArrayDict]
StatsDict = Dict[str, Dict[str, np.ndarray]]


@dataclass(frozen=True)
class DatasetSplit:
    """Container that holds a dataset split.

    Attributes
    ----------
    week: np.ndarray
        Array with shape ``(samples, 1, num_nodes, week_len)``.
    week_mask: np.ndarray
        Mask aligned with ``week``.  Mask values are 1 where the signal is
        observed and 0 when the value is missing.
    day: np.ndarray
        Array with shape ``(samples, 1, num_nodes, day_len)``.
    day_mask: np.ndarray
        Mask aligned with ``day``.
    recent: np.ndarray
        Array with shape ``(samples, 1, num_nodes, recent_len)``.
    recent_mask: np.ndarray
        Mask aligned with ``recent``.
    target: np.ndarray
        Array with shape ``(samples, num_nodes, target_len)``.
    target_mask: np.ndarray
        Mask aligned with ``target``.
    """

    week: np.ndarray
    week_mask: np.ndarray
    day: np.ndarray
    day_mask: np.ndarray
    recent: np.ndarray
    recent_mask: np.ndarray
    target: np.ndarray
    target_mask: np.ndarray


def _load_csv(path: Path, *, delimiter: str = ",") -> np.ndarray:
    """Load a CSV file into a float32 NumPy array.

    Parameters
    ----------
    path:
        File system path pointing to the CSV file.
    delimiter:
        Column delimiter used in the file.

    Returns
    -------
    np.ndarray
        Two-dimensional array containing the parsed values.
    """

    data = np.loadtxt(str(path), delimiter=delimiter, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"CSV file {path} must be 2-D, got shape {data.shape}")
    return data.astype(np.float32)


def load_time_series(
    path: str | Path,
    *,
    delimiter: str = ",",
    transpose: bool = False,
    missing_value: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the time-series CSV file.

    The repository expects the data to be arranged as ``(num_nodes, T)``.
    ``transpose=True`` can be used when the CSV file stores the opposite
    orientation ``(T, num_nodes)``.

    Missing entries are detected via ``NaN`` values and optionally through a
    specific placeholder ``missing_value``.  The function returns both the
    data matrix with missing values replaced by the per-node temporal mean
    and a mask where ``1`` denotes that the corresponding value was observed.

    Parameters
    ----------
    path:
        Location of the CSV file with the raw signal values.
    delimiter:
        Delimiter used inside the CSV file.
    transpose:
        Set to ``True`` when nodes are stored across columns.
    missing_value:
        Optional scalar that should be treated as missing.  The raw entries
        equal to this value will be converted to ``NaN`` before computing
        the mask.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple ``(filled_data, observation_mask)`` where ``filled_data`` has
        shape ``(num_nodes, T)`` and ``observation_mask`` shares the same
        shape with ones on observed entries and zeros on missing ones.
    """

    array = _load_csv(Path(path), delimiter=delimiter)
    if transpose:
        array = array.T

    if array.ndim != 2:
        raise ValueError(
            "Time-series array must be 2-D after optional transpose, "
            f"got shape {array.shape}"
        )

    if missing_value is not None:
        array = array.copy()
        array[array == missing_value] = np.nan

    nan_mask = np.isnan(array)
    observation_mask = (~nan_mask).astype(np.float32)

    if np.all(nan_mask):
        raise ValueError("The provided time-series file only contains missing values.")

    filled = _fill_missing_with_mean(array, nan_mask)
    return filled, observation_mask


def _fill_missing_with_mean(data: np.ndarray, nan_mask: np.ndarray) -> np.ndarray:
    """Replace NaN values with the per-node temporal mean."""

    filled = data.copy()
    # Compute the mean along the temporal axis for every node.
    node_means = np.nanmean(filled, axis=1, keepdims=True)
    # Guard against nodes that are entirely NaN by falling back to zeros.
    node_means = np.nan_to_num(node_means, nan=0.0)
    # ``nan_mask`` is a 2-D boolean array; broadcasting ``node_means`` ensures
    # each missing entry receives the corresponding node's temporal mean
    # without relying on flattened indexing that breaks on NumPy >= 1.24.
    filled = np.where(nan_mask, node_means, filled)
    return filled.astype(np.float32)


def load_adjacency_matrix(path: str | Path, *, delimiter: str = ",") -> np.ndarray:
    """Load the adjacency matrix stored inside a CSV file."""

    adj = _load_csv(Path(path), delimiter=delimiter)
    if adj.shape[0] != adj.shape[1]:
        raise ValueError(
            "Adjacency matrix must be square; received shape " f"{adj.shape}"
        )
    return adj.astype(np.float32)


def _generate_samples(
    data: np.ndarray,
    mask: np.ndarray,
    *,
    week_len: int,
    day_len: int,
    recent_len: int,
    target_len: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Convert the full sequence into sliding window samples."""

    num_nodes, total_steps = data.shape
    history = week_len + day_len + recent_len
    if history <= 0:
        raise ValueError("The total history length must be positive.")
    if total_steps <= history + target_len:
        raise ValueError(
            "Not enough temporal points to construct at least one sample. "
            f"Need > {history + target_len} steps, received {total_steps}."
        )

    samples: List[
        Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ] = []

    for end_idx in range(history, total_steps - target_len + 1):
        history_start = end_idx - history
        week_slice = slice(history_start, history_start + week_len)
        day_slice = slice(history_start + week_len, history_start + week_len + day_len)
        recent_slice = slice(end_idx - recent_len, end_idx)
        target_slice = slice(end_idx, end_idx + target_len)

        week = data[:, week_slice]
        week_mask = mask[:, week_slice]
        day = data[:, day_slice]
        day_mask = mask[:, day_slice]
        recent = data[:, recent_slice]
        recent_mask = mask[:, recent_slice]
        target = data[:, target_slice]
        target_mask = mask[:, target_slice]
        samples.append(
            (week, week_mask, day, day_mask, recent, recent_mask, target, target_mask)
        )

    return samples


def _split_train_test(
    samples: Sequence[Tuple[np.ndarray, ...]],
    *,
    train_ratio: float,
    test_ratio: float,
) -> Tuple[Sequence[Tuple[np.ndarray, ...]], Sequence[Tuple[np.ndarray, ...]]]:
    """Split samples chronologically into training and testing segments."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 < test_ratio <= 1:
        raise ValueError("test_ratio must be between 0 and 1 inclusive.")

    # When the provided ratios do not sum to 1, rescale them so that the full
    # dataset is consumed by the two splits while keeping their relative
    # proportions intact.
    ratio_sum = train_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("train_ratio + test_ratio must be positive.")
    if ratio_sum != 1.0:
        train_ratio = train_ratio / ratio_sum
        test_ratio = test_ratio / ratio_sum

    total = len(samples)
    if total < 2:
        raise ValueError(
            "At least two samples are required to create train/test splits."
        )

    # Convert the ratios into counts while ensuring that both splits contain at
    # least one sample and that rounding does not drop any data.
    train_count = int(round(total * train_ratio))
    train_count = max(min(train_count, total - 1), 1)
    test_count = total - train_count
    if test_count <= 0:
        # Extremely small datasets may round the testing portion down to zero.
        # In that case, steal one element from the training set.
        if train_count <= 1:
            raise ValueError(
                "Unable to create a testing split; provide more temporal samples."
            )
        train_count -= 1
        test_count = 1

    train_split = samples[:train_count]
    test_split = samples[train_count:]
    return train_split, test_split


def _stack_component(
    component: Sequence[np.ndarray],
    *,
    add_channel_dim: bool,
) -> np.ndarray:
    """Stack a component extracted from every sample."""

    stacked = np.stack(component, axis=0).astype(np.float32)
    if add_channel_dim:
        stacked = np.expand_dims(stacked, axis=1)
    return stacked


def _to_split_dict(samples: Sequence[Tuple[np.ndarray, ...]]) -> DatasetSplit:
    """Convert raw sample tuples into the structured :class:`DatasetSplit`."""

    week, week_mask, day, day_mask, recent, recent_mask, target, target_mask = zip(
        *samples
    )

    week_arr = _stack_component(week, add_channel_dim=True)
    day_arr = _stack_component(day, add_channel_dim=True)
    recent_arr = _stack_component(recent, add_channel_dim=True)

    # Masks use 1 for observed values so they can be multiplied directly.
    week_mask_arr = _stack_component(week_mask, add_channel_dim=True)
    day_mask_arr = _stack_component(day_mask, add_channel_dim=True)
    recent_mask_arr = _stack_component(recent_mask, add_channel_dim=True)

    target_arr = _stack_component(target, add_channel_dim=False)
    target_mask_arr = _stack_component(target_mask, add_channel_dim=False)

    return DatasetSplit(
        week=week_arr,
        week_mask=week_mask_arr,
        day=day_arr,
        day_mask=day_mask_arr,
        recent=recent_arr,
        recent_mask=recent_mask_arr,
        target=target_arr,
        target_mask=target_mask_arr,
    )


def _compute_normalization(
    train: np.ndarray, val: np.ndarray, test: np.ndarray
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Normalize datasets using statistics estimated from the training split."""

    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    def normalize(x: np.ndarray) -> np.ndarray:
        return (x - mean) / std

    return {"mean": mean, "std": std}, normalize(train), normalize(val), normalize(test)


def _apply_normalization(split: DatasetSplit) -> DatasetSplit:
    """Return a copy of the split with arrays converted to ``float32``."""

    return DatasetSplit(
        week=split.week.astype(np.float32),
        week_mask=split.week_mask.astype(np.float32),
        day=split.day.astype(np.float32),
        day_mask=split.day_mask.astype(np.float32),
        recent=split.recent.astype(np.float32),
        recent_mask=split.recent_mask.astype(np.float32),
        target=split.target.astype(np.float32),
        target_mask=split.target_mask.astype(np.float32),
    )


def prepare_custom_dataset(
    *,
    timeseries_path: str | Path,
    adjacency_path: str | Path,
    delimiter: str = ",",
    transpose_timeseries: bool = False,
    missing_value: float | None = None,
    week_len: int = 12,
    day_len: int = 12,
    recent_len: int = 36,
    target_len: int = 6,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    task: str = "forecast",
    impute_rate: float = 0.1,
    impute_seed: int | None = None,
) -> Tuple[DatasetDict, StatsDict, np.ndarray]:
    """Prepare a dataset compatible with the training scripts.

    Parameters
    ----------
    timeseries_path:
        CSV file containing the graph signals.
    adjacency_path:
        CSV file containing the square adjacency matrix.
    delimiter:
        Optional delimiter used in both CSV files.
    transpose_timeseries:
        Set to ``True`` when the CSV file stores timesteps as rows.
    missing_value:
        Optional sentinel value that represents a missing observation.
    week_len, day_len, recent_len, target_len:
        Window sizes for the temporal context and prediction horizon.  The
        defaults match the configuration used by the original repository.  When
        ``task="impute"`` the ``target_len`` argument is ignored and the
        complete temporal context (``week_len + day_len + recent_len``) becomes
        the reconstruction target.
    train_ratio, val_ratio:
        Fractions used to split the samples chronologically.  ``train_ratio``
        controls the fraction assigned to the training split while
        ``val_ratio`` denotes the desired fraction for testing.  The
        validation split mirrors the testing portion so downstream code can
        continue to access both keys.
    task:
        Specifies the learning objective.  ``"forecast"`` (default) trains the
        original six-step forecasting head.  ``"impute"`` corrupts the inputs
        with additional random masks and teaches the network to reconstruct the
        masked values within the temporal context window.
    impute_rate:
        When ``task="impute"``, the fraction of observed entries that are
        dropped at random and used as supervised targets.  The mask is applied
        independently to the train/validation/testing splits with distinct
        random generators for reproducibility.
    impute_seed:
        Optional random seed used when generating the artificial missing masks
        for the imputation task.  ``None`` (default) relies on NumPy's global
        random state.

    Returns
    -------
    Tuple[DatasetDict, StatsDict, np.ndarray]
        ``dataset`` containing the ``train``/``val``/``test`` splits (with the
        validation split mirroring the testing data), ``stats`` with the
        normalization information and the adjacency matrix as a NumPy array.
    """

    data, observation_mask = load_time_series(
        timeseries_path,
        delimiter=delimiter,
        transpose=transpose_timeseries,
        missing_value=missing_value,
    )
    adjacency = load_adjacency_matrix(adjacency_path, delimiter=delimiter)

    if data.shape[0] != adjacency.shape[0]:
        raise ValueError(
            "Number of nodes in the time-series (%d) does not match the adjacency matrix (%d)."
            % (data.shape[0], adjacency.shape[0])
        )

    if task not in {"forecast", "impute"}:
        raise ValueError("task must be either 'forecast' or 'impute'.")

    if task == "impute" and not 0 < impute_rate <= 1.0:
        raise ValueError("impute_rate must be in the interval (0, 1].")

    samples = _generate_samples(
        data,
        observation_mask,
        week_len=week_len,
        day_len=day_len,
        recent_len=recent_len,
        target_len=target_len,
    )
    train_samples, test_samples = _split_train_test(
        samples, train_ratio=train_ratio, test_ratio=val_ratio
    )

    train_split = _to_split_dict(train_samples)
    # The validation split is intentionally identical to the testing split to
    # accommodate training loops that expect both keys while operating with a
    # classic train/test partition.
    val_split = _to_split_dict(test_samples)
    test_split = _to_split_dict(test_samples)

    # Normalise each component separately using training statistics.
    week_stats, train_week, val_week, test_week = _compute_normalization(
        train_split.week, val_split.week, test_split.week
    )
    day_stats, train_day, val_day, test_day = _compute_normalization(
        train_split.day, val_split.day, test_split.day
    )
    recent_stats, train_recent, val_recent, test_recent = _compute_normalization(
        train_split.recent, val_split.recent, test_split.recent
    )

    def _build_forecast_split(
        template: DatasetSplit,
        *,
        week: np.ndarray,
        day: np.ndarray,
        recent: np.ndarray,
    ) -> ArrayDict:
        """Convert a ``DatasetSplit`` plus normalized components into an array dict."""

        return {
            "week": week.astype(np.float32, copy=False),
            "week_mask": template.week_mask.astype(np.float32, copy=False),
            "day": day.astype(np.float32, copy=False),
            "day_mask": template.day_mask.astype(np.float32, copy=False),
            "recent": recent.astype(np.float32, copy=False),
            "recent_mask": template.recent_mask.astype(np.float32, copy=False),
            "target": template.target.astype(np.float32, copy=False),
            "target_mask": template.target_mask.astype(np.float32, copy=False),
        }

    def _concatenate_context(split: DatasetSplit) -> Tuple[np.ndarray, np.ndarray]:
        """Return the stacked temporal context and corresponding mask."""

        context = np.concatenate(
            [
                split.week.squeeze(1),
                split.day.squeeze(1),
                split.recent.squeeze(1),
            ],
            axis=-1,
        )
        context_mask = np.concatenate(
            [
                split.week_mask.squeeze(1),
                split.day_mask.squeeze(1),
                split.recent_mask.squeeze(1),
            ],
            axis=-1,
        )
        return context.astype(np.float32), context_mask.astype(np.float32)

    def _build_imputation_split(
        template: DatasetSplit,
        *,
        week: np.ndarray,
        day: np.ndarray,
        recent: np.ndarray,
        rng: np.random.Generator,
        rate: float,
    ) -> ArrayDict:
        """Create an imputation split by masking additional entries."""

        week_len_local = week.shape[-1]
        day_len_local = day.shape[-1]
        recent_len_local = recent.shape[-1]
        temporal_window = week_len_local + day_len_local + recent_len_local

        context, context_mask = _concatenate_context(template)
        drop_candidates = context_mask > 0.5
        random_matrix = rng.random(context_mask.shape, dtype=np.float64)
        drop_mask = (random_matrix < rate) & drop_candidates

        flat_drop = drop_mask.reshape(drop_mask.shape[0], -1)
        flat_candidates = drop_candidates.reshape(drop_candidates.shape[0], -1)
        for row_idx in range(flat_drop.shape[0]):
            if not flat_drop[row_idx].any():
                candidate_indices = np.flatnonzero(flat_candidates[row_idx])
                if candidate_indices.size:
                    chosen = rng.choice(candidate_indices)
                    flat_drop[row_idx, chosen] = True
        drop_mask = flat_drop.reshape(drop_mask.shape)

        week_mask = template.week_mask.astype(np.float32, copy=True)
        day_mask = template.day_mask.astype(np.float32, copy=True)
        recent_mask = template.recent_mask.astype(np.float32, copy=True)

        week_mask[week_mask < 0.5] = 0.0
        day_mask[day_mask < 0.5] = 0.0
        recent_mask[recent_mask < 0.5] = 0.0

        # Ensure artificially dropped entries are hidden from the model inputs.
        week_slice = np.expand_dims(drop_mask[:, :, :week_len_local], axis=1)
        day_slice = np.expand_dims(
            drop_mask[:, :, week_len_local : week_len_local + day_len_local], axis=1
        )
        recent_slice = np.expand_dims(drop_mask[:, :, week_len_local + day_len_local :], axis=1)

        week_mask[week_slice] = 0.0
        day_mask[day_slice] = 0.0
        recent_mask[recent_slice] = 0.0

        # Replace hidden entries with zeros (equivalent to the normalized mean).
        week = week.astype(np.float32, copy=True)
        day = day.astype(np.float32, copy=True)
        recent = recent.astype(np.float32, copy=True)

        week[week_slice] = 0.0
        day[day_slice] = 0.0
        recent[recent_slice] = 0.0

        # Also blank out positions that were already missing in the raw data.
        week[week_mask < 0.5] = 0.0
        day[day_mask < 0.5] = 0.0
        recent[recent_mask < 0.5] = 0.0

        target = context.astype(np.float32, copy=True)
        target_mask = drop_mask.astype(np.float32)

        return {
            "week": week,
            "week_mask": week_mask,
            "day": day,
            "day_mask": day_mask,
            "recent": recent,
            "recent_mask": recent_mask,
            "target": target,
            "target_mask": target_mask,
        }

    if task == "forecast":
        dataset: DatasetDict = {
            "train": _build_forecast_split(
                train_split, week=train_week, day=train_day, recent=train_recent
            ),
            "val": _build_forecast_split(val_split, week=val_week, day=val_day, recent=val_recent),
            "test": _build_forecast_split(
                test_split, week=test_week, day=test_day, recent=test_recent
            ),
        }
    else:
        if impute_seed is None:
            train_rng = np.random.default_rng()
            val_rng = np.random.default_rng()
            test_rng = np.random.default_rng()
        else:
            train_rng = np.random.default_rng(impute_seed)
            val_rng = np.random.default_rng(impute_seed + 1)
            test_rng = np.random.default_rng(impute_seed + 2)

        dataset = {
            "train": _build_imputation_split(
                train_split, week=train_week, day=train_day, recent=train_recent, rng=train_rng, rate=impute_rate
            ),
            "val": _build_imputation_split(
                val_split, week=val_week, day=val_day, recent=val_recent, rng=val_rng, rate=impute_rate
            ),
            "test": _build_imputation_split(
                test_split,
                week=test_week,
                day=test_day,
                recent=test_recent,
                rng=test_rng,
                rate=impute_rate,
            ),
        }

    stats: StatsDict = {
        component: {
            name: values.astype(np.float32) for name, values in component_stats.items()
        }
        for component, component_stats in {
            "week": week_stats,
            "day": day_stats,
            "recent": recent_stats,
        }.items()
    }

    return dataset, stats, adjacency.astype(np.float32)


__all__ = [
    "DatasetSplit",
    "prepare_custom_dataset",
    "load_adjacency_matrix",
    "load_time_series",
]
