"""Train an HSPGNN model on custom CSV data for missing-value imputation."""

from __future__ import annotations

import argparse
from argparse import Namespace, SUPPRESS
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from custom_dataset_utils import prepare_custom_dataset
from model import HSPGCNImputer


MODEL_FACTORY = {
    "HSPGCNImputer": HSPGCNImputer,
}


def _strip_json_comments(text: str) -> str:
    """Remove JavaScript-style comments from JSON-like text."""

    result_chars = []
    in_string = False
    escape = False
    in_single_line_comment = False
    in_multi_line_comment = False
    i = 0
    length = len(text)

    while i < length:
        ch = text[i]
        next_ch = text[i + 1] if i + 1 < length else ""

        if in_single_line_comment:
            if ch in "\n\r":
                in_single_line_comment = False
                result_chars.append(ch)
            i += 1
            continue

        if in_multi_line_comment:
            if ch == "*" and next_ch == "/":
                in_multi_line_comment = False
                i += 2
            else:
                i += 1
            continue

        if in_string:
            result_chars.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            result_chars.append(ch)
            i += 1
            continue

        if ch == "/" and next_ch == "/":
            in_single_line_comment = True
            i += 2
            continue

        if ch == "/" and next_ch == "*":
            in_multi_line_comment = True
            i += 2
            continue

        if ch == "#":
            in_single_line_comment = True
            i += 1
            continue

        result_chars.append(ch)
        i += 1

    return "".join(result_chars)


def _load_config_file(path: Path) -> Dict[str, Any]:
    """Load a JSON configuration file and return its contents."""

    try:
        raw_text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Configuration file {path} does not exist.") from exc

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        sanitized = _strip_json_comments(raw_text)
        if sanitized != raw_text:
            try:
                data = json.loads(sanitized)
            except json.JSONDecodeError as exc2:
                raise ValueError(
                    (
                        "Failed to parse JSON configuration file "
                        f"{path}: {exc2.msg} (line {exc2.lineno}, column {exc2.colno})"
                    )
                ) from exc2
        else:
            raise ValueError(
                (
                    "Failed to parse JSON configuration file "
                    f"{path}: {exc.msg} (line {exc.lineno}, column {exc.colno})"
                )
            ) from exc
    else:
        sanitized = None

    if not isinstance(data, dict):
        raise ValueError(
            f"Configuration file {path} must contain a JSON object at the top level."
        )

    return data


def build_dataloaders(
    dataset: Dict[str, Dict[str, np.ndarray]],
    *,
    batch_size: int,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Construct ``DataLoader`` objects for the train/val/test splits."""

    def to_tensor_dataset(split: Dict[str, np.ndarray]) -> TensorDataset:
        tensors = [
            torch.from_numpy(split["week"]),
            torch.from_numpy(split["week_mask"]),
            torch.from_numpy(split["day"]),
            torch.from_numpy(split["day_mask"]),
            torch.from_numpy(split["recent"]),
            torch.from_numpy(split["recent_mask"]),
            torch.from_numpy(split["target"]),
            torch.from_numpy(split["target_mask"]),
        ]
        return TensorDataset(*tensors)

    train_loader = DataLoader(
        to_tensor_dataset(dataset["train"]),
        batch_size=batch_size,
        shuffle=shuffle_train,
    )
    val_loader = DataLoader(
        to_tensor_dataset(dataset["val"]),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        to_tensor_dataset(dataset["test"]),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader


def evaluate_split(
    net: nn.Module,
    loader: DataLoader,
    supports: torch.Tensor,
    device: torch.device,
    loss_fn: nn.Module,
) -> Tuple[float, float, float]:
    """Evaluate loss, MAE and RMSE on a data split."""

    net.eval()
    total_loss = 0.0
    count = 0
    mae_numerator = 0.0
    mse_numerator = 0.0
    mask_denominator = 0.0

    with torch.no_grad():
        for batch in loader:
            (week, week_mask, day, day_mask, recent, recent_mask, target, mask) = [
                tensor.to(device) for tensor in batch
            ]

            output, _, _, _ = net(week, week_mask, day, day_mask, recent, recent_mask, mask, supports)
            loss = loss_fn(output * mask, target * mask)
            total_loss += float(loss.item())
            count += 1

            abs_err = torch.abs(output - target) * mask
            sq_err = torch.square(output - target) * mask
            mae_numerator += abs_err.sum().item()
            mse_numerator += sq_err.sum().item()
            mask_denominator += mask.sum().item()

    avg_loss = total_loss / max(count, 1)
    if mask_denominator > 0:
        mae = mae_numerator / mask_denominator
        rmse = float(np.sqrt(mse_numerator / mask_denominator))
    else:
        mae = float("nan")
        rmse = float("nan")
    return avg_loss, mae, rmse


def train(args: argparse.Namespace) -> None:
    dataset, stats, adjacency = prepare_custom_dataset(
        timeseries_path=args.timeseries,
        adjacency_path=args.adjacency,
        delimiter=args.delimiter,
        transpose_timeseries=args.transpose_timeseries,
        missing_value=args.missing_value,
        week_len=args.week_len,
        day_len=args.day_len,
        recent_len=args.recent_len,
        target_len=args.target_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        impute_rate=args.impute_rate,
        impute_seed=args.impute_seed,
    )

    temporal_window = args.week_len + args.day_len + args.recent_len
    target_horizon = dataset["train"]["target"].shape[-1]
    if target_horizon != temporal_window:
        raise ValueError(
            "Imputation datasets must expose the full temporal window as the target. "
            f"Received {target_horizon} steps instead of {temporal_window}."
        )

    num_nodes = adjacency.shape[0]
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print(
            "CUDA was requested but is not available; falling back to the CPU instead.",
            flush=True,
        )
        requested_device = torch.device("cpu")

    device = requested_device

    train_loader, val_loader, test_loader = build_dataloaders(
        dataset, batch_size=args.batch_size, shuffle_train=not args.no_shuffle
    )

    model_cls = MODEL_FACTORY[args.model]
    net = model_cls(
        c_in=dataset["train"]["week"].shape[1],
        c_out=args.hidden_dim,
        num_nodes=num_nodes,
        week=args.week_len,
        day=args.day_len,
        recent=args.recent_len,
        K=args.K,
        Kt=args.Kt,
    )
    net.to(device)

    supports = torch.from_numpy(adjacency).type(torch.float32).to(device)

    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = None
    if args.lr_decay < 1.0:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    loss_fn = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    epoch = 0
    stop_reason: Optional[str] = None

    while True:
        epoch += 1
        net.train()
        epoch_losses = []
        for batch in train_loader:
            (week, week_mask, day, day_mask, recent, recent_mask, target, mask) = [
                tensor.to(device) for tensor in batch
            ]
            optimizer.zero_grad()
            output, _, _, _ = net(week, week_mask, day, day_mask, recent, recent_mask, mask, supports)
            loss = loss_fn(output * mask, target * mask)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_loss, val_mae, val_rmse = evaluate_split(net, val_loader, supports, device, loss_fn)
        print(
            f"Epoch {epoch:03d} | train loss {train_loss:.6f} | val loss {val_loss:.6f} "
            f"| val MAE {val_mae:.6f} | val RMSE {val_rmse:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        patience = args.patience
        if patience is not None and epochs_without_improvement >= patience:
            stop_reason = f"patience ({patience} epochs without improvement)"
            break

        if args.max_epoch is not None and epoch >= args.max_epoch:
            stop_reason = "max_epoch"
            break

    if stop_reason is None and args.patience is None and args.max_epoch is not None:
        stop_reason = "max_epoch"

    if stop_reason is not None:
        print(f"Stopping training due to {stop_reason}.")

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "best_model.pt"
    torch.save(best_state, checkpoint_path)
    print(f"Saved best model to {checkpoint_path}")

    net.load_state_dict(best_state["model_state_dict"])

    test_loss, test_mae, test_rmse = evaluate_split(net, test_loader, supports, device, loss_fn)
    print(
        f"Test results | loss {test_loss:.6f} | MAE {test_mae:.6f} | RMSE {test_rmse:.6f}"
    )

    if args.save_predictions:
        predictions = []
        masks = []
        targets = []
        net.eval()
        with torch.no_grad():
            for batch in test_loader:
                (week, week_mask, day, day_mask, recent, recent_mask, target, mask) = [
                    tensor.to(device) for tensor in batch
                ]
                output, _, _, _ = net(
                    week, week_mask, day, day_mask, recent, recent_mask, mask, supports
                )
                predictions.append(output.cpu().numpy())
                masks.append(mask.cpu().numpy())
                targets.append(target.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        masks = np.concatenate(masks, axis=0)
        targets = np.concatenate(targets, axis=0)

        prediction_path = output_dir / "test_predictions.npz"
        np.savez_compressed(
            prediction_path,
            predictions=predictions,
            masks=masks,
            targets=targets,
            adjacency=adjacency,
            stats=stats,
        )
        print(f"Saved predictions to {prediction_path}")

    metrics_path = output_dir / "metrics.json"
    metrics = {
        "val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "best_epoch": best_state["epoch"],
        "epochs_trained": epoch,
    }
    if stop_reason is not None:
        metrics["stop_reason"] = stop_reason
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Logged metrics to {metrics_path}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments and optional JSON configuration files."""

    if argv is None:
        argv = sys.argv[1:]

    # ``ArgumentParser`` marks options as required when neither defaults nor
    # values are supplied before parsing.  To support configuration files, we
    # first extract ``--config`` and load its contents so that the subsequent
    # parser can treat everything as optional until we combine the sources.
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=str,
        help=(
            "Path to a JSON configuration file. Values defined in the file are used "
            "as defaults and may be overridden via additional command line flags."
        ),
        default=None,
    )
    config_args, remaining_argv = config_parser.parse_known_args(argv)

    config_data: Dict[str, Any] = {}
    config_path: Optional[Path] = None
    if config_args.config is not None:
        config_path = Path(config_args.config).expanduser()
        if not config_path.is_file():
            config_parser.error(f"Configuration file {config_path} does not exist.")
        try:
            config_data = _load_config_file(config_path)
        except ValueError as exc:  # pragma: no cover - defensive branch
            config_parser.error(str(exc))

    defaults: Dict[str, Any] = {
        "timeseries": None,
        "adjacency": None,
        "delimiter": ",",
        "transpose_timeseries": False,
        "missing_value": None,
        "week_len": 12,
        "day_len": 12,
        "recent_len": 36,
        "target_len": 0,
        "train_ratio": 0.8,
        "val_ratio": 0.2,
        "impute_rate": 0.0,
        "impute_seed": None,
        "task": "impute",
        "batch_size": 16,
        "patience": 10,
        "max_epoch": None,
        "learning_rate": 5e-4,
        "weight_decay": 0.0,
        "lr_decay": 1.0,
        "device": "cpu",
        "model": "HSPGCNImputer",
        "hidden_dim": 64,
        "K": 3,
        "Kt": 3,
        "output_dir": "custom_experiments",
        "save_predictions": False,
        "no_shuffle": False,
        "single_scale": False,
    }

    if config_data:
        # Work on a shallow copy to avoid mutating the dictionary returned from
        # ``json.load`` if the caller reuses it elsewhere.
        config_data = dict(config_data)

        def _consume_alias(alias: str, canonical: str) -> None:
            """Map alternate configuration keys to the canonical CLI flag."""

            if alias in config_data:
                if canonical not in config_data:
                    config_data[canonical] = config_data[alias]
                config_data.pop(alias, None)

        # Allow intuitive aliases that mirror the helper function signature or
        # documentation examples.  Users that followed earlier revisions of the
        # script may still rely on these names.
        _consume_alias("timeseries_path", "timeseries")
        _consume_alias("adjacency_path", "adjacency")
        _consume_alias("output_path", "output_dir")
        _consume_alias("output_directory", "output_dir")
        _consume_alias("save_prediction", "save_predictions")
        _consume_alias("disable_multiscale", "single_scale")
        _consume_alias("single_scale_mode", "single_scale")

    parser = argparse.ArgumentParser(description=__doc__, argument_default=SUPPRESS)
    parser.add_argument(
        "--config",
        type=str,
        help=(
            "Path to a JSON configuration file. Values defined in the file are used "
            "as defaults and may be overridden via additional command line flags."
        ),
        default=SUPPRESS,
    )
    parser.add_argument("--timeseries", help="CSV file with graph signals.", default=SUPPRESS)
    parser.add_argument(
        "--adjacency", help="CSV file with the adjacency matrix.", default=SUPPRESS
    )
    parser.add_argument(
        "--delimiter",
        default=SUPPRESS,
        help="Delimiter shared by the CSV files (default: ',').",
    )
    parser.add_argument(
        "--transpose-timeseries",
        action="store_true",
        default=SUPPRESS,
        help="Set when the time-series CSV stores timesteps as rows instead of columns.",
    )
    parser.add_argument(
        "--missing-value",
        type=float,
        default=SUPPRESS,
        help="Optional placeholder value in the CSV that should be treated as missing.",
    )
    parser.add_argument("--week-len", type=int, default=SUPPRESS)
    parser.add_argument("--day-len", type=int, default=SUPPRESS)
    parser.add_argument("--recent-len", type=int, default=SUPPRESS)
    parser.add_argument(
        "--single-scale",
        action="store_true",
        default=SUPPRESS,
        help=(
            "Collapse the temporal context into a single window so the model only "
            "observes one scale of history."
        ),
    )
    parser.add_argument(
        "--target-len",
        type=int,
        default=SUPPRESS,
        help="Ignored for imputation; the reconstruction horizon equals week+day+recent.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=SUPPRESS,
        help="Fraction of samples allocated to training before chronological splitting.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=SUPPRESS,
        help=(
            "Fraction of samples allocated to testing. The validation split mirrors this "
            "portion so existing training loops can continue to reference 'val'."
        ),
    )
    parser.add_argument(
        "--task",
        choices=("forecast", "impute"),
        default=SUPPRESS,
        help=(
            "Task type. Selecting the HSPGCNImputer forces 'impute'; other models may"
            " expose forecasting heads."
        ),
    )
    parser.add_argument(
        "--impute-rate",
        type=float,
        default=SUPPRESS,
        help=(
            "Fraction of observed entries masked per sample. Use 0 to rely solely on"
            " the naturally missing values (default: 0)."
        ),
    )
    parser.add_argument(
        "--impute-seed",
        type=int,
        default=SUPPRESS,
        help="Random seed controlling the artificial masks for imputation (default: random).",
    )
    parser.add_argument("--batch-size", type=int, default=SUPPRESS)
    parser.add_argument(
        "--patience",
        type=int,
        default=SUPPRESS,
        help=(
            "Early-stopping patience. Training stops after this many epochs without "
            "validation improvement."
        ),
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=SUPPRESS,
        help=(
            "Optional cap on the number of training epochs. When omitted, training "
            "relies solely on the patience criterion."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=SUPPRESS)
    parser.add_argument("--weight-decay", type=float, default=SUPPRESS)
    parser.add_argument("--lr-decay", type=float, default=SUPPRESS)
    parser.add_argument(
        "--device",
        default=SUPPRESS,
        help="Computation device, e.g. 'cpu' or 'cuda:0'. Default uses the CPU.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_FACTORY.keys()),
        default=SUPPRESS,
        help="Model variant to train (default: HSPGCNImputer).",
    )
    parser.add_argument("--hidden-dim", type=int, default=SUPPRESS, help="Hidden dimension for the model.")
    parser.add_argument("--K", type=int, default=SUPPRESS, help="Chebyshev polynomial order.")
    parser.add_argument("--Kt", type=int, default=SUPPRESS, help="Temporal kernel size.")
    parser.add_argument(
        "--output-dir",
        default=SUPPRESS,
        help="Directory where checkpoints and predictions will be stored.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        default=SUPPRESS,
        help="Store the model predictions for the testing split as a compressed NPZ file.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        default=SUPPRESS,
        help="Disable shuffling of the training set batches.",
    )

    raw_args = parser.parse_args(remaining_argv)
    raw_dict = vars(raw_args)

    if config_data:
        known_options = set(defaults)
        known_options.add("config")
        unknown_keys = sorted(set(config_data) - known_options)
        if unknown_keys:
            parser.error("Unknown configuration options: " + ", ".join(unknown_keys))

    resolved: Dict[str, Any] = {**defaults, **config_data, **raw_dict}
    resolved["config"] = str(config_path) if config_path is not None else None

    args = Namespace(**resolved)

    if config_path is not None:
        config_dir = config_path.parent

        def normalize_path(field: str, *, must_exist: bool) -> None:
            if field not in config_data or field in raw_dict:
                return
            current_value = getattr(args, field, None)
            if not isinstance(current_value, str):
                return

            candidate = Path(current_value).expanduser()
            if candidate.is_absolute():
                setattr(args, field, str(candidate))
                return

            config_relative = (config_dir / candidate).resolve(strict=False)
            cwd_relative = candidate.resolve(strict=False)

            chosen_path: Optional[Path] = None
            if must_exist:
                if config_relative.exists():
                    chosen_path = config_relative
                elif cwd_relative.exists():
                    chosen_path = cwd_relative
            else:
                config_parent_exists = config_relative.parent.exists()
                cwd_parent_exists = cwd_relative.parent.exists()
                if config_parent_exists and not cwd_parent_exists:
                    chosen_path = config_relative
                elif cwd_parent_exists and not config_parent_exists:
                    chosen_path = cwd_relative

            if chosen_path is None:
                chosen_path = config_relative

            setattr(args, field, str(chosen_path))

        normalize_path("timeseries", must_exist=True)
        normalize_path("adjacency", must_exist=True)
        normalize_path("output_dir", must_exist=False)

    if args.timeseries is None:
        parser.error(
            "A time-series CSV must be provided via --timeseries or the configuration file."
        )
    if args.adjacency is None:
        parser.error(
            "An adjacency CSV must be provided via --adjacency or the configuration file."
        )

    if "model" not in raw_dict and "model" not in config_data:
        args.model = "HSPGCNImputer"

    # ``task`` used to be a required CLI flag in earlier revisions when both
    # forecasting and imputation were supported.  Some user configurations still
    # provide it, and the legacy command-line check raised an error if the
    # imputation head was selected without explicitly setting ``--task impute``.
    # To keep those setups working, automatically coerce the task to ``impute``
    # whenever the dedicated imputation model is requested.
    if getattr(args, "model", None) == "HSPGCNImputer":
        args.task = "impute"

    if args.patience is not None and args.patience < 1:
        parser.error("--patience must be a positive integer when provided.")

    if args.max_epoch is not None and args.max_epoch < 1:
        parser.error("--max-epoch must be a positive integer when provided.")

    if getattr(args, "single_scale", False):
        total_history = args.week_len + args.day_len + args.recent_len
        if total_history <= 0:
            parser.error(
                "--single-scale requires a positive total history length; adjust the "
                "week/day/recent parameters."
            )
        args.week_len = total_history
        args.day_len = 0
        args.recent_len = 0

    if args.patience is None and args.max_epoch is None:
        parser.error("At least one of --patience or --max-epoch must be specified.")

    return args


if __name__ == "__main__":
    train(parse_args())
