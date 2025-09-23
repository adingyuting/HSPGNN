"""Train an HSPGNN model on a custom CSV dataset without modifying the repo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from custom_dataset_utils import prepare_custom_dataset
from model import HSPGCN, HSPGCN_L


MODEL_FACTORY = {
    "HSPGCN": HSPGCN,
    "HSPGCN_L": HSPGCN_L,
}


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
    )

    temporal_window = args.week_len + args.day_len + args.recent_len
    if temporal_window != 60:
        raise ValueError(
            "The current model expects week_len + day_len + recent_len to equal 60."
        )
    if args.target_len != 6:
        raise ValueError("The current model outputs six steps; set --target-len 6.")

    num_nodes = adjacency.shape[0]
    device = torch.device(args.device)

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

    for epoch in range(1, args.max_epoch + 1):
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
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Logged metrics to {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeseries", required=True, help="CSV file with graph signals.")
    parser.add_argument("--adjacency", required=True, help="CSV file with the adjacency matrix.")
    parser.add_argument(
        "--delimiter", default=",", help="Delimiter shared by the CSV files (default: ',')."
    )
    parser.add_argument(
        "--transpose-timeseries",
        action="store_true",
        help="Set when the time-series CSV stores timesteps as rows instead of columns.",
    )
    parser.add_argument(
        "--missing-value",
        type=float,
        default=None,
        help="Optional placeholder value in the CSV that should be treated as missing.",
    )
    parser.add_argument("--week-len", type=int, default=12)
    parser.add_argument("--day-len", type=int, default=12)
    parser.add_argument("--recent-len", type=int, default=36)
    parser.add_argument("--target-len", type=int, default=6)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epoch", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-decay", type=float, default=1.0)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Computation device, e.g. 'cpu' or 'cuda:0'. Default uses the CPU.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_FACTORY.keys()),
        default="HSPGCN",
        help="Model variant to train (default: HSPGCN).",
    )
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension for the model.")
    parser.add_argument("--K", type=int, default=3, help="Chebyshev polynomial order.")
    parser.add_argument("--Kt", type=int, default=3, help="Temporal kernel size.")
    parser.add_argument(
        "--output-dir",
        default="custom_experiments",
        help="Directory where checkpoints and predictions will be stored.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Store the model predictions for the testing split as a compressed NPZ file.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling of the training set batches.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
