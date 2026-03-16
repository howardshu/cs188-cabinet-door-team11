"""
Step 11: Train a Low-Dim BC Policy with 1D U-Net
=================================================
Trains a behavior cloning policy that predicts action chunks from
low-dimensional state + handle features. This is the recommended
pipeline for the OpenCabinet dataset scale (≈100 demos).

Prerequisites:
    python 05b_augment_handle_data.py   # creates augmented/ parquet data

Usage:
    python 11_train_bc_unet.py
    python 11_train_bc_unet.py --epochs 100 --batch_size 128
    python 11_train_bc_unet.py --checkpoint_dir /tmp/bc_unet_checkpoints
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

from policy_models import BCUnet1DPolicy


def ensure_local_dependency_paths():
    repo_root = Path(__file__).resolve().parents[1]
    for dep in ("robocasa", "robosuite"):
        dep_path = repo_root / dep
        if dep_path.exists():
            dep_str = str(dep_path)
            if dep_str not in sys.path:
                sys.path.insert(0, dep_str)
    dp_path = Path(__file__).resolve().parent / "diffusion_policy"
    if dp_path.exists():
        dp_str = str(dp_path)
        if dp_str not in sys.path:
            sys.path.insert(0, dp_str)


ensure_local_dependency_paths()


def get_dataset_path(override=None):
    if override and os.path.exists(override):
        return override
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry_utils import get_ds_path

    path = get_ds_path("OpenCabinet", source="human")
    if path is None or not os.path.exists(path):
        print("ERROR: Dataset not found. Run 04_download_dataset.py first.")
        sys.exit(1)
    return path


AUGMENTED_OBS_KEYS = [
    "observation.handle_pos",
    "observation.handle_to_eef_pos",
]


def _load_augmented_episodes(
    dataset_path,
    state_key="observation.state",
    augmented_keys=None,
    max_episodes=None,
):
    import pyarrow.parquet as pq

    dataset_path = Path(dataset_path)
    aug_dir = dataset_path / "augmented"
    if not aug_dir.exists():
        orig_data = dataset_path / "data" / "chunk-000"
        if not orig_data.exists():
            orig_data = dataset_path / "lerobot" / "data" / "chunk-000"
        aug_dir = orig_data
        print(f"WARNING: No augmented/ directory found; falling back to {aug_dir}")
        print("         Run 05b_augment_handle_data.py first for best results.")

    if augmented_keys is None:
        augmented_keys = AUGMENTED_OBS_KEYS

    parquet_files = sorted(aug_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {aug_dir}")

    episodes = []
    episodes_loaded = 0
    for pf in parquet_files:
        df = pq.read_table(pf).to_pandas()
        if state_key not in df.columns or "action" not in df.columns:
            continue

        states = np.stack(df[state_key].to_list()).astype(np.float32)
        actions = np.stack(df["action"].to_list()).astype(np.float32)

        aug_parts = []
        for key in augmented_keys:
            if key in df.columns:
                arr = np.stack(df[key].to_list()).astype(np.float32)
                if arr.ndim == 1:
                    arr = arr[:, None]
                aug_parts.append(arr)
            else:
                print(f"WARNING: Missing {key} in {pf.name}; skipping episode.")
                aug_parts = None
                break

        if aug_parts is None:
            continue

        if aug_parts:
            aug_feats = np.concatenate(aug_parts, axis=1)
            states = np.concatenate([states, aug_feats], axis=1)

        episodes.append({"states": states, "actions": actions})
        episodes_loaded += 1
        if max_episodes is not None and episodes_loaded >= int(max_episodes):
            break

    if not episodes:
        raise RuntimeError("No valid episodes found.")

    return episodes, augmented_keys


class SequenceDataset:
    def __init__(self, episodes, episode_indices, n_obs_steps, n_action_steps):
        self.episodes = episodes
        self.episode_indices = episode_indices
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.sample_index = []

        for ep_idx in episode_indices:
            ep = episodes[ep_idx]
            states = ep["states"]
            if len(states) < (n_obs_steps + n_action_steps):
                continue
            max_anchor = len(states) - n_action_steps
            for anchor in range(n_obs_steps - 1, max_anchor):
                self.sample_index.append((ep_idx, anchor))

        if not self.sample_index:
            raise RuntimeError("No valid sequences found for the selected episodes.")

        self.state_mean = None
        self.state_std = None
        self.action_mean = None
        self.action_std = None

    def set_normalization(self, state_mean, state_std, action_mean, action_std):
        self.state_mean = state_mean
        self.state_std = state_std
        self.action_mean = action_mean
        self.action_std = action_std

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        import torch

        ep_idx, anchor = self.sample_index[idx]
        ep = self.episodes[ep_idx]
        obs_start = anchor - self.n_obs_steps + 1
        obs_end = anchor + 1
        act_end = anchor + self.n_action_steps

        states = ep["states"][obs_start:obs_end]
        actions = ep["actions"][anchor:act_end]

        if self.state_mean is not None:
            states = (states - self.state_mean) / self.state_std
        if self.action_mean is not None:
            actions = (actions - self.action_mean) / self.action_std

        return {
            "state": torch.from_numpy(states.astype(np.float32)),
            "action": torch.from_numpy(actions.astype(np.float32)),
        }


def compute_stats(episodes, episode_indices):
    states = []
    actions = []
    for ep_idx in episode_indices:
        ep = episodes[ep_idx]
        states.append(ep["states"])
        actions.append(ep["actions"])
    all_states = np.concatenate(states, axis=0)
    all_actions = np.concatenate(actions, axis=0)
    state_mean = all_states.mean(axis=0, keepdims=True).astype(np.float32)
    state_std = (all_states.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    action_mean = all_actions.mean(axis=0, keepdims=True).astype(np.float32)
    action_std = (all_actions.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    return state_mean, state_std, action_mean, action_std


def main():
    parser = argparse.ArgumentParser(description="Train low-dim BC U-Net policy")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--n_obs_steps", type=int, default=2)
    parser.add_argument("--n_action_steps", type=int, default=16)
    parser.add_argument("--execute_steps", type=int, default=8)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--channel_mults", type=str, default="1,2")
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--cond_dim", type=int, default=256)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="/tmp/bc_unet_checkpoints")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min_epochs", type=int, default=2)
    parser.add_argument(
        "--no_handle_pos",
        action="store_false",
        dest="use_handle_pos",
        help="Disable handle_pos feature",
    )
    parser.add_argument(
        "--no_handle_to_eef",
        action="store_false",
        dest="use_handle_to_eef",
        help="Disable handle_to_eef feature",
    )
    parser.set_defaults(use_handle_pos=True, use_handle_to_eef=True)
    args = parser.parse_args()

    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install with: pip install torch pyarrow")
        sys.exit(1)

    print("=" * 60)
    print("  OpenCabinet - BC 1D U-Net Policy Training")
    print("=" * 60)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    dataset_path = get_dataset_path(args.dataset_path)
    print(f"Dataset: {dataset_path}")

    augmented_keys = []
    if args.use_handle_pos:
        augmented_keys.append("observation.handle_pos")
    if args.use_handle_to_eef:
        augmented_keys.append("observation.handle_to_eef_pos")
    if not augmented_keys:
        augmented_keys = []
        print("NOTE: Training without handle features (proprio only).")

    episodes, used_keys = _load_augmented_episodes(
        dataset_path=dataset_path,
        augmented_keys=augmented_keys,
        max_episodes=args.max_episodes,
    )

    rng = np.random.default_rng(args.seed)
    ep_indices = np.arange(len(episodes))
    rng.shuffle(ep_indices)
    val_count = max(1, int(len(ep_indices) * args.val_fraction))
    val_indices = ep_indices[:val_count].tolist()
    train_indices = ep_indices[val_count:].tolist()

    train_dataset = SequenceDataset(
        episodes, train_indices, args.n_obs_steps, args.n_action_steps
    )
    val_dataset = SequenceDataset(
        episodes, val_indices, args.n_obs_steps, args.n_action_steps
    )

    state_mean, state_std, action_mean, action_std = compute_stats(episodes, train_indices)
    train_dataset.set_normalization(state_mean, state_std, action_mean, action_std)
    val_dataset.set_normalization(state_mean, state_std, action_mean, action_std)

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    channel_mults = [int(x) for x in args.channel_mults.split(",") if x.strip()]
    min_len = 2 ** len(channel_mults)
    if args.n_action_steps < min_len:
        print(
            f"WARNING: n_action_steps={args.n_action_steps} may be too small for "
            f"channel_mults={channel_mults} (downsample factor={min_len})."
        )

    model = BCUnet1DPolicy(
        state_dim=episodes[0]["states"].shape[-1],
        action_dim=episodes[0]["actions"].shape[-1],
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        base_channels=args.base_channels,
        channel_mults=channel_mults,
        kernel_size=args.kernel_size,
        cond_dim=args.cond_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    metrics_path = os.path.join(args.checkpoint_dir, "training_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "val_action_mse", "early_stop"]
        )

    action_mean_t = torch.from_numpy(action_mean).to(device).view(1, 1, -1)
    action_std_t = torch.from_numpy(action_std).to(device).view(1, 1, -1)

    print(
        f"\nTraining: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}"
    )
    print(
        f"  n_obs_steps={args.n_obs_steps}, n_action_steps={args.n_action_steps}, execute_steps={args.execute_steps}"
    )
    print(f"  train episodes={len(train_indices)}, val episodes={len(val_indices)}")
    print(f"  train samples={len(train_dataset)}, val samples={len(val_dataset)}")
    print(f"  augmented keys: {used_keys}")
    print()

    best_loss = float("inf")
    best_epoch = 0
    patience_left = args.patience

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            obs_state = batch["state"].to(device)
            action_chunk = batch["action"].to(device)

            pred = model(obs_state)
            loss = F.mse_loss(pred, action_chunk)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)

        model.eval()
        val_losses = []
        val_mses = []
        with torch.no_grad():
            for vb in val_dataloader:
                obs_v = vb["state"].to(device)
                act_v = vb["action"].to(device)
                pred_v = model(obs_v)
                val_losses.append(F.mse_loss(pred_v, act_v).item())

                pred_denorm = pred_v * action_std_t + action_mean_t
                gt_denorm = act_v * action_std_t + action_mean_t
                val_mses.append(F.mse_loss(pred_denorm, gt_denorm).item())

        val_loss = float(np.mean(val_losses))
        val_action_mse = float(np.mean(val_mses))
        print(
            f"  Epoch {epoch+1:4d}/{args.epochs}  train={avg_loss:.6f}  "
            f"val={val_loss:.6f}  val_action_mse={val_action_mse:.4f}"
        )

        early_stop = False
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            patience_left = args.patience
            ckpt_data = {
                "epoch": epoch,
                "loss": best_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_type": "bc_unet_lowdim",
                "state_dim": episodes[0]["states"].shape[-1],
                "action_dim": episodes[0]["actions"].shape[-1],
                "n_obs_steps": args.n_obs_steps,
                "n_action_steps": args.n_action_steps,
                "execute_steps": args.execute_steps,
                "base_channels": args.base_channels,
                "channel_mults": channel_mults,
                "kernel_size": args.kernel_size,
                "cond_dim": args.cond_dim,
                "augmented_obs_keys": used_keys,
                "state_mean": state_mean.astype(np.float32),
                "state_std": state_std.astype(np.float32),
                "action_mean": action_mean.astype(np.float32),
                "action_std": action_std.astype(np.float32),
            }
            torch.save(ckpt_data, os.path.join(args.checkpoint_dir, "best_policy.pt"))
        else:
            if epoch + 1 >= args.min_epochs:
                patience_left -= 1
                if patience_left <= 0:
                    early_stop = True

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, avg_loss, val_loss, val_action_mse, early_stop])

        if early_stop:
            print(
                f"Early stopping at epoch {epoch+1}. Best val loss at epoch {best_epoch}."
            )
            break

    final_path = os.path.join(args.checkpoint_dir, "final_policy.pt")
    ckpt_data = {
        "epoch": epoch + 1,
        "loss": val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_type": "bc_unet_lowdim",
        "state_dim": episodes[0]["states"].shape[-1],
        "action_dim": episodes[0]["actions"].shape[-1],
        "n_obs_steps": args.n_obs_steps,
        "n_action_steps": args.n_action_steps,
        "execute_steps": args.execute_steps,
        "base_channels": args.base_channels,
        "channel_mults": channel_mults,
        "kernel_size": args.kernel_size,
        "cond_dim": args.cond_dim,
        "augmented_obs_keys": used_keys,
        "state_mean": state_mean.astype(np.float32),
        "state_std": state_std.astype(np.float32),
        "action_mean": action_mean.astype(np.float32),
        "action_std": action_std.astype(np.float32),
    }
    torch.save(ckpt_data, final_path)

    print("\nTraining complete!")
    print(f"Best checkpoint: {os.path.join(args.checkpoint_dir, 'best_policy.pt')}")
    print(f"Final checkpoint: {final_path}")
    print("\nTo evaluate:")
    print(f"  python 07_evaluate_policy.py --checkpoint {args.checkpoint_dir}/best_policy.pt")


if __name__ == "__main__":
    main()
