"""
Step 6b: Train a Low-Dim Diffusion Policy with 1D U-Net
=======================================================
Trains a state-based (no images) diffusion policy using the
ConditionalUnet1D architecture from the diffusion_policy repo.
The model is conditioned on the robot's low-dim state (including
augmented handle features from 05b_augment_handle_data.py) and
predicts action chunks.

This is significantly more capable than the MLP denoiser (~15M params
vs ~2M) while being small enough to train without a large GPU.

Prerequisite:
    python 05b_augment_handle_data.py   # creates augmented/ parquet data

Usage:
    python 06b_train_diffusion_unet.py
    python 06b_train_diffusion_unet.py --epochs 200 --batch_size 256
    python 06b_train_diffusion_unet.py --checkpoint_dir /tmp/unet_checkpoints
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np


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


class AugmentedLowDimDataset:
    """Loads state + augmented handle features + actions from parquet files."""

    def __init__(
        self,
        dataset_path,
        n_obs_steps=2,
        n_action_steps=8,
        state_key="observation.state",
        augmented_keys=None,
        max_episodes=None,
    ):
        import pyarrow.parquet as pq
        import torch

        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.episodes = []
        self.sample_index = []

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

            if aug_parts:
                aug_feats = np.concatenate(aug_parts, axis=1)
                states = np.concatenate([states, aug_feats], axis=1)

            if len(states) < (n_obs_steps + n_action_steps):
                continue

            self.episodes.append({"states": states, "actions": actions})
            ep_idx = len(self.episodes) - 1
            max_anchor = len(states) - n_action_steps
            for anchor in range(n_obs_steps - 1, max_anchor):
                self.sample_index.append((ep_idx, anchor))

            episodes_loaded += 1
            if max_episodes is not None and episodes_loaded >= int(max_episodes):
                break

        if not self.episodes:
            raise RuntimeError("No valid episodes found.")

        all_states = np.concatenate([ep["states"] for ep in self.episodes], axis=0)
        all_actions = np.concatenate([ep["actions"] for ep in self.episodes], axis=0)
        self.state_mean = all_states.mean(axis=0, keepdims=True).astype(np.float32)
        self.state_std = (all_states.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
        self.action_mean = all_actions.mean(axis=0, keepdims=True).astype(np.float32)
        self.action_std = (all_actions.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
        self.state_dim = all_states.shape[-1]
        self.action_dim = all_actions.shape[-1]

        print(f"Loaded {len(self.episodes)} episodes, {len(self.sample_index)} samples")
        print(f"  state_dim={self.state_dim}  action_dim={self.action_dim}")
        has_aug = any(k in df.columns for k in augmented_keys) if aug_parts else False
        if has_aug:
            print(f"  augmented features: {[k for k in augmented_keys if k in df.columns]}")

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
        states = (states - self.state_mean) / self.state_std
        actions = ep["actions"][anchor:act_end]
        actions = (actions - self.action_mean) / self.action_std

        return {
            "state": torch.from_numpy(states.astype(np.float32)),
            "action": torch.from_numpy(actions.astype(np.float32)),
        }


def build_unet_model(action_dim, obs_cond_dim, config):
    """Build the ConditionalUnet1D model."""
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

    down_dims = config.get("down_dims", [256, 512, 1024])
    if isinstance(down_dims, str):
        down_dims = [int(x) for x in down_dims.split(",")]

    model = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_cond_dim,
        diffusion_step_embed_dim=config.get("diffusion_step_embed_dim", 256),
        down_dims=down_dims,
        kernel_size=config.get("kernel_size", 5),
        n_groups=config.get("n_groups", 8),
        cond_predict_scale=config.get("cond_predict_scale", False),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ConditionalUnet1D: {n_params/1e6:.1f}M parameters")
    return model


def sample_actions(model, obs_cond, scheduler, n_action_steps, action_dim, device, num_inference_steps=16):
    """Sample action chunk from the diffusion model."""
    import torch

    scheduler.set_timesteps(num_inference_steps)
    bsz = obs_cond.shape[0]
    x = torch.randn((bsz, n_action_steps, action_dim), device=device)

    for t in scheduler.timesteps:
        t_batch = torch.full((bsz,), int(t.item()), device=device, dtype=torch.long)
        pred_noise = model(x, t_batch, global_cond=obs_cond)
        x = scheduler.step(pred_noise, t, x).prev_sample

    return x


def main():
    parser = argparse.ArgumentParser(description="Train low-dim U-Net diffusion policy")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--n_obs_steps", type=int, default=2)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--num_inference_steps", type=int, default=16)
    parser.add_argument("--down_dims", type=str, default="256,512")
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="diffusion_unet_policy_checkpoints/")
    parser.add_argument("--checkpoint_every", type=int, default=20)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_every", type=int, default=20)
    args = parser.parse_args()

    try:
        import torch
        import torch.nn.functional as F
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from torch.utils.data import DataLoader, Subset
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install with: pip install torch diffusers einops pyarrow")
        sys.exit(1)

    print("=" * 60)
    print("  OpenCabinet - Low-Dim U-Net Diffusion Policy Training")
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

    down_dims = [int(x) for x in args.down_dims.split(",")] 

    dataset = AugmentedLowDimDataset(
        dataset_path=dataset_path,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        max_episodes=args.max_episodes,
    )

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(dataset))
    val_count = max(1, int(len(dataset) * args.val_fraction))
    val_indices = perm[:val_count].tolist()
    train_indices = perm[val_count:].tolist()
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

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

    obs_cond_dim = args.n_obs_steps * dataset.state_dim
    config = {
        "down_dims": down_dims,
        "diffusion_step_embed_dim": 256,
        "kernel_size": args.kernel_size,
        "n_groups": 8,
        "cond_predict_scale": False,
    }
    model = build_unet_model(dataset.action_dim, obs_cond_dim, config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_diffusion_steps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    warmup = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, float(step + 1) / max(1, args.lr_warmup_steps)),
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    metrics_path = os.path.join(args.checkpoint_dir, "training_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_action_mse"])

    state_mean_t = torch.from_numpy(dataset.state_mean).to(device)
    state_std_t = torch.from_numpy(dataset.state_std).to(device)
    action_mean_t = torch.from_numpy(dataset.action_mean).to(device).view(1, 1, -1)
    action_std_t = torch.from_numpy(dataset.action_std).to(device).view(1, 1, -1)

    best_loss = float("inf")
    global_step = 0

    print(f"\nTraining: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"  n_obs_steps={args.n_obs_steps}, n_action_steps={args.n_action_steps}")
    print(f"  state_dim={dataset.state_dim}, action_dim={dataset.action_dim}")
    print(f"  obs_cond_dim={obs_cond_dim}")
    print(f"  train samples={len(train_dataset)}, val samples={len(val_dataset)}")
    print()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            obs_state = batch["state"].to(device)
            action_chunk = batch["action"].to(device)
            bsz = obs_state.shape[0]

            obs_cond = obs_state.reshape(bsz, -1)

            noise = torch.randn_like(action_chunk)
            timesteps = torch.randint(
                0, args.num_diffusion_steps, (bsz,), device=device
            ).long()
            noisy = noise_scheduler.add_noise(action_chunk, noise, timesteps)

            pred_noise = model(noisy, timesteps, global_cond=obs_cond)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            warmup.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

        avg_loss = epoch_loss / max(1, num_batches)

        val_loss = float("nan")
        val_action_mse = float("nan")
        if (epoch + 1) % args.eval_every == 0 or epoch == 0:
            model.eval()
            val_losses = []
            val_mses = []
            with torch.no_grad():
                for vb in val_dataloader:
                    obs_v = vb["state"].to(device)
                    act_v = vb["action"].to(device)
                    bsz_v = obs_v.shape[0]
                    obs_cond_v = obs_v.reshape(bsz_v, -1)

                    noise_v = torch.randn_like(act_v)
                    ts_v = torch.randint(
                        0, args.num_diffusion_steps, (bsz_v,), device=device
                    ).long()
                    noisy_v = noise_scheduler.add_noise(act_v, noise_v, ts_v)
                    pred_v = model(noisy_v, ts_v, global_cond=obs_cond_v)
                    val_losses.append(F.mse_loss(pred_v, noise_v).item())

                    pred_act = sample_actions(
                        model, obs_cond_v, noise_scheduler,
                        args.n_action_steps, dataset.action_dim, device,
                        num_inference_steps=args.num_inference_steps,
                    )
                    pred_denorm = pred_act * action_std_t + action_mean_t
                    gt_denorm = act_v * action_std_t + action_mean_t
                    val_mses.append(F.mse_loss(pred_denorm, gt_denorm).item())

            val_loss = float(np.mean(val_losses))
            val_action_mse = float(np.mean(val_mses))
            print(
                f"  Epoch {epoch+1:4d}/{args.epochs}  train={avg_loss:.6f}  "
                f"val_denoise={val_loss:.6f}  val_action_mse={val_action_mse:.4f}"
            )
        else:
            print(f"  Epoch {epoch+1:4d}/{args.epochs}  train={avg_loss:.6f}")

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, avg_loss, val_loss, val_action_mse])

        ckpt_data = {
            "epoch": epoch,
            "loss": avg_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_type": "unet_lowdim_diffusion",
            "state_dim": dataset.state_dim,
            "action_dim": dataset.action_dim,
            "n_obs_steps": args.n_obs_steps,
            "n_action_steps": args.n_action_steps,
            "num_diffusion_steps": args.num_diffusion_steps,
            "num_inference_steps": args.num_inference_steps,
            "down_dims": down_dims,
            "kernel_size": args.kernel_size,
            "diffusion_step_embed_dim": 256,
            "n_groups": 8,
            "cond_predict_scale": False,
            "augmented_obs_keys": AUGMENTED_OBS_KEYS,
            "state_mean": dataset.state_mean.astype(np.float32),
            "state_std": dataset.state_std.astype(np.float32),
            "action_mean": dataset.action_mean.astype(np.float32),
            "action_std": dataset.action_std.astype(np.float32),
        }

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt_data, os.path.join(args.checkpoint_dir, "best_policy.pt"))

        if (epoch + 1) % args.checkpoint_every == 0:
            torch.save(
                ckpt_data,
                os.path.join(args.checkpoint_dir, f"epoch_{epoch+1:04d}.pt"),
            )

    final_path = os.path.join(args.checkpoint_dir, "final_policy.pt")
    ckpt_data["epoch"] = args.epochs
    ckpt_data["loss"] = avg_loss
    ckpt_data["model_state_dict"] = model.state_dict()
    torch.save(ckpt_data, final_path)

    print(f"\nTraining complete!")
    print(f"Best checkpoint: {os.path.join(args.checkpoint_dir, 'best_policy.pt')}")
    print(f"Final checkpoint: {final_path}")
    print(f"\nTo evaluate:")
    print(f"  python 07_evaluate_policy.py --checkpoint {args.checkpoint_dir}/best_policy.pt")


if __name__ == "__main__":
    main()
