"""
Step 6: Train a Diffusion Policy
==================================
This script provides self-contained policy training for OpenCabinet.
It now includes a low-dimensional diffusion policy baseline that does
not depend on the external Hydra-based diffusion_policy repo.
This script provides self-contained policy training for OpenCabinet.
It now includes a low-dimensional diffusion policy baseline that does
not depend on the external Hydra-based diffusion_policy repo.

For production-quality training, use the official Diffusion Policy repo:
    git clone https://github.com/robocasa-benchmark/diffusion_policy
    cd diffusion_policy && pip install -e .
    python train.py --config-name=train_diffusion_transformer_bs192 task=robocasa/OpenCabinet

By default this script trains the diffusion baseline. A simple MLP BC
baseline remains available for comparison.
By default this script trains the diffusion baseline. A simple MLP BC
baseline remains available for comparison.

Usage:
    python 06_train_policy.py [--epochs 50] [--batch_size 32] [--lr 1e-4]
    python 06_train_policy.py --policy simple
    python 06_train_policy.py --policy diffusion
    python 06_train_policy.py --policy simple
    python 06_train_policy.py --policy diffusion
"""

import argparse
import csv
import csv
import os
import sys
import yaml
from pathlib import Path
from pathlib import Path

import numpy as np
from policy_models import DiffusionActionMLP, SimplePolicy, VisionDiffusionChunkPolicy


def ensure_local_dependency_paths():
    repo_root = Path(__file__).resolve().parents[1]
    for dep in ("robocasa", "robosuite"):
        dep_path = repo_root / dep
        if dep_path.exists():
            dep_str = str(dep_path)
            if dep_str not in sys.path:
                sys.path.insert(0, dep_str)
from policy_models import DiffusionActionMLP, SimplePolicy, VisionDiffusionChunkPolicy


def ensure_local_dependency_paths():
    repo_root = Path(__file__).resolve().parents[1]
    for dep in ("robocasa", "robosuite"):
        dep_path = repo_root / dep
        if dep_path.exists():
            dep_str = str(dep_path)
            if dep_str not in sys.path:
                sys.path.insert(0, dep_str)


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def load_config(config_path):
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_dataset_path(dataset_path_override=None):
def get_dataset_path(dataset_path_override=None):
    """Get the path to the OpenCabinet dataset."""
    if dataset_path_override:
        if os.path.exists(dataset_path_override):
            return dataset_path_override
        print(f"ERROR: dataset_path does not exist: {dataset_path_override}")
        sys.exit(1)
    ensure_local_dependency_paths()
    if dataset_path_override:
        if os.path.exists(dataset_path_override):
            return dataset_path_override
        print(f"ERROR: dataset_path does not exist: {dataset_path_override}")
        sys.exit(1)
    ensure_local_dependency_paths()
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry_utils import get_ds_path

    path = get_ds_path("OpenCabinet", source="human")
    if path is None or not os.path.exists(path):
        print("ERROR: Dataset not found. Run 04_download_dataset.py first.")
        sys.exit(1)
    return path


def get_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_lerobot_dirs(dataset_path):
    data_dir = os.path.join(dataset_path, "data")
    videos_dir = os.path.join(dataset_path, "videos")
    if not os.path.exists(data_dir):
        data_dir = os.path.join(dataset_path, "lerobot", "data")
    if not os.path.exists(videos_dir):
        videos_dir = os.path.join(dataset_path, "lerobot", "videos")
    if not os.path.exists(data_dir) or not os.path.exists(videos_dir):
        raise FileNotFoundError(
            f"Could not find data/videos under {dataset_path}. "
            "Expected LeRobot dataset layout."
        )
    return data_dir, videos_dir


class LerobotVisionChunkDataset:
    def __init__(
        self,
        dataset_path,
        image_keys,
        n_obs_steps,
        n_action_steps,
        image_size,
        state_key="observation.state",
        max_episodes=70,
        use_handle_pos=False,
    ):
        import pyarrow.parquet as pq

        self.image_keys = image_keys
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.image_size = image_size
        self.state_key = state_key
        self.use_handle_pos = use_handle_pos
        self.episodes = []
        self.sample_index = []
        self._video_cache = {}

        data_dir, videos_dir = get_lerobot_dirs(dataset_path)
        chunk_dir = os.path.join(data_dir, "chunk-000")
        parquet_files = sorted(f for f in os.listdir(chunk_dir) if f.endswith(".parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {chunk_dir}")

        extras_dir = self._find_extras_dir(dataset_path)

        episodes_loaded = 0
        handle_pos_missing = 0
        for pf in parquet_files:
            ep_name = os.path.splitext(pf)[0]
            df = pq.read_table(os.path.join(chunk_dir, pf)).to_pandas()
            if state_key not in df.columns or "action" not in df.columns:
                continue
            states = np.stack(df[state_key].to_list()).astype(np.float32)
            actions = np.stack(df["action"].to_list()).astype(np.float32)
            frame_idxs = np.asarray(df["frame_index"], dtype=np.int32)
            if len(states) < (self.n_obs_steps + self.n_action_steps):
                continue

            if self.use_handle_pos and extras_dir is not None:
                handle_path = os.path.join(extras_dir, ep_name, "handle_pos.npy")
                if not os.path.exists(handle_path):
                    handle_pos_missing += 1
                    continue
                handle_pos = np.load(handle_path).astype(np.float32)
                if len(handle_pos) != len(states):
                    handle_pos = handle_pos[:len(states)]
                    if len(handle_pos) < len(states):
                        pad = np.tile(handle_pos[-1:], (len(states) - len(handle_pos), 1))
                        handle_pos = np.concatenate([handle_pos, pad], axis=0)
                states = np.concatenate([states, handle_pos], axis=1)

            video_paths = {}
            valid = True
            for cam in self.image_keys:
                cam_dir = os.path.join(videos_dir, "chunk-000", f"observation.images.{cam}")
                cam_path = os.path.join(cam_dir, f"{ep_name}.mp4")
                if not os.path.exists(cam_path):
                    valid = False
                    break
                video_paths[cam] = cam_path
            if not valid:
                continue

            self.episodes.append(
                {
                    "states": states,
                    "actions": actions,
                    "frame_idxs": frame_idxs,
                    "video_paths": video_paths,
                }
            )
            ep_idx = len(self.episodes) - 1
            max_anchor = len(states) - self.n_action_steps
            for anchor in range(self.n_obs_steps - 1, max_anchor):
                self.sample_index.append((ep_idx, anchor))
            episodes_loaded += 1
            if max_episodes is not None and episodes_loaded >= int(max_episodes):
                break

        if not self.episodes:
            raise RuntimeError("No valid episodes found with state/action + all cameras.")

        if self.use_handle_pos and handle_pos_missing > 0:
            print(
                f"WARNING: {handle_pos_missing} episodes skipped (no handle_pos.npy). "
                "Run 09_cache_handle_positions.py first."
            )

        all_states = np.concatenate([ep["states"] for ep in self.episodes], axis=0)
        all_actions = np.concatenate([ep["actions"] for ep in self.episodes], axis=0)
        self.state_mean = all_states.mean(axis=0, keepdims=True).astype(np.float32)
        self.state_std = (all_states.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
        self.action_mean = all_actions.mean(axis=0, keepdims=True).astype(np.float32)
        self.action_std = (all_actions.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
        self.state_dim = all_states.shape[-1]
        self.action_dim = all_actions.shape[-1]

    @staticmethod
    def _find_extras_dir(dataset_path):
        """Locate the extras/ directory within the dataset."""
        candidates = [
            os.path.join(dataset_path, "extras"),
            os.path.join(dataset_path, "lerobot", "extras"),
        ]
        for c in candidates:
            if os.path.isdir(c):
                return c
        return None

    def __len__(self):
        return len(self.sample_index)

    def _get_reader(self, path):
        import imageio.v2 as imageio

        if path not in self._video_cache:
            self._video_cache[path] = imageio.get_reader(path, "ffmpeg")
        return self._video_cache[path]

    def _read_frame(self, path, frame_idx):
        import torch

        reader = self._get_reader(path)
        try:
            frame = reader.get_data(int(frame_idx))
        except Exception:
            frame = reader.get_data(max(int(frame_idx) - 1, 0))

        if frame.shape[0] != self.image_size or frame.shape[1] != self.image_size:
            frame_t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
            frame_t = torch.nn.functional.interpolate(
                frame_t,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            frame = frame_t.squeeze(0).permute(1, 2, 0).byte().numpy()

        frame = frame.astype(np.float32) / 255.0
        return np.transpose(frame, (2, 0, 1))

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
        frame_idxs = ep["frame_idxs"][obs_start:obs_end]

        sample = {
            "state": torch.from_numpy(states.astype(np.float32)),
            "action": torch.from_numpy(actions.astype(np.float32)),
        }
        for cam in self.image_keys:
            cam_frames = [self._read_frame(ep["video_paths"][cam], fi) for fi in frame_idxs]
            sample[cam] = torch.from_numpy(np.stack(cam_frames).astype(np.float32))
        return sample


def train_simple_policy(config):
    """
    Train a simple behavior-cloning policy.

    This is a simplified training loop to illustrate the pipeline.
    For real results, use the official Diffusion Policy codebase.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError:
        print("ERROR: PyTorch is required for training.")
        print("Install with: pip install torch torchvision")
        sys.exit(1)

    print_section("Simple Behavior Cloning Policy")

    dataset_path = get_dataset_path(config.get("dataset_path", None))
    dataset_path = get_dataset_path(config.get("dataset_path", None))
    print(f"Dataset: {dataset_path}")

    # ----------------------------------------------------------------
    # 1. Build a simple dataset from the LeRobot format
    # ----------------------------------------------------------------
    print("\nLoading dataset...")

    class CabinetDemoDataset(Dataset):
        """
        Loads state-action pairs from the LeRobot-format dataset.

        For simplicity, this uses only the low-dimensional state observations
        (gripper qpos, base pose, eef pose) rather than images.
        Full visuomotor training with images requires the Diffusion Policy repo.
        """

        def __init__(self, dataset_path, max_episodes=None):
            import pyarrow.parquet as pq

            self.states = []
            self.actions = []

            # The dataset path from get_ds_path may point to the lerobot dir directly
            # or to the parent. Try both layouts.
            data_dir = os.path.join(dataset_path, "data")
            if not os.path.exists(data_dir):
                data_dir = os.path.join(dataset_path, "lerobot", "data")
            if not os.path.exists(data_dir):
                raise FileNotFoundError(
                    f"Data directory not found under: {dataset_path}\n"
                    "Make sure you downloaded the dataset with 04_download_dataset.py"
                )

            # Load parquet files
            chunk_dir = os.path.join(data_dir, "chunk-000")
            if not os.path.exists(chunk_dir):
                raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

            parquet_files = sorted(
                f for f in os.listdir(chunk_dir) if f.endswith(".parquet")
            )
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found in {chunk_dir}")

            episodes_loaded = 0
            for pf in parquet_files:
                table = pq.read_table(os.path.join(chunk_dir, pf))
                df = table.to_pandas()

                # Extract state and action columns
                state_cols = [
                    c for c in df.columns if c.startswith("observation.state")
                ]
                action_cols = [
                    c for c in df.columns
                    if c == "action" or c.startswith("action.")
                ]

                if not state_cols or not action_cols:
                    # Try alternative column naming
                    state_cols = [
                        c
                        for c in df.columns
                        if "gripper" in c or "base" in c or "eef" in c
                    ]
                    action_cols = [c for c in df.columns if "action" in c]

                if state_cols and action_cols:
                    for _, row in df.iterrows():
                        # Values may be numpy arrays (object columns) or scalars
                        state_parts = []
                        for c in state_cols:
                            val = row[c]
                            if isinstance(val, np.ndarray):
                                state_parts.extend(val.flatten().tolist())
                            elif isinstance(val, (int, float, np.floating)):
                                state_parts.append(float(val))
                        action_parts = []
                        for c in action_cols:
                            val = row[c]
                            if isinstance(val, np.ndarray):
                                action_parts.extend(val.flatten().tolist())
                            elif isinstance(val, (int, float, np.floating)):
                                action_parts.append(float(val))

                        if state_parts and action_parts:
                            self.states.append(np.array(state_parts, dtype=np.float32))
                            self.actions.append(np.array(action_parts, dtype=np.float32))

                episodes_loaded += 1
                if max_episodes and episodes_loaded >= max_episodes:
                    break

            if len(self.states) == 0:
                print("WARNING: Could not extract state-action pairs from parquet files.")
                print("The dataset may use a different format.")
                print("Generating synthetic demo data for illustration...")
                self._generate_synthetic_data()

            self.states = np.array(self.states, dtype=np.float32)
            self.actions = np.array(self.actions, dtype=np.float32)

            print(f"Loaded {len(self.states)} state-action pairs")
            print(f"State dim:  {self.states.shape[-1]}")
            print(f"Action dim: {self.actions.shape[-1]}")

        def _generate_synthetic_data(self):
            """Generate synthetic data for demonstration purposes."""
            rng = np.random.default_rng(42)
            for _ in range(1000):
                state = rng.standard_normal(16).astype(np.float32)
                action = rng.standard_normal(12).astype(np.float32) * 0.1
                self.states.append(state)
                self.actions.append(action)

        def __len__(self):
            return len(self.states)

        def __getitem__(self, idx):
            return (
                torch.from_numpy(self.states[idx]),
                torch.from_numpy(self.actions[idx]),
            )

    dataset = CabinetDemoDataset(dataset_path, max_episodes=50)
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    # ----------------------------------------------------------------
    # 2. Define a simple MLP policy
    # ----------------------------------------------------------------
    state_dim = dataset.states.shape[-1]
    action_dim = dataset.actions.shape[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = SimplePolicy(state_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # ----------------------------------------------------------------
    # 3. Training loop
    # ----------------------------------------------------------------
    print_section("Training")
    print(f"Epochs:     {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"LR:         {config['learning_rate']}")

    checkpoint_dir = config.get("checkpoint_dir", "/tmp/cabinet_policy_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_loss = float("inf")
    avg_loss = float("inf")
    ckpt_path = os.path.join(checkpoint_dir, "best_policy.pt")
    for epoch in range(config["epochs"]):
        epoch_loss = 0.0
        num_batches = 0

        model.train()
        for states_batch, actions_batch in dataloader:
            states_batch = states_batch.to(device)
            actions_batch = actions_batch.to(device)

            pred_actions = model(states_batch)
            loss = nn.functional.mse_loss(pred_actions, actions_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:4d}/{config['epochs']}  Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(checkpoint_dir, "best_policy.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "model_type": "simple_mlp",
                    "model_type": "simple_mlp",
                },
                ckpt_path,
            )

    # Save final checkpoint
    final_path = os.path.join(checkpoint_dir, "final_policy.pt")
    torch.save(
        {
            "epoch": config["epochs"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "model_type": "simple_mlp",
            "model_type": "simple_mlp",
        },
        final_path,
    )

    print(f"\nTraining complete!")
    print(f"Best loss:        {best_loss:.6f}")
    print(f"Best checkpoint:  {ckpt_path}")
    print(f"Final checkpoint: {final_path}")

    print_section("Next Steps")
    print(
        "Try the built-in diffusion baseline for stronger performance:\n"
        "  python 06_train_policy.py --policy diffusion\n"
    )


def train_vision_diffusion_chunk_policy(config):
    """
    Train a vision-conditioned diffusion policy that predicts action chunks.
    """
    try:
        import torch
        import torch.nn.functional as F
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from torch.utils.data import DataLoader, Subset
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install with: pip install torch diffusers imageio pyarrow")
        sys.exit(1)

    print_section("Vision Diffusion Chunk Policy (Starter)")
    dataset_path = get_dataset_path(config.get("dataset_path", None))
    print(f"Dataset: {dataset_path}")

    n_obs_steps = int(config.get("n_obs_steps", 2))
    n_action_steps = int(config.get("n_action_steps", 8))
    image_keys = list(
        config.get(
            "image_keys",
            ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
        )
    )
    state_key = config.get("state_key", "observation.state")
    image_size = int(config.get("image_size", 96))
    max_episodes = config.get("max_episodes", 70)
    num_diffusion_steps = int(config.get("num_diffusion_steps", 100))
    num_inference_steps = int(config.get("num_inference_steps", 32))

    use_handle_pos = bool(config.get("use_handle_pos", False))
    dataset = LerobotVisionChunkDataset(
        dataset_path=dataset_path,
        image_keys=image_keys,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        image_size=image_size,
        state_key=state_key,
        max_episodes=max_episodes,
        use_handle_pos=use_handle_pos,
    )
    val_fraction = float(config.get("val_fraction", 0.1))
    seed = int(config.get("seed", 42))
    val_count = int(len(dataset) * val_fraction)
    if val_count <= 0 and len(dataset) > 1:
        val_count = 1
    if val_count >= len(dataset):
        val_count = max(1, len(dataset) - 1)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(dataset))
    val_indices = perm[:val_count]
    train_indices = perm[val_count:]
    train_dataset = Subset(dataset, train_indices.tolist())
    val_dataset = Subset(dataset, val_indices.tolist())

    num_workers = int(config.get("num_workers", 0))
    allow_mp_dataloader = bool(config.get("allow_multiprocess_dataloader", False))
    is_spawn_platform = sys.platform == "darwin" or sys.platform.startswith("win")
    if is_spawn_platform and num_workers > 0 and not allow_mp_dataloader:
        platform_name = "Windows" if sys.platform.startswith("win") else "macOS"
        print(
            f"{platform_name} stability guard: forcing num_workers=0. "
            "Set allow_multiprocess_dataloader=true to override."
        )
        num_workers = 0

    dataloader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=bool(num_workers > 0),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=int(config.get("val_batch_size", config["batch_size"])),
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    device = get_device(torch)
    print(
        f"Loaded {len(dataset)} sequence samples from {len(dataset.episodes)} episodes "
        f"(state_dim={dataset.state_dim}, action_dim={dataset.action_dim})"
    )
    print(f"Train samples: {len(train_dataset)}  Val samples: {len(val_dataset)}")
    print(f"Device: {device}")

    model = VisionDiffusionChunkPolicy(
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        image_keys=image_keys,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        num_diffusion_steps=num_diffusion_steps,
        vision_feature_dim=int(config.get("vision_feature_dim", 256)),
        hidden_dim=int(config.get("hidden_dim", 768)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 1e-6)),
    )
    scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_steps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    lr_warmup_steps = int(config.get("lr_warmup_steps", 500))
    warmup = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, float(step + 1) / float(max(1, lr_warmup_steps))),
    )
    max_train_steps = config.get("max_train_steps_per_epoch", None)
    log_every_steps = int(config.get("log_every_steps", 50))
    eval_every_epochs = int(config.get("eval_every_epochs", 10))
    val_max_batches = int(config.get("val_max_batches", 50))
    rollout_eval_rollouts = int(config.get("rollout_eval_rollouts", 5))
    rollout_eval_max_steps = int(config.get("rollout_eval_max_steps", 500))
    rollout_eval_every_epochs = int(config.get("rollout_eval_every_epochs", 20))
    enable_rollout_eval = bool(config.get("enable_rollout_eval", True))
    sweep_steps = config.get("rollout_inference_steps_list", [16, 32, 64])
    sweep_steps = sorted(set(int(x) for x in sweep_steps))
    if rollout_eval_every_epochs <= 0:
        enable_rollout_eval = False

    create_env = None
    if enable_rollout_eval:
        try:
            from robocasa.utils.env_utils import create_env as _create_env

            create_env = _create_env
        except ModuleNotFoundError:
            print(
                "WARNING: robocasa is unavailable; disabling rollout success evaluation. "
                "Install robocasa to log success_pretrain/target during training."
            )
            enable_rollout_eval = False

    checkpoint_dir = config.get("checkpoint_dir", "/tmp/cabinet_policy_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    epoch_metrics_path = os.path.join(checkpoint_dir, "training_metrics.csv")
    sweep_metrics_path = os.path.join(checkpoint_dir, "inference_sweep_metrics.csv")
    step_metrics_path = os.path.join(checkpoint_dir, "training_steps.csv")
    with open(epoch_metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "global_step",
                "train_denoise_loss",
                "val_denoise_loss",
                "val_action_mse",
                "success_pretrain",
                "success_target",
                "base_inference_steps",
            ]
        )
    with open(sweep_metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "split",
                "inference_steps",
                "success_rate",
                "num_rollouts",
                "max_steps",
            ]
        )
    with open(step_metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "step_in_epoch",
                "global_step",
                "running_train_loss",
            ]
        )
    best_loss = float("inf")
    global_step = 0
    state_mean_t = torch.from_numpy(dataset.state_mean).to(device).view(1, 1, -1)
    state_std_t = torch.from_numpy(dataset.state_std).to(device).view(1, 1, -1)
    action_mean_t = torch.from_numpy(dataset.action_mean).to(device).view(1, 1, -1)
    action_std_t = torch.from_numpy(dataset.action_std).to(device).view(1, 1, -1)

    def preprocess_image_for_model(img):
        img_t = (
            torch.from_numpy(img.astype(np.float32))
            .permute(2, 0, 1)
            .unsqueeze(0)
            / 255.0
        )
        if img_t.shape[-1] != image_size or img_t.shape[-2] != image_size:
            img_t = torch.nn.functional.interpolate(
                img_t,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            )
        return img_t.squeeze(0).numpy()

    ROBOCASA_STATE_KEYS = [
        "robot0_base_pos",
        "robot0_base_quat",
        "robot0_base_to_eef_pos",
        "robot0_base_to_eef_quat",
        "robot0_gripper_qpos",
    ]

    def _quat_to_rot(quat):
        w, x, y, z = quat
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ])

    def extract_state(obs, env=None):
        parts = []
        for key in ROBOCASA_STATE_KEYS:
            if key in obs:
                parts.append(obs[key].flatten())
        if not parts:
            return np.zeros(dataset.state_dim, dtype=np.float32)
        state = np.concatenate(parts).astype(np.float32)
        if use_handle_pos and env is not None:
            handle_base = _get_handle_base_pos(obs, env)
            state = np.concatenate([state, handle_base])
        if len(state) < dataset.state_dim:
            state = np.pad(state, (0, dataset.state_dim - len(state)))
        elif len(state) > dataset.state_dim:
            state = state[: dataset.state_dim]
        return state

    def _collect_handle_sites(env, fxtr_name):
        sites = [
            n for n in env.sim.model.site_names
            if "handle" in n.lower() and fxtr_name in n
        ]
        if not sites:
            sites = [n for n in env.sim.model.site_names if "handle" in n.lower()]
        return sites

    def _collect_handle_bodies(env, fxtr_name):
        bodies = [
            n for n in env.sim.model.body_names
            if "handle" in n.lower() and fxtr_name in n
        ]
        if not bodies:
            bodies = [n for n in env.sim.model.body_names if "handle" in n.lower()]
        return bodies

    def _collect_handle_geoms(env):
        return [n for n in env.sim.model.geom_names if "handle" in n.lower()]

    def _select_nearest_handle(positions, eef_world):
        if eef_world is None or not positions:
            return positions[0] if positions else None
        dists = [np.linalg.norm(p - eef_world) for p in positions]
        return positions[int(np.argmin(dists))]

    def _get_handle_world_pos(env, eef_world=None):
        fxtr = env.fxtr
        fxtr_name = fxtr.name if hasattr(fxtr, "name") else ""
        sites = _collect_handle_sites(env, fxtr_name)
        if sites:
            positions = []
            for name in sites:
                site_id = env.sim.model.site_name2id(name)
                positions.append(env.sim.data.site_xpos[site_id].copy())
            selected = _select_nearest_handle(positions, eef_world)
            if selected is not None:
                return selected

        bodies = _collect_handle_bodies(env, fxtr_name)
        if bodies:
            positions = []
            for name in bodies:
                body_id = env.sim.model.body_name2id(name)
                positions.append(env.sim.data.body_xpos[body_id].copy())
            selected = _select_nearest_handle(positions, eef_world)
            if selected is not None:
                return selected

        geoms = _collect_handle_geoms(env)
        if geoms:
            positions = []
            for name in geoms:
                geom_id = env.sim.model.geom_name2id(name)
                positions.append(env.sim.data.geom_xpos[geom_id].copy())
            selected = _select_nearest_handle(positions, eef_world)
            if selected is not None:
                return selected

        return np.zeros(3, dtype=np.float32)

    def _get_handle_base_pos(obs, env):
        base_pos = obs["robot0_base_pos"]
        base_quat = obs["robot0_base_quat"]
        R = _quat_to_rot(base_quat)
        eef_rel = obs["robot0_base_to_eef_pos"].flatten()
        eef_world = base_pos + R @ eef_rel
        handle_world = _get_handle_world_pos(env, eef_world=eef_world)
        return (R.T @ (handle_world - base_pos)).astype(np.float32)

    def remap_action_dataset_to_env(action, gripper_threshold=0.0, base_mode_threshold=0.0):
        env_action = np.zeros_like(action)
        gripper = 1.0 if float(action[11]) > gripper_threshold else -1.0
        base_mode = 1.0 if float(action[4]) > base_mode_threshold else -1.0
        env_action[0:3] = action[5:8]
        env_action[3:6] = action[8:11]
        env_action[6] = gripper
        env_action[7:10] = action[0:3]
        env_action[10] = action[3]
        env_action[11] = base_mode
        return env_action

    def check_any_door_open(env, threshold_rad=0.3):
        try:
            fxtr = env.fxtr
            fxtr_name = fxtr.name if hasattr(fxtr, "name") else ""
            joint_ids = [
                i for i, n in enumerate(env.sim.model.joint_names)
                if fxtr_name in n and "door" in n.lower()
            ]
            if joint_ids:
                for jidx in joint_ids:
                    addr = env.sim.model.joint(jidx).qposadr[0]
                    qpos = env.sim.data.qpos[addr]
                    if abs(float(qpos)) > threshold_rad:
                        return True
                return False
            door_state = env.fxtr.get_door_state(env)
            norm_thresh = threshold_rad / (np.pi / 2)
            return any(v >= norm_thresh for v in door_state.values())
        except Exception:
            return env._check_success()

    def compute_validation_metrics():
        model.eval()
        denoise_losses = []
        action_mses = []
        with torch.no_grad():
            for b_idx, batch in enumerate(val_dataloader):
                obs_dict = {"state": batch["state"].to(device)}
                for cam in image_keys:
                    obs_dict[cam] = batch[cam].to(device)
                gt_action_norm = batch["action"].to(device)
                noise = torch.randn_like(gt_action_norm)
                timesteps = torch.randint(
                    0, num_diffusion_steps, (gt_action_norm.shape[0],), device=device
                ).long()
                noisy = scheduler.add_noise(gt_action_norm, noise, timesteps)
                pred_noise = model(noisy, timesteps, obs_dict)
                denoise_losses.append(float(F.mse_loss(pred_noise, noise).item()))

                pred_action_norm = model.sample_action_chunk(
                    obs_dict, num_inference_steps=num_inference_steps
                )
                pred_action = pred_action_norm * action_std_t + action_mean_t
                gt_action = gt_action_norm * action_std_t + action_mean_t
                action_mses.append(float(F.mse_loss(pred_action, gt_action).item()))
                if b_idx + 1 >= val_max_batches:
                    break
        model.train()
        return (
            float(np.mean(denoise_losses)) if denoise_losses else float("nan"),
            float(np.mean(action_mses)) if action_mses else float("nan"),
        )

    def evaluate_success_rate(split, inference_steps):
        if create_env is None:
            return float("nan")

        model.eval()
        successes = 0
        for ep in range(rollout_eval_rollouts):
            env = create_env(
                env_name="OpenCabinet",
                render_onscreen=False,
                seed=seed + ep,
                split=split,
                camera_widths=256,
                camera_heights=256,
            )
            obs = env.reset()
            state_hist = []
            img_hist = {k: [] for k in image_keys}
            action_queue = []
            ok = False
            for _ in range(rollout_eval_max_steps):
                state = extract_state(obs, env=env)
                state_hist.append(state.astype(np.float32))
                if len(state_hist) > n_obs_steps:
                    state_hist.pop(0)
                for cam in image_keys:
                    img_hist[cam].append(
                        preprocess_image_for_model(obs[f"{cam}_image"]).astype(np.float32)
                    )
                    if len(img_hist[cam]) > n_obs_steps:
                        img_hist[cam].pop(0)
                while len(state_hist) < n_obs_steps:
                    state_hist.insert(0, state_hist[0].copy())
                    for cam in image_keys:
                        img_hist[cam].insert(0, img_hist[cam][0].copy())

                with torch.no_grad():
                    if not action_queue:
                        state_np = np.stack(state_hist, axis=0)[None]
                        obs_dict = {
                            "state": (
                                torch.from_numpy(state_np).to(device) - state_mean_t
                            )
                            / state_std_t
                        }
                        for cam in image_keys:
                            obs_dict[cam] = torch.from_numpy(
                                np.stack(img_hist[cam], axis=0)[None]
                            ).to(device)
                        action_norm = model.sample_action_chunk(
                            obs_dict, num_inference_steps=int(inference_steps)
                        )
                        action_chunk = (
                            action_norm * action_std_t + action_mean_t
                        ).cpu().numpy().squeeze(0)
                        action_queue.extend([a for a in action_chunk])
                action = action_queue.pop(0)
                action = remap_action_dataset_to_env(action)
                env_action_dim = env.action_dim
                if len(action) < env_action_dim:
                    action = np.pad(action, (0, env_action_dim - len(action)))
                elif len(action) > env_action_dim:
                    action = action[:env_action_dim]
                obs, _, _, _ = env.step(action)
                if check_any_door_open(env):
                    ok = True
                    break
            successes += int(ok)
            env.close()
        model.train()
        return successes / max(1, rollout_eval_rollouts)

    print_section("Training")
    print(
        f"Epochs={config['epochs']}  Batch={config['batch_size']}  LR={config['learning_rate']}  "
        f"n_obs={n_obs_steps}  n_action={n_action_steps}  diff_steps={num_diffusion_steps}"
    )
    if log_every_steps > 0:
        print(f"Live logs every {log_every_steps} steps")

    for epoch in range(int(config["epochs"])):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            obs_dict = {"state": batch["state"].to(device)}
            for cam in image_keys:
                obs_dict[cam] = batch[cam].to(device)
            action_chunk = batch["action"].to(device)
            noise = torch.randn_like(action_chunk)
            timesteps = torch.randint(
                0, num_diffusion_steps, (action_chunk.shape[0],), device=device
            ).long()
            noisy = scheduler.add_noise(action_chunk, noise, timesteps)
            pred_noise = model(noisy, timesteps, obs_dict)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            warmup.step()

            epoch_loss += float(loss.item())
            num_batches += 1
            global_step += 1
            if log_every_steps > 0 and (num_batches % log_every_steps) == 0:
                running_loss = epoch_loss / max(1, num_batches)
                print(
                    f"    epoch {epoch + 1:4d} step {num_batches:5d} "
                    f"global_step {global_step:7d} loss {running_loss:.6f}",
                    flush=True,
                )
                with open(step_metrics_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            epoch + 1,
                            num_batches,
                            global_step,
                            float(running_loss),
                        ]
                    )
            if max_train_steps is not None and num_batches >= int(max_train_steps):
                break

        avg_loss = epoch_loss / max(1, num_batches)
        print(f"  Epoch {epoch + 1:4d}/{config['epochs']}  Train loss: {avg_loss:.6f}")

        val_denoise_loss = float("nan")
        val_action_mse = float("nan")
        success_pretrain = float("nan")
        success_target = float("nan")
        if ((epoch + 1) % eval_every_epochs) == 0 or epoch == 0:
            val_denoise_loss, val_action_mse = compute_validation_metrics()
            print(
                f"    Val denoise loss: {val_denoise_loss:.6f}  "
                f"Val action MSE: {val_action_mse:.6f}"
            )
        if enable_rollout_eval and (
            ((epoch + 1) % rollout_eval_every_epochs) == 0 or epoch == 0
        ):
            success_pretrain = evaluate_success_rate("pretrain", num_inference_steps)
            success_target = evaluate_success_rate("target", num_inference_steps)
            print(
                f"    Success@{num_inference_steps} pretrain={100*success_pretrain:.1f}% "
                f"target={100*success_target:.1f}%"
            )
            for steps in sweep_steps:
                for split_name in ("pretrain", "target"):
                    sr = evaluate_success_rate(split_name, steps)
                    with open(sweep_metrics_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                epoch + 1,
                                split_name,
                                int(steps),
                                float(sr),
                                rollout_eval_rollouts,
                                rollout_eval_max_steps,
                            ]
                        )

        with open(epoch_metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1,
                    global_step,
                    float(avg_loss),
                    float(val_denoise_loss),
                    float(val_action_mse),
                    float(success_pretrain),
                    float(success_target),
                    int(num_inference_steps),
                ]
            )

        should_checkpoint = ((epoch + 1) % int(config.get("checkpoint_every", 10))) == 0
        if avg_loss < best_loss:
            best_loss = avg_loss
            should_checkpoint = True
            best_path = os.path.join(checkpoint_dir, "best_policy.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "loss": best_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_type": "vision_diffusion_chunk",
                    "state_dim": dataset.state_dim,
                    "action_dim": dataset.action_dim,
                    "image_keys": image_keys,
                    "state_key": state_key,
                    "n_obs_steps": n_obs_steps,
                    "n_action_steps": n_action_steps,
                    "num_diffusion_steps": num_diffusion_steps,
                    "num_inference_steps": num_inference_steps,
                    "vision_feature_dim": int(config.get("vision_feature_dim", 256)),
                    "hidden_dim": int(config.get("hidden_dim", 768)),
                    "image_size": image_size,
                    "use_handle_pos": use_handle_pos,
                    "state_mean": dataset.state_mean.astype(np.float32),
                    "state_std": dataset.state_std.astype(np.float32),
                    "action_mean": dataset.action_mean.astype(np.float32),
                    "action_std": dataset.action_std.astype(np.float32),
                },
                best_path,
            )
        if should_checkpoint:
            torch.save(
                {
                    "epoch": epoch,
                    "loss": avg_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_type": "vision_diffusion_chunk",
                    "state_dim": dataset.state_dim,
                    "action_dim": dataset.action_dim,
                    "image_keys": image_keys,
                    "state_key": state_key,
                    "n_obs_steps": n_obs_steps,
                    "n_action_steps": n_action_steps,
                    "num_diffusion_steps": num_diffusion_steps,
                    "num_inference_steps": num_inference_steps,
                    "vision_feature_dim": int(config.get("vision_feature_dim", 256)),
                    "hidden_dim": int(config.get("hidden_dim", 768)),
                    "image_size": image_size,
                    "use_handle_pos": use_handle_pos,
                    "state_mean": dataset.state_mean.astype(np.float32),
                    "state_std": dataset.state_std.astype(np.float32),
                    "action_mean": dataset.action_mean.astype(np.float32),
                    "action_std": dataset.action_std.astype(np.float32),
                },
                os.path.join(checkpoint_dir, f"epoch_{epoch + 1:04d}.pt"),
            )

    final_path = os.path.join(checkpoint_dir, "final_policy.pt")
    torch.save(
        {
            "epoch": int(config["epochs"]),
            "loss": avg_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_type": "vision_diffusion_chunk",
            "state_dim": dataset.state_dim,
            "action_dim": dataset.action_dim,
            "image_keys": image_keys,
            "state_key": state_key,
            "n_obs_steps": n_obs_steps,
            "n_action_steps": n_action_steps,
            "num_diffusion_steps": num_diffusion_steps,
            "num_inference_steps": num_inference_steps,
            "vision_feature_dim": int(config.get("vision_feature_dim", 256)),
            "hidden_dim": int(config.get("hidden_dim", 768)),
            "image_size": image_size,
            "use_handle_pos": use_handle_pos,
            "state_mean": dataset.state_mean.astype(np.float32),
            "state_std": dataset.state_std.astype(np.float32),
            "action_mean": dataset.action_mean.astype(np.float32),
            "action_std": dataset.action_std.astype(np.float32),
        },
        final_path,
    )
    print("\nTraining complete!")
    print(f"Best checkpoint:  {os.path.join(checkpoint_dir, 'best_policy.pt')}")
    print(f"Final checkpoint: {final_path}")


def train_diffusion_policy(config):
    """
    Train a low-dimensional diffusion policy that predicts one action from state.
    """
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install with: pip install torch diffusers")
        sys.exit(1)

    print_section("Diffusion Policy (Starter Code)")
    dataset_path = get_dataset_path(config.get("dataset_path", None))
    print(f"Dataset: {dataset_path}")

    class CabinetDemoDataset(Dataset):
        def __init__(self, dataset_path, max_episodes=None):
            import pyarrow.parquet as pq

            self.states = []
            self.actions = []
            data_dir = os.path.join(dataset_path, "data")
            if not os.path.exists(data_dir):
                data_dir = os.path.join(dataset_path, "lerobot", "data")
            if not os.path.exists(data_dir):
                raise FileNotFoundError(
                    f"Data directory not found under: {dataset_path}\n"
                    "Make sure you downloaded the dataset with 04_download_dataset.py"
                )
            chunk_dir = os.path.join(data_dir, "chunk-000")
            parquet_files = sorted(
                f for f in os.listdir(chunk_dir) if f.endswith(".parquet")
            )
            episodes_loaded = 0
            for pf in parquet_files:
                table = pq.read_table(os.path.join(chunk_dir, pf))
                df = table.to_pandas()
                state_cols = [c for c in df.columns if c.startswith("observation.state")]
                action_cols = [c for c in df.columns if c == "action" or c.startswith("action.")]
                if not state_cols or not action_cols:
                    state_cols = [
                        c for c in df.columns if "gripper" in c or "base" in c or "eef" in c
                    ]
                    action_cols = [c for c in df.columns if "action" in c]
                if state_cols and action_cols:
                    for _, row in df.iterrows():
                        state_parts = []
                        action_parts = []
                        for c in state_cols:
                            val = row[c]
                            if isinstance(val, np.ndarray):
                                state_parts.extend(val.flatten().tolist())
                            elif isinstance(val, (int, float, np.floating)):
                                state_parts.append(float(val))
                        for c in action_cols:
                            val = row[c]
                            if isinstance(val, np.ndarray):
                                action_parts.extend(val.flatten().tolist())
                            elif isinstance(val, (int, float, np.floating)):
                                action_parts.append(float(val))
                        if state_parts and action_parts:
                            self.states.append(np.array(state_parts, dtype=np.float32))
                            self.actions.append(np.array(action_parts, dtype=np.float32))
                episodes_loaded += 1
                if max_episodes and episodes_loaded >= max_episodes:
                    break
            if len(self.states) == 0:
                raise RuntimeError("No state-action pairs extracted from dataset.")
            self.states = np.array(self.states, dtype=np.float32)
            self.actions = np.array(self.actions, dtype=np.float32)

        def __len__(self):
            return len(self.states)

        def __getitem__(self, idx):
            return self.states[idx], self.actions[idx]

    dataset = CabinetDemoDataset(
        dataset_path, max_episodes=config.get("max_episodes", 50)
    )
    state_dim = dataset.states.shape[-1]
    action_dim = dataset.actions.shape[-1]
    print(f"Loaded {len(dataset)} samples  state_dim={state_dim}  action_dim={action_dim}")

    state_mean = dataset.states.mean(axis=0, keepdims=True)
    state_std = dataset.states.std(axis=0, keepdims=True) + 1e-6
    action_mean = dataset.actions.mean(axis=0, keepdims=True)
    action_std = dataset.actions.std(axis=0, keepdims=True) + 1e-6

    norm_states = (dataset.states - state_mean) / state_std
    norm_actions = (dataset.actions - action_mean) / action_std

    class NormalizedDataset(Dataset):
        def __init__(self, states, actions):
            self.states = states.astype(np.float32)
            self.actions = actions.astype(np.float32)

        def __len__(self):
            return len(self.states)

        def __getitem__(self, idx):
            return (
                torch.from_numpy(self.states[idx]),
                torch.from_numpy(self.actions[idx]),
            )

    dataloader = DataLoader(
        NormalizedDataset(norm_states, norm_actions),
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_diffusion_steps = config.get("num_diffusion_steps", 100)
    num_inference_steps = config.get("num_inference_steps", 20)
    model = DiffusionActionMLP(
        state_dim=state_dim,
        action_dim=action_dim,
        num_diffusion_steps=num_diffusion_steps,
        hidden_dim=config.get("hidden_dim", 512),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 1e-6),
    )
    scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_steps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )

    print_section("Training")
    print(f"Epochs:              {config['epochs']}")
    print(f"Batch size:          {config['batch_size']}")
    print(f"LR:                  {config['learning_rate']}")
    print(f"Diffusion steps:     {num_diffusion_steps}")
    print(f"Inference steps:     {num_inference_steps}")

    checkpoint_dir = config.get("checkpoint_dir", "/tmp/cabinet_policy_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_loss = float("inf")
    avg_loss = float("inf")

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for states_batch, actions_batch in dataloader:
            states_batch = states_batch.to(device)
            actions_batch = actions_batch.to(device)
            noise = torch.randn_like(actions_batch)
            timesteps = torch.randint(
                0, num_diffusion_steps, (actions_batch.shape[0],), device=device
            ).long()
            noisy_actions = scheduler.add_noise(actions_batch, noise, timesteps)
            pred_noise = model(noisy_actions, timesteps, states_batch)
            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        avg_loss = epoch_loss / max(num_batches, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:4d}/{config['epochs']}  Train loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(checkpoint_dir, "best_policy.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "model_type": "diffusion_mlp",
                    "num_diffusion_steps": num_diffusion_steps,
                    "num_inference_steps": num_inference_steps,
                    "hidden_dim": config.get("hidden_dim", 512),
                    "state_mean": state_mean.astype(np.float32),
                    "state_std": state_std.astype(np.float32),
                    "action_mean": action_mean.astype(np.float32),
                    "action_std": action_std.astype(np.float32),
                },
                ckpt_path,
            )

    final_path = os.path.join(checkpoint_dir, "final_policy.pt")
    torch.save(
        {
            "epoch": config["epochs"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "model_type": "diffusion_mlp",
            "num_diffusion_steps": num_diffusion_steps,
            "num_inference_steps": num_inference_steps,
            "hidden_dim": config.get("hidden_dim", 512),
            "state_mean": state_mean.astype(np.float32),
            "state_std": state_std.astype(np.float32),
            "action_mean": action_mean.astype(np.float32),
            "action_std": action_std.astype(np.float32),
        },
        final_path,
    )

    print("\nTraining complete!")
    print(f"Best loss:        {best_loss:.6f}")
    print(f"Best checkpoint:  {os.path.join(checkpoint_dir, 'best_policy.pt')}")
    print(f"Final checkpoint: {final_path}")


def print_diffusion_policy_instructions():
    """Print instructions for using the official Diffusion Policy repo."""
    print_section("Official Diffusion Policy Training")
    print(
        "For production-quality policy training, use the official repos:\n"
        "\n"
        "Option A: Diffusion Policy (recommended for single-task)\n"
        "  git clone https://github.com/robocasa-benchmark/diffusion_policy\n"
        "  cd diffusion_policy && pip install -e .\n"
        "\n"
        "  # Train\n"
        "  python train.py \\\n"
        "    --config-name=train_diffusion_transformer_bs192 \\\n"
        "    task=robocasa/OpenCabinet\n"
        "\n"
        "  # Evaluate\n"
        "  python eval_robocasa.py \\\n"
        "    --checkpoint <path-to-checkpoint> \\\n"
        "    --task_set atomic \\\n"
        "    --split target\n"
        "\n"
        "Option B: pi-0 via OpenPi (for foundation model fine-tuning)\n"
        "  git clone https://github.com/robocasa-benchmark/openpi\n"
        "  cd openpi && pip install -e . && pip install -e packages/openpi-client/\n"
        "\n"
        "  XLA_PYTHON_CLIENT_MEM_FRACTION=1.0 python scripts/train.py \\\n"
        "    robocasa_OpenCabinet --exp-name=cabinet_door\n"
        "\n"
        "Option C: GR00T N1.5 (NVIDIA foundation model)\n"
        "  git clone https://github.com/robocasa-benchmark/Isaac-GR00T\n"
        "  cd groot && pip install -e .\n"
        "\n"
        "  python scripts/gr00t_finetune.py \\\n"
        "    --output-dir experiments/cabinet_door \\\n"
        "    --dataset_soup robocasa_OpenCabinet \\\n"
        "    --max_steps 50000\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Train a policy for OpenCabinet")
    parser.add_argument(
        "--policy",
        type=str,
        default="vision_diffusion_chunk",
        choices=["vision_diffusion_chunk", "diffusion", "simple"],
        help="Policy type to train",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="vision_diffusion_chunk",
        choices=["vision_diffusion_chunk", "diffusion", "simple"],
        help="Policy type to train",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dim for diffusion MLP",
    )
    parser.add_argument(
        "--num_diffusion_steps",
        type=int,
        default=100,
        help="DDPM training steps",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=32,
        help="DDPM sampling steps at evaluation",
    )
    parser.add_argument("--horizon", type=int, default=10, help="Sequence horizon")
    parser.add_argument(
        "--n_obs_steps", type=int, default=2, help="Observation history steps"
    )
    parser.add_argument(
        "--n_action_steps", type=int, default=8, help="Predicted action chunk length"
    )
    parser.add_argument(
        "--vision_feature_dim", type=int, default=256, help="Per-image feature size"
    )
    parser.add_argument("--image_size", type=int, default=96, help="Training image size")
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Data loader workers"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-6, help="AdamW weight decay"
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Linear warmup steps"
    )
    parser.add_argument(
        "--checkpoint_every", type=int, default=10, help="Save every N epochs"
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=50,
        help="Max parquet episodes to load (None loads all)",
    )
    parser.add_argument(
        "--log_every_steps", 
        type=int, 
        default=50, 
        help="Log every N steps"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dim for diffusion MLP",
    )
    parser.add_argument(
        "--num_diffusion_steps",
        type=int,
        default=100,
        help="DDPM training steps",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=32,
        help="DDPM sampling steps at evaluation",
    )
    parser.add_argument("--horizon", type=int, default=10, help="Sequence horizon")
    parser.add_argument(
        "--n_obs_steps", type=int, default=2, help="Observation history steps"
    )
    parser.add_argument(
        "--n_action_steps", type=int, default=8, help="Predicted action chunk length"
    )
    parser.add_argument(
        "--vision_feature_dim", type=int, default=256, help="Per-image feature size"
    )
    parser.add_argument("--image_size", type=int, default=96, help="Training image size")
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Data loader workers"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-6, help="AdamW weight decay"
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Linear warmup steps"
    )
    parser.add_argument(
        "--checkpoint_every", type=int, default=10, help="Save every N epochs"
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=50,
        help="Max parquet episodes to load (None loads all)",
    )
    parser.add_argument(
        "--log_every_steps", 
        type=int, 
        default=50, 
        help="Log every N steps"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/tmp/cabinet_policy_checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides other args)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Optional path to LeRobot OpenCabinet dataset root",
    )
    parser.add_argument(
        "--use_handle_pos",
        action="store_true",
        help="Augment state with cached handle positions (run 09_cache_handle_positions.py first)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenCabinet - Policy Training")
    print("=" * 60)

    # Build config from args or YAML file
    if args.config:
        config = load_config(args.config)
        # CLI flags explicitly provided by user should override YAML values.
        # This keeps `--config ... --checkpoint_dir ...` behavior intuitive.
        cli_overrides = {
            "policy": args.policy,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "checkpoint_dir": args.checkpoint_dir,
            "hidden_dim": args.hidden_dim,
            "num_diffusion_steps": args.num_diffusion_steps,
            "num_inference_steps": args.num_inference_steps,
            "max_episodes": args.max_episodes,
            "horizon": args.horizon,
            "n_obs_steps": args.n_obs_steps,
            "n_action_steps": args.n_action_steps,
            "vision_feature_dim": args.vision_feature_dim,
            "image_size": args.image_size,
            "num_workers": args.num_workers,
            "weight_decay": args.weight_decay,
            "lr_warmup_steps": args.lr_warmup_steps,
            "checkpoint_every": args.checkpoint_every,
            "log_every_steps": args.log_every_steps,
            "dataset_path": args.dataset_path,
            "use_handle_pos": args.use_handle_pos,
        }
        for key, value in cli_overrides.items():
            if f"--{key}" in sys.argv:
                config[key] = value
    else:
        config = {
            "policy": args.policy,
            "policy": args.policy,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "checkpoint_dir": args.checkpoint_dir,
            "hidden_dim": args.hidden_dim,
            "num_diffusion_steps": args.num_diffusion_steps,
            "num_inference_steps": args.num_inference_steps,
            "max_episodes": args.max_episodes,
            "horizon": args.horizon,
            "n_obs_steps": args.n_obs_steps,
            "n_action_steps": args.n_action_steps,
            "vision_feature_dim": args.vision_feature_dim,
            "image_size": args.image_size,
            "num_workers": args.num_workers,
            "weight_decay": args.weight_decay,
            "lr_warmup_steps": args.lr_warmup_steps,
            "checkpoint_every": args.checkpoint_every,
            "log_every_steps": args.log_every_steps,
            "dataset_path": args.dataset_path,
            "use_handle_pos": args.use_handle_pos,
        }

    policy_type = config.get("policy", "vision_diffusion_chunk")
    if policy_type == "simple":
        train_simple_policy(config)
    elif policy_type == "vision_diffusion_chunk":
        train_vision_diffusion_chunk_policy(config)
    elif policy_type == "diffusion":
        train_diffusion_policy(config)
    else:
        print(f"ERROR: Unknown policy type: {policy_type}")
        sys.exit(1)


if __name__ == "__main__":
    main()
