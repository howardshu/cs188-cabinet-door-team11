"""
Step 7: Evaluate a Trained Policy
===================================
Runs a trained policy in the OpenCabinet environment and reports
success rate across multiple episodes and kitchen scenes.

Usage:
    # Evaluate the simple BC policy from Step 6
    python 07_evaluate_policy.py --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt

    # Evaluate with more episodes
    python 07_evaluate_policy.py --checkpoint path/to/policy.pt --num_rollouts 50

    # Evaluate on target (held-out) kitchen scenes
    python 07_evaluate_policy.py --checkpoint path/to/policy.pt --split target

    # Save evaluation videos
    python 07_evaluate_policy.py --checkpoint path/to/policy.pt --video_path /tmp/eval_videos.mp4

For evaluating official Diffusion Policy / pi-0 / GR00T checkpoints,
use the evaluation scripts from those repos instead (see 06_train_policy.py).
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Force osmesa (CPU offscreen renderer) on Linux/WSL2 -- EGL requires
# /dev/dri device access that is unavailable in WSL environments.
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np


def ensure_local_dependency_paths():
    repo_root = Path(__file__).resolve().parents[1]
    for dep in ("robocasa", "robosuite"):
        dep_path = repo_root / dep
        if dep_path.exists():
            dep_str = str(dep_path)
            if dep_str not in sys.path:
                sys.path.insert(0, dep_str)


ensure_local_dependency_paths()
import robocasa  # noqa: F401
from robocasa.utils.env_utils import create_env
from policy_models import DiffusionActionMLP, SimplePolicy, VisionDiffusionChunkPolicy


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def load_policy(checkpoint_path, device, forced_model_type="auto"):
    """Load a trained policy checkpoint."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dim = checkpoint["state_dim"]
    action_dim = checkpoint["action_dim"]
    ckpt_model_type = checkpoint.get("model_type", "simple_mlp")
    model_type = ckpt_model_type if forced_model_type == "auto" else forced_model_type
    valid_types = {"simple_mlp", "diffusion_mlp", "vision_diffusion_chunk"}
    if model_type not in valid_types:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Supported: {sorted(valid_types)}"
        )

    if model_type == "vision_diffusion_chunk":
        model = VisionDiffusionChunkPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            image_keys=checkpoint.get(
                "image_keys",
                ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
            ),
            n_obs_steps=checkpoint.get("n_obs_steps", 2),
            n_action_steps=checkpoint.get("n_action_steps", 8),
            num_diffusion_steps=checkpoint.get("num_diffusion_steps", 100),
            vision_feature_dim=checkpoint.get("vision_feature_dim", 256),
            hidden_dim=checkpoint.get("hidden_dim", 768),
        ).to(device)
    elif model_type == "diffusion_mlp":
        model = DiffusionActionMLP(
            state_dim=state_dim,
            action_dim=action_dim,
            num_diffusion_steps=checkpoint.get("num_diffusion_steps", 100),
            hidden_dim=checkpoint.get("hidden_dim", 512),
        ).to(device)
    else:
        model = SimplePolicy(state_dim, action_dim).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except Exception as e:
        raise RuntimeError(
            f"Failed to load checkpoint weights with model_type='{model_type}'. "
            f"Checkpoint model_type='{ckpt_model_type}'. "
            "Try --model_type auto or match the checkpoint type explicitly."
        ) from e
    model.eval()

    print(f"Loaded policy from: {checkpoint_path}")
    print(f"  Trained for {checkpoint['epoch']} epochs, loss={checkpoint['loss']:.6f}")
    print(
        f"  State dim: {state_dim}, Action dim: {action_dim}, "
        f"Type: {model_type} (ckpt: {ckpt_model_type})"
    )

    return model, state_dim, action_dim, checkpoint


ROBOCASA_STATE_KEYS = [
    "robot0_base_pos",
    "robot0_base_quat",
    "robot0_base_to_eef_pos",
    "robot0_base_to_eef_quat",
    "robot0_gripper_qpos",
]


def extract_state(obs, state_dim):
    """Extract a fixed-size state vector from observations.

    Uses the exact key ordering from the RoboCasa LeRobot dataset's
    modality.json so that the vector matches what the policy was trained on.
    """
    parts = []
    for key in ROBOCASA_STATE_KEYS:
        if key in obs:
            parts.append(obs[key].flatten())

    if not parts:
        return np.zeros(state_dim, dtype=np.float32)

    state = np.concatenate(parts).astype(np.float32)

    if len(state) < state_dim:
        state = np.pad(state, (0, state_dim - len(state)))
    elif len(state) > state_dim:
        state = state[:state_dim]

    return state


def remap_action_dataset_to_env(action):
    """Remap a 12-dim action from dataset ordering to environment ordering.

    Dataset (modality.json):  [base_motion(4), control_mode(1), eef_pos(3), eef_rot(3), gripper(1)]
    Environment (composite controller order): [eef_pos(3), eef_rot(3), torso(1), base_fwd/side/yaw(3), gripper(1), control_mode(1)]
    """
    env_action = np.zeros_like(action)
    env_action[0:3] = action[5:8]    # eef_position
    env_action[3:6] = action[8:11]   # eef_rotation
    env_action[6] = action[3]        # torso (dataset base_motion[3])
    env_action[7:10] = action[0:3]   # base_motion (fwd, side, yaw)
    env_action[10] = action[11]      # gripper
    env_action[11] = action[4]       # control_mode
    return env_action


def preprocess_image_for_model(img, image_size):
    import torch

    img_t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
    if img_t.shape[-1] != image_size or img_t.shape[-2] != image_size:
        img_t = torch.nn.functional.interpolate(
            img_t, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
    return img_t.squeeze(0).numpy()


def run_evaluation(
    model,
    state_dim,
    action_dim,
    checkpoint,
    num_rollouts,
    max_steps,
    split,
    video_path,
    seed,
):
    """Run evaluation rollouts and collect statistics."""
    import torch
    import imageio

    device = next(model.parameters()).device
    model_type = checkpoint.get("model_type", "simple_mlp")
    state_mean = state_std = action_mean = action_std = None
    image_keys = checkpoint.get("image_keys", [])
    if "state_mean" in checkpoint:
        state_mean = torch.as_tensor(checkpoint["state_mean"], device=device).squeeze(0)
        state_std = torch.as_tensor(checkpoint["state_std"], device=device).squeeze(0)
        action_mean = torch.as_tensor(checkpoint["action_mean"], device=device).squeeze(0)
        action_std = torch.as_tensor(checkpoint["action_std"], device=device).squeeze(0)

    env = create_env(
        env_name="OpenCabinet",
        render_onscreen=False,
        seed=seed,
        split=split,
        camera_widths=256,
        camera_heights=256,
    )

    video_writer = None
    if video_path:
        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        video_writer = imageio.get_writer(video_path, fps=20)

    results = {
        "successes": [],
        "episode_lengths": [],
        "rewards": [],
    }

    for ep in range(num_rollouts):
        obs = env.reset()
        ep_meta = env.get_ep_meta()
        lang = ep_meta.get("lang", "")

        ep_reward = 0.0
        success = False
        action_queue = []
        state_hist = []
        img_hist = {k: [] for k in image_keys}
        n_obs_steps = int(checkpoint.get("n_obs_steps", 2))
        image_size = int(checkpoint.get("image_size", 96))

        for step in range(max_steps):
            # Extract observation and predict action/chunk
            state = extract_state(obs, state_dim)
            if model_type == "vision_diffusion_chunk":
                state_hist.append(state.astype(np.float32))
                if len(state_hist) > n_obs_steps:
                    state_hist.pop(0)
                for cam in image_keys:
                    obs_key = f"{cam}_image"
                    img_hist[cam].append(
                        preprocess_image_for_model(obs[obs_key], image_size).astype(
                            np.float32
                        )
                    )
                    if len(img_hist[cam]) > n_obs_steps:
                        img_hist[cam].pop(0)
                while len(state_hist) < n_obs_steps:
                    state_hist.insert(0, state_hist[0].copy())
                    for cam in image_keys:
                        img_hist[cam].insert(0, img_hist[cam][0].copy())

            with torch.no_grad():
                if model_type == "vision_diffusion_chunk":
                    if not action_queue:
                        state_np = np.stack(state_hist, axis=0)[None]
                        state_t = torch.from_numpy(state_np).to(device)
                        state_t = (state_t - state_mean.view(1, 1, -1)) / state_std.view(
                            1, 1, -1
                        )
                        obs_dict = {"state": state_t}
                        for cam in image_keys:
                            img_np = np.stack(img_hist[cam], axis=0)[None]
                            obs_dict[cam] = torch.from_numpy(img_np).to(device)
                        action_norm = model.sample_action_chunk(
                            obs_dict,
                            num_inference_steps=int(
                                checkpoint.get("num_inference_steps", 32)
                            ),
                        )
                        action_chunk = (
                            action_norm * action_std.view(1, 1, -1)
                            + action_mean.view(1, 1, -1)
                        ).cpu().numpy().squeeze(0)
                        action_queue.extend([a for a in action_chunk])
                    action = action_queue.pop(0)
                elif model_type == "diffusion_mlp":
                    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
                    state_norm = (state_tensor - state_mean) / state_std
                    action_norm = model.sample_actions(
                        state_norm,
                        num_inference_steps=checkpoint.get("num_inference_steps", 20),
                    )
                    action = (
                        action_norm * action_std.unsqueeze(0) + action_mean.unsqueeze(0)
                    ).cpu().numpy().squeeze(0)
                else:
                    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
                    action = model(state_tensor).cpu().numpy().squeeze(0)

            # Remap from dataset action order to environment action order
            action = remap_action_dataset_to_env(action)

            # Pad action to match environment action dim if needed
            env_action_dim = env.action_dim
            if len(action) < env_action_dim:
                action = np.pad(action, (0, env_action_dim - len(action)))
            elif len(action) > env_action_dim:
                action = action[:env_action_dim]

            obs, reward, done, info = env.step(action)
            ep_reward += reward

            if video_writer is not None:
                frame = env.sim.render(
                    height=512, width=768, camera_name="robot0_agentview_center"
                )[::-1]
                video_writer.append_data(frame)

            if env._check_success():
                success = True
                break

        results["successes"].append(success)
        results["episode_lengths"].append(step + 1)
        results["rewards"].append(ep_reward)

        status = "SUCCESS" if success else "FAIL"
        print(
            f"  Episode {ep + 1:3d}/{num_rollouts}: {status:7s} "
            f"(steps={step + 1:4d}, reward={ep_reward:.1f}) "
            f'layout={env.layout_id}, style={env.style_id}, task="{lang}"'
        )

    if video_writer:
        video_writer.close()

    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained OpenCabinet policy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to policy checkpoint (.pt file)",
    )
    parser.add_argument(
        "--num_rollouts", type=int, default=20, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--max_steps", type=int, default=500, help="Max steps per episode"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="pretrain",
        choices=["pretrain", "target"],
        help="Kitchen scene split to evaluate on",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to save evaluation video (optional)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "simple_mlp", "diffusion_mlp", "vision_diffusion_chunk"],
        help="Model type override. Use auto to read from checkpoint metadata.",
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    print("=" * 60)
    print("  OpenCabinet - Policy Evaluation")
    print("=" * 60)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load the trained policy
    model, state_dim, action_dim, checkpoint = load_policy(
        args.checkpoint, device, forced_model_type=args.model_type
    )

    # Run evaluation
    print_section(f"Evaluating on {args.split} split ({args.num_rollouts} episodes)")

    results = run_evaluation(
        model=model,
        state_dim=state_dim,
        action_dim=action_dim,
        checkpoint=checkpoint,
        num_rollouts=args.num_rollouts,
        max_steps=args.max_steps,
        split=args.split,
        video_path=args.video_path,
        seed=args.seed,
    )

    # Print summary
    print_section("Evaluation Results")

    num_success = sum(results["successes"])
    success_rate = num_success / args.num_rollouts * 100
    avg_length = np.mean(results["episode_lengths"])
    avg_reward = np.mean(results["rewards"])

    print(f"  Split:          {args.split}")
    print(f"  Episodes:       {args.num_rollouts}")
    print(f"  Successes:      {num_success}/{args.num_rollouts}")
    print(f"  Success rate:   {success_rate:.1f}%")
    print(f"  Avg ep length:  {avg_length:.1f} steps")
    print(f"  Avg reward:     {avg_reward:.3f}")

    if args.video_path:
        print(f"\n  Video saved to: {args.video_path}")

    # Context for expected performance
    print_section("Performance Context")
    print(
        "Expected success rates from the RoboCasa benchmark:\n"
        "\n"
        "  Method            | Pretrain | Target\n"
        "  ------------------|----------|-------\n"
        "  Random actions    |    ~0%   |   ~0%\n"
        "  Diffusion Policy  |  ~30-60% | ~20-50%\n"
        "  pi-0              |  ~40-70% | ~30-60%\n"
        "  GR00T N1.5        |  ~35-65% | ~25-55%\n"
        "\n"
        "Note: The simple MLP policy from Step 6 is not expected to\n"
        "achieve meaningful success rates. Use the official Diffusion\n"
        "Policy repo for real results."
    )


if __name__ == "__main__":
    main()
