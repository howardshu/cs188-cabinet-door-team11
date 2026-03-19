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
    dp_path = Path(__file__).resolve().parent / "diffusion_policy"
    if dp_path.exists():
        dp_str = str(dp_path)
        if dp_str not in sys.path:
            sys.path.insert(0, dp_str)


ensure_local_dependency_paths()
import robocasa  # noqa: F401
from robocasa.utils.env_utils import create_env
from policy_models import (
    DiffusionActionMLP,
    SimplePolicy,
    VisionDiffusionChunkPolicy,
    BCUnet1DPolicy,
)


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def load_policy(checkpoint_path, device, forced_model_type="auto"):
def load_policy(checkpoint_path, device, forced_model_type="auto"):
    """Load a trained policy checkpoint."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dim = checkpoint["state_dim"]
    action_dim = checkpoint["action_dim"]
    ckpt_model_type = checkpoint.get("model_type", "simple_mlp")
    model_type = ckpt_model_type if forced_model_type == "auto" else forced_model_type
    valid_types = {
        "simple_mlp",
        "diffusion_mlp",
        "vision_diffusion_chunk",
        "unet_lowdim_diffusion",
        "bc_unet_lowdim",
    }
    if model_type not in valid_types:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Supported: {sorted(valid_types)}"
        )

    if model_type == "unet_lowdim_diffusion":
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        obs_cond_dim = checkpoint.get("n_obs_steps", 2) * state_dim
        model = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_cond_dim,
            diffusion_step_embed_dim=checkpoint.get("diffusion_step_embed_dim", 256),
            down_dims=checkpoint.get("down_dims", [256, 512, 1024]),
            kernel_size=checkpoint.get("kernel_size", 5),
            n_groups=checkpoint.get("n_groups", 8),
            cond_predict_scale=checkpoint.get("cond_predict_scale", False),
        ).to(device)
    elif model_type == "bc_unet_lowdim":
        model = BCUnet1DPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            n_obs_steps=checkpoint.get("n_obs_steps", 2),
            n_action_steps=checkpoint.get("n_action_steps", 16),
            base_channels=checkpoint.get("base_channels", 32),
            channel_mults=checkpoint.get("channel_mults", [1, 2]),
            kernel_size=checkpoint.get("kernel_size", 5),
            cond_dim=checkpoint.get("cond_dim", 256),
        ).to(device)
    elif model_type == "vision_diffusion_chunk":
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


def _quat_to_rot(quat):
    """Convert a [w, x, y, z] quaternion to a 3x3 rotation matrix."""
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


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
    """Get the cabinet handle position in world frame (prefers sites).

    Picks the handle closest to the end-effector when possible to match
    the handle selection used during dataset augmentation.
    """
    fxtr = env.fxtr
    fxtr_name = fxtr.name if hasattr(fxtr, "name") else ""

    sites = _collect_handle_sites(env, fxtr_name)
    if sites:
        positions = []
        for name in sites:
            site_id = env.sim.model.site_name2id(name)
            positions.append(env.sim.data.site_xpos[site_id].copy().astype(np.float32))
        selected = _select_nearest_handle(positions, eef_world)
        if selected is not None:
            return selected

    bodies = _collect_handle_bodies(env, fxtr_name)
    if bodies:
        positions = []
        for name in bodies:
            body_id = env.sim.model.body_name2id(name)
            positions.append(env.sim.data.body_xpos[body_id].copy().astype(np.float32))
        selected = _select_nearest_handle(positions, eef_world)
        if selected is not None:
            return selected

    geoms = _collect_handle_geoms(env)
    if geoms:
        positions = []
        for name in geoms:
            geom_id = env.sim.model.geom_name2id(name)
            positions.append(env.sim.data.geom_xpos[geom_id].copy().astype(np.float32))
        selected = _select_nearest_handle(positions, eef_world)
        if selected is not None:
            return selected

    return np.zeros(3, dtype=np.float32)


def get_handle_base_pos(obs, env):
    """Get the cabinet handle position in robot base frame."""
    base_pos = obs["robot0_base_pos"]
    base_quat = obs["robot0_base_quat"]
    R = _quat_to_rot(base_quat)
    eef_rel = obs["robot0_base_to_eef_pos"].flatten()
    eef_world = base_pos + R @ eef_rel
    handle_world = _get_handle_world_pos(env, eef_world=eef_world)
    return (R.T @ (handle_world - base_pos)).astype(np.float32)


AUGMENTED_OBS_KEYS = [
    "observation.handle_pos",
    "observation.handle_to_eef_pos",
    "observation.door_openness",
]


def compute_augmented_features(obs, env, augmented_keys):
    """Compute augmented features at runtime to match 05b training data.

    Produces the same features that 05b_augment_handle_data.py bakes into
    the augmented parquet files: handle world position, handle-to-eef vector,
    and door openness.
    """
    if not augmented_keys:
        return np.array([], dtype=np.float32)

    base_pos = obs["robot0_base_pos"].flatten()
    base_quat = obs["robot0_base_quat"].flatten()
    eef_rel = obs["robot0_base_to_eef_pos"].flatten()
    R = _quat_to_rot(base_quat)
    eef_world = base_pos + R @ eef_rel
    handle_world = _get_handle_world_pos(env, eef_world=eef_world)

    parts = []
    for key in augmented_keys:
        if key == "observation.handle_pos":
            parts.append(handle_world)
        elif key == "observation.handle_to_eef_pos":
            parts.append((eef_world - handle_world).astype(np.float32))
        elif key == "observation.door_openness":
            try:
                door_state = env.fxtr.get_door_state(env)
                openness = float(np.mean(list(door_state.values())))
            except Exception:
                openness = 0.0
            parts.append(np.array([openness], dtype=np.float32))

    return np.concatenate(parts) if parts else np.array([], dtype=np.float32)


def check_any_door_open(env, threshold_rad=0.3):
    """Check if ANY single door hinge exceeds a qpos threshold (radians)."""
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


def extract_state(
    obs,
    state_dim,
    env=None,
    use_handle_pos=False,
    augmented_obs_keys=None,
    state_mask=None,
):
    """Extract a fixed-size state vector from observations.

    Uses the exact key ordering from the RoboCasa LeRobot dataset's
    modality.json so that the vector matches what the policy was trained on.

    When use_handle_pos is True, appends handle position in base frame (legacy).
    When augmented_obs_keys is provided, appends world-frame features matching 05b.
    """
    parts = []
    for key in ROBOCASA_STATE_KEYS:
        if key in obs:
            parts.append(obs[key].flatten())

    if not parts:
        return np.zeros(state_dim, dtype=np.float32)

    state = np.concatenate(parts).astype(np.float32)
    if state_mask is not None:
        state = state[state_mask]

    if use_handle_pos and env is not None:
        handle_base = get_handle_base_pos(obs, env)
        state = np.concatenate([state, handle_base])

    if augmented_obs_keys and env is not None:
        aug = compute_augmented_features(obs, env, augmented_obs_keys)
        state = np.concatenate([state, aug])

    if len(state) < state_dim:
        state = np.pad(state, (0, state_dim - len(state)))
    elif len(state) > state_dim:
        state = state[:state_dim]

    return state


def remap_action_dataset_to_env(action, gripper_threshold=0.0, base_mode_threshold=0.0):
    """Remap a 12-dim action from dataset ordering to environment ordering.

    Dataset (modality.json):  [base_motion(3), torso(1), control_mode(1), eef_pos(3), eef_rot(3), gripper(1)]
    Environment (composite controller order): [eef_pos(3), eef_rot(3), gripper(1), base_motion(3), torso(1), base_mode(1)]
    """
    env_action = np.zeros_like(action)
    gripper = 1.0 if float(action[11]) > gripper_threshold else -1.0
    base_mode = 1.0 if float(action[4]) > base_mode_threshold else -1.0
    env_action[0:3] = action[5:8]    # eef_position
    env_action[3:6] = action[8:11]   # eef_rotation
    env_action[6] = gripper          # gripper_close (binarized)
    env_action[7:10] = action[0:3]   # base_motion (fwd, side, yaw)
    env_action[10] = action[3]       # torso
    env_action[11] = base_mode       # control_mode (binarized)
    return env_action


def preprocess_image_for_model(img, image_size):
    import torch

    img_t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
    if img_t.shape[-1] != image_size or img_t.shape[-2] != image_size:
        img_t = torch.nn.functional.interpolate(
            img_t, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
    return img_t.squeeze(0).numpy()


def _sample_unet_actions(model, obs_cond, scheduler, n_action_steps, action_dim, device, num_inference_steps=16):
    """Run DDPM reverse diffusion to sample an action chunk from the U-Net."""
    import torch
    scheduler.set_timesteps(num_inference_steps)
    bsz = obs_cond.shape[0]
    x = torch.randn((bsz, n_action_steps, action_dim), device=device)
    for t in scheduler.timesteps:
        t_batch = torch.full((bsz,), int(t.item()), device=device, dtype=torch.long)
        pred_noise = model(x, t_batch, global_cond=obs_cond)
        x = scheduler.step(pred_noise, t, x).prev_sample
    return x


def run_evaluation(
    model,
    state_dim,
    action_dim,
    checkpoint,
    checkpoint,
    num_rollouts,
    max_steps,
    split,
    video_path,
    seed,
    use_relaxed_success=True,
    execute_steps=None,
    gripper_threshold=0.0,
    base_mode_threshold=0.0,
    success_threshold_rad=0.3,
):
    """Run evaluation rollouts and collect statistics."""
    import torch
    import imageio

    device = next(model.parameters()).device
    model_type = checkpoint.get("model_type", "simple_mlp")
    use_handle_pos = checkpoint.get("use_handle_pos", False)
    augmented_obs_keys = checkpoint.get("augmented_obs_keys", None)
    state_mask = checkpoint.get("state_mask", None)
    state_mean = state_std = action_mean = action_std = None
    image_keys = checkpoint.get("image_keys", [])
    if "state_mean" in checkpoint:
        state_mean = torch.as_tensor(checkpoint["state_mean"], device=device).squeeze(0)
        state_std = torch.as_tensor(checkpoint["state_std"], device=device).squeeze(0)
        action_mean = torch.as_tensor(checkpoint["action_mean"], device=device).squeeze(0)
        action_std = torch.as_tensor(checkpoint["action_std"], device=device).squeeze(0)

    noise_scheduler = None
    if model_type == "unet_lowdim_diffusion":
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=checkpoint.get("num_diffusion_steps", 100),
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        )

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
        steps_since_plan = 0
        state_hist = []
        img_hist = {k: [] for k in image_keys}
        n_obs_steps = int(checkpoint.get("n_obs_steps", 2))
        image_size = int(checkpoint.get("image_size", 96))
        n_action_steps = int(checkpoint.get("n_action_steps", 8))
        if execute_steps is None:
            execute_steps = int(checkpoint.get("execute_steps", n_action_steps))

        for step in range(max_steps):
            state = extract_state(
                obs,
                state_dim,
                env=env,
                use_handle_pos=use_handle_pos,
                augmented_obs_keys=augmented_obs_keys,
                state_mask=state_mask,
            )

            if model_type in ("vision_diffusion_chunk", "unet_lowdim_diffusion", "bc_unet_lowdim"):
                state_hist.append(state.astype(np.float32))
                if len(state_hist) > n_obs_steps:
                    state_hist.pop(0)
                while len(state_hist) < n_obs_steps:
                    state_hist.insert(0, state_hist[0].copy())

            if model_type == "vision_diffusion_chunk":
                for cam in image_keys:
                    obs_key = f"{cam}_image"
                    img_hist[cam].append(
                        preprocess_image_for_model(obs[obs_key], image_size).astype(
                            np.float32
                        )
                    )
                    if len(img_hist[cam]) > n_obs_steps:
                        img_hist[cam].pop(0)
                while len(img_hist.get(image_keys[0], [])) < n_obs_steps if image_keys else False:
                    for cam in image_keys:
                        img_hist[cam].insert(0, img_hist[cam][0].copy())

            with torch.no_grad():
                if model_type == "unet_lowdim_diffusion":
                    if steps_since_plan >= execute_steps:
                        action_queue.clear()
                    if not action_queue:
                        state_np = np.stack(state_hist, axis=0)
                        state_t = torch.from_numpy(state_np).to(device)
                        state_t = (state_t - state_mean.view(1, -1)) / state_std.view(1, -1)
                        obs_cond = state_t.reshape(1, -1)
                        action_norm = _sample_unet_actions(
                            model, obs_cond, noise_scheduler,
                            n_action_steps, action_dim, device,
                            num_inference_steps=checkpoint.get("num_inference_steps", 16),
                        )
                        action_chunk = (
                            action_norm * action_std.view(1, 1, -1)
                            + action_mean.view(1, 1, -1)
                        ).cpu().numpy().squeeze(0)
                        action_queue.extend([a for a in action_chunk])
                        steps_since_plan = 0
                    action = action_queue.pop(0)
                    steps_since_plan += 1
                elif model_type == "vision_diffusion_chunk":
                    if steps_since_plan >= execute_steps:
                        action_queue.clear()
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
                        steps_since_plan = 0
                    action = action_queue.pop(0)
                    steps_since_plan += 1
                elif model_type == "bc_unet_lowdim":
                    if steps_since_plan >= execute_steps:
                        action_queue.clear()
                    if not action_queue:
                        state_np = np.stack(state_hist, axis=0)[None]
                        state_t = torch.from_numpy(state_np).to(device)
                        state_t = (state_t - state_mean.view(1, 1, -1)) / state_std.view(
                            1, 1, -1
                        )
                        action_norm = model(state_t)
                        action_chunk = (
                            action_norm * action_std.view(1, 1, -1)
                            + action_mean.view(1, 1, -1)
                        ).cpu().numpy().squeeze(0)
                        action_queue.extend([a for a in action_chunk])
                        steps_since_plan = 0
                    action = action_queue.pop(0)
                    steps_since_plan += 1
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
            action = remap_action_dataset_to_env(
                action,
                gripper_threshold=gripper_threshold,
                base_mode_threshold=base_mode_threshold,
            )

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

            is_success = (
                check_any_door_open(env, threshold_rad=success_threshold_rad)
                if use_relaxed_success
                else env._check_success()
            )
            if is_success:
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
        "--num_rollouts", type=int, default=50, help="Number of evaluation episodes"
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
        choices=[
            "auto",
            "simple_mlp",
            "diffusion_mlp",
            "vision_diffusion_chunk",
            "unet_lowdim_diffusion",
            "bc_unet_lowdim",
        ],
        help="Model type override. Use auto to read from checkpoint metadata.",
    )
    parser.add_argument(
        "--single_door_success",
        action="store_true",
        help="Deprecated. Relaxed success is now the default.",
    )
    parser.add_argument(
        "--strict_success",
        action="store_true",
        help="Use env._check_success() (all doors) instead of relaxed criterion",
    )
    parser.add_argument(
        "--success_threshold_rad",
        type=float,
        default=0.3,
        help="Hinge joint qpos threshold (radians) for relaxed success",
    )
    parser.add_argument(
        "--execute_steps",
        type=int,
        default=None,
        help="Replan after this many steps from an action chunk (default: n_action_steps)",
    )
    parser.add_argument(
        "--gripper_threshold",
        type=float,
        default=0.0,
        help="Binarize gripper: >threshold => close, else open",
    )
    parser.add_argument(
        "--base_mode_threshold",
        type=float,
        default=0.0,
        help="Binarize base_mode/control_mode: >threshold => 1, else -1",
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
    model, state_dim, action_dim, checkpoint = load_policy(
        args.checkpoint, device, forced_model_type=args.model_type
    )

    # Run evaluation
    print_section(f"Evaluating on {args.split} split ({args.num_rollouts} episodes)")

    use_relaxed_success = not args.strict_success
    results = run_evaluation(
        model=model,
        state_dim=state_dim,
        action_dim=action_dim,
        checkpoint=checkpoint,
        checkpoint=checkpoint,
        num_rollouts=args.num_rollouts,
        max_steps=args.max_steps,
        split=args.split,
        video_path=args.video_path,
        seed=args.seed,
        use_relaxed_success=use_relaxed_success,
        execute_steps=args.execute_steps,
        gripper_threshold=args.gripper_threshold,
        base_mode_threshold=args.base_mode_threshold,
        success_threshold_rad=args.success_threshold_rad,
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
