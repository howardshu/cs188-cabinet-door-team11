"""
Step 8: Visualize a Policy Rollout
=====================================
Loads a trained policy checkpoint from 06_train_policy.py and runs it
live in the OpenCabinet environment so you can watch the robot.

This is your primary debugging tool: watch exactly where and why the policy
fails — does it reach for the handle? Does it grasp? Does it pull correctly?

Two rendering modes:
  On-screen  (default)  — interactive MuJoCo viewer window, real-time
  Off-screen (--offscreen) — renders to a video file, works without a display

Usage:
    # Watch live in a window (WSL/Linux) + save video
    python 08_visualize_policy_rollout.py --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt

    # Save to video only (no display needed — works headless / in notebooks)
    python 08_visualize_policy_rollout.py --checkpoint ... --offscreen

    # Run 3 episodes, slow down playback so you can follow along
    python 08_visualize_policy_rollout.py --checkpoint ... --num_episodes 3 --max_steps 200

    # Mac users must use mjpython for the on-screen window
    mjpython 08_visualize_policy_rollout.py --checkpoint ...
"""

import os
import sys
from pathlib import Path

# ── Rendering mode detection ────────────────────────────────────────────────
# We peek at sys.argv *before* argparse so we can configure the GL backend
# before any library is imported.  Wrong GL backend = gladLoadGL error.
_OFFSCREEN = "--offscreen" in sys.argv

if _OFFSCREEN:
    # Off-screen mode: use Mesa's software osmesa renderer.
    # EGL is the default on headless Linux but fails on WSL2 (no /dev/dri).
    if sys.platform == "linux":
        os.environ.setdefault("MUJOCO_GL", "osmesa")
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
else:
    # On-screen mode: re-exec with correct display vars baked into the OS
    # environment so Mesa (GLFW) sees them before any C library initializes.
    # On WSLg the .bashrc often sets a stale VcXsrv-style DISPLAY that
    # breaks GLFW; os.execve() restarts the process cleanly.
    if sys.platform == "linux" and "__TELEOP_DISPLAY_OK" not in os.environ:
        _env = dict(os.environ)
        _changed = False
        if _env.get("WAYLAND_DISPLAY"):
            if not _env.get("DISPLAY", "").startswith(":"):
                _env["DISPLAY"] = ":0"
                _changed = True
            if _env.get("GALLIUM_DRIVER") != "llvmpipe":
                _env["GALLIUM_DRIVER"] = "llvmpipe"
                _changed = True
            if _env.get("MESA_GL_VERSION_OVERRIDE") != "4.5":
                _env["MESA_GL_VERSION_OVERRIDE"] = "4.5"
                _changed = True
        if _changed:
            _env["__TELEOP_DISPLAY_OK"] = "1"
            os.execve(sys.executable, [sys.executable] + sys.argv, _env)
        else:
            os.environ["__TELEOP_DISPLAY_OK"] = "1"
# ────────────────────────────────────────────────────────────────────────────

import argparse
import time

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
import robocasa  # noqa: F401 — registers OpenCabinet environment
import robosuite
from policy_models import (
    DiffusionActionMLP,
    SimplePolicy,
    VisionDiffusionChunkPolicy,
    BCUnet1DPolicy,
)
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper


# ── Policy loading (identical to 07_evaluate_policy.py) ─────────────────────

def load_policy(checkpoint_path, device):
    """Load a trained policy checkpoint."""
    import torch

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dim = ckpt["state_dim"]
    action_dim = ckpt["action_dim"]
    model_type = ckpt.get("model_type", "simple_mlp")

    if model_type == "unet_lowdim_diffusion":
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        obs_cond_dim = ckpt.get("n_obs_steps", 2) * state_dim
        model = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_cond_dim,
            diffusion_step_embed_dim=ckpt.get("diffusion_step_embed_dim", 256),
            down_dims=ckpt.get("down_dims", [256, 512, 1024]),
            kernel_size=ckpt.get("kernel_size", 5),
            n_groups=ckpt.get("n_groups", 8),
            cond_predict_scale=ckpt.get("cond_predict_scale", False),
        ).to(device)
    elif model_type == "bc_unet_lowdim":
        model = BCUnet1DPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            n_obs_steps=ckpt.get("n_obs_steps", 2),
            n_action_steps=ckpt.get("n_action_steps", 16),
            base_channels=ckpt.get("base_channels", 32),
            channel_mults=ckpt.get("channel_mults", [1, 2]),
            kernel_size=ckpt.get("kernel_size", 5),
            cond_dim=ckpt.get("cond_dim", 256),
        ).to(device)
    elif model_type == "vision_diffusion_chunk":
        model = VisionDiffusionChunkPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            image_keys=ckpt.get(
                "image_keys",
                ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
            ),
            n_obs_steps=ckpt.get("n_obs_steps", 2),
            n_action_steps=ckpt.get("n_action_steps", 8),
            num_diffusion_steps=ckpt.get("num_diffusion_steps", 100),
            vision_feature_dim=ckpt.get("vision_feature_dim", 256),
            hidden_dim=ckpt.get("hidden_dim", 768),
        ).to(device)
    elif model_type == "diffusion_mlp":
        model = DiffusionActionMLP(
            state_dim=state_dim,
            action_dim=action_dim,
            num_diffusion_steps=ckpt.get("num_diffusion_steps", 100),
            hidden_dim=ckpt.get("hidden_dim", 512),
        ).to(device)
    else:
        model = SimplePolicy(state_dim, action_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, state_dim, action_dim, ckpt


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


def _find_handle_site(env):
    fxtr = env.fxtr
    fxtr_name = fxtr.name if hasattr(fxtr, "name") else ""
    candidates = [
        n for n in env.sim.model.site_names
        if "handle" in n.lower() and fxtr_name in n
    ]
    if not candidates:
        candidates = [n for n in env.sim.model.site_names if "handle" in n.lower()]
    return candidates[0] if candidates else None


def _get_handle_world_pos(env):
    """Get the cabinet handle position in world frame (prefers sites)."""
    site = _find_handle_site(env)
    if site is not None:
        site_id = env.sim.model.site_name2id(site)
        return env.sim.data.site_xpos[site_id].copy().astype(np.float32)

    fxtr = env.fxtr
    fxtr_name = fxtr.name if hasattr(fxtr, "name") else ""
    for name in env.sim.model.body_names:
        if fxtr_name in name and "handle" in name:
            body_id = env.sim.model.body_name2id(name)
            return env.sim.data.body_xpos[body_id].copy().astype(np.float32)
    for name in env.sim.model.body_names:
        if "handle" in name.lower():
            body_id = env.sim.model.body_name2id(name)
            return env.sim.data.body_xpos[body_id].copy().astype(np.float32)

    candidates = [n for n in env.sim.model.geom_names if "handle" in n.lower()]
    if candidates:
        geom_id = env.sim.model.geom_name2id(candidates[0])
        return env.sim.data.geom_xpos[geom_id].copy().astype(np.float32)
    return np.zeros(3, dtype=np.float32)


def get_handle_base_pos(obs, env):
    """Get the cabinet handle position in robot base frame."""
    handle_world = _get_handle_world_pos(env)
    base_pos = obs["robot0_base_pos"]
    base_quat = obs["robot0_base_quat"]
    R = _quat_to_rot(base_quat)
    return (R.T @ (handle_world - base_pos)).astype(np.float32)


AUGMENTED_OBS_KEYS = [
    "observation.handle_pos",
    "observation.handle_to_eef_pos",
    "observation.door_openness",
]


def compute_augmented_features(obs, env, augmented_keys):
    """Compute augmented features at runtime to match 05b training data."""
    if not augmented_keys:
        return np.array([], dtype=np.float32)

    handle_world = _get_handle_world_pos(env)

    base_pos = obs["robot0_base_pos"].flatten()
    base_quat = obs["robot0_base_quat"].flatten()
    eef_rel = obs["robot0_base_to_eef_pos"].flatten()
    R = _quat_to_rot(base_quat)
    eef_world = base_pos + R @ eef_rel

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


def extract_state(obs, state_dim, env=None, use_handle_pos=False, augmented_obs_keys=None):
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

    Dataset (modality.json):  [base_motion(4), control_mode(1), eef_pos(3), eef_rot(3), gripper(1)]
    Environment (composite controller order): [eef_pos(3), eef_rot(3), gripper(1), base_motion(4), control_mode(1)]
    """
    env_action = np.zeros_like(action)
    gripper = 1.0 if float(action[11]) > gripper_threshold else -1.0
    base_mode = 1.0 if float(action[4]) > base_mode_threshold else -1.0
    env_action[0:3] = action[5:8]    # eef_position
    env_action[3:6] = action[8:11]   # eef_rotation
    env_action[6] = gripper          # gripper_close (binarized)
    env_action[7:11] = action[0:4]   # base_motion (fwd, side, yaw, torso)
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


# ── On-screen rollout ────────────────────────────────────────────────────────

def run_onscreen(model, state_dim, action_dim, args):
    """
    Run the policy with an interactive MuJoCo viewer window.

    The viewer opens automatically; you can pan/zoom/rotate the camera
    with the mouse while the robot executes the policy.
    """
    import torch

    device = next(model.parameters()).device
    model_type = args.ckpt.get("model_type", "simple_mlp")
    use_handle_pos = args.ckpt.get("use_handle_pos", False)
    augmented_obs_keys = args.ckpt.get("augmented_obs_keys", None)
    use_relaxed_success = not getattr(args, "strict_success", False)
    state_mean = state_std = action_mean = action_std = None
    image_keys = args.ckpt.get("image_keys", [])
    n_obs_steps = int(args.ckpt.get("n_obs_steps", 2))
    n_action_steps = int(args.ckpt.get("n_action_steps", 8))
    execute_steps = (
        args.execute_steps
        if args.execute_steps is not None
        else int(args.ckpt.get("execute_steps", n_action_steps))
    )
    image_size = int(args.ckpt.get("image_size", 96))
    if "state_mean" in args.ckpt:
        state_mean = torch.as_tensor(args.ckpt["state_mean"], device=device).squeeze(0)
        state_std = torch.as_tensor(args.ckpt["state_std"], device=device).squeeze(0)
        action_mean = torch.as_tensor(args.ckpt["action_mean"], device=device).squeeze(0)
        action_std = torch.as_tensor(args.ckpt["action_std"], device=device).squeeze(0)

    noise_scheduler = None
    if model_type == "unet_lowdim_diffusion":
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ckpt.get("num_diffusion_steps", 100),
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        )

    env = robosuite.make(
        env_name="OpenCabinet",
        robots="PandaOmron",
        controller_configs=load_composite_controller_config(robot="PandaOmron"),
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="robot0_frontview",
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer="mjviewer",
    )
    env = VisualizationWrapper(env)

    successes = 0
    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep + 1}/{args.num_episodes} ---")
        obs = env.reset()
        ep_meta = env.get_ep_meta()
        lang = ep_meta.get("lang", "")
        print(f"  Task:    {lang}")
        print(f"  Layout:  {env.layout_id}   Style: {env.style_id}")
        print(f"  Running for up to {args.max_steps} steps...")
        print(f"  (Watch the viewer window — use mouse to orbit the camera)\n")

        success = False
        hold_count = 0
        action_queue = []
        steps_since_plan = 0
        state_hist = []
        img_hist = {k: [] for k in image_keys}

        for step in range(args.max_steps):
            state = extract_state(
                obs, state_dim, env=env,
                use_handle_pos=use_handle_pos,
                augmented_obs_keys=augmented_obs_keys,
            )

            if model_type in ("vision_diffusion_chunk", "unet_lowdim_diffusion", "bc_unet_lowdim"):
                state_hist.append(state.astype(np.float32))
                if len(state_hist) > n_obs_steps:
                    state_hist.pop(0)
                while len(state_hist) < n_obs_steps:
                    state_hist.insert(0, state_hist[0].copy())

            if model_type == "vision_diffusion_chunk":
                for cam in image_keys:
                    img_hist[cam].append(
                        preprocess_image_for_model(
                            obs[f"{cam}_image"], image_size
                        ).astype(np.float32)
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
                            num_inference_steps=args.ckpt.get("num_inference_steps", 16),
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
                        obs_dict = {
                            "state": (
                                torch.from_numpy(state_np).to(device) - state_mean.view(1, 1, -1)
                            )
                            / state_std.view(1, 1, -1)
                        }
                        for cam in image_keys:
                            obs_dict[cam] = torch.from_numpy(
                                np.stack(img_hist[cam], axis=0)[None]
                            ).to(device)
                        action_norm = model.sample_action_chunk(
                            obs_dict,
                            num_inference_steps=args.ckpt.get("num_inference_steps", 32),
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
                        num_inference_steps=args.ckpt.get("num_inference_steps", 20),
                    )
                    action = (
                        action_norm * action_std.unsqueeze(0) + action_mean.unsqueeze(0)
                    ).cpu().numpy().squeeze(0)
                else:
                    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
                    action = model(state_tensor).cpu().numpy().squeeze(0)

            action = remap_action_dataset_to_env(
                action,
                gripper_threshold=args.gripper_threshold,
                base_mode_threshold=args.base_mode_threshold,
            )

            env_dim = env.action_dim
            if len(action) < env_dim:
                action = np.pad(action, (0, env_dim - len(action)))
            elif len(action) > env_dim:
                action = action[:env_dim]

            obs, reward, done, info = env.step(action)

            if step % 20 == 0:
                is_open = (
                    check_any_door_open(env, threshold_rad=args.success_threshold_rad)
                    if use_relaxed_success
                    else env._check_success()
                )
                status = "cabinet OPEN" if is_open else "in progress"
                act_mag = float(np.abs(action).mean())
                print(
                    f"  step {step:4d}  reward={reward:+.3f}  "
                    f"action_mag={act_mag:.3f}  [{status}]"
                )

            is_success = (
                check_any_door_open(env, threshold_rad=args.success_threshold_rad)
                if use_relaxed_success
                else env._check_success()
            )
            if is_success:
                hold_count += 1
                if hold_count >= 15:
                    success = True
                    break
            else:
                hold_count = 0

            time.sleep(1.0 / args.max_fr)

        result = "SUCCESS" if success else "did not open cabinet"
        print(f"\n  Result: {result}")
        if success:
            successes += 1

    env.close()
    print(f"\nFinal: {successes}/{args.num_episodes} episodes succeeded.")


# ── Off-screen rollout with video ────────────────────────────────────────────

def run_offscreen(model, state_dim, action_dim, args):
    """
    Run the policy headlessly and save a side-by-side annotated video.

    Each frame shows the robot from the front-view camera; per-step
    diagnostics (step count, reward, success flag) are printed to the
    terminal.
    """
    import torch
    import imageio
    from robocasa.utils.env_utils import create_env

    device = next(model.parameters()).device
    model_type = args.ckpt.get("model_type", "simple_mlp")
    use_handle_pos = args.ckpt.get("use_handle_pos", False)
    augmented_obs_keys = args.ckpt.get("augmented_obs_keys", None)
    use_relaxed_success = not getattr(args, "strict_success", False)
    state_mean = state_std = action_mean = action_std = None
    image_keys = args.ckpt.get("image_keys", [])
    n_obs_steps = int(args.ckpt.get("n_obs_steps", 2))
    n_action_steps = int(args.ckpt.get("n_action_steps", 8))
    execute_steps = (
        args.execute_steps
        if args.execute_steps is not None
        else int(args.ckpt.get("execute_steps", n_action_steps))
    )
    image_size = int(args.ckpt.get("image_size", 96))
    if "state_mean" in args.ckpt:
        state_mean = torch.as_tensor(args.ckpt["state_mean"], device=device).squeeze(0)
        state_std = torch.as_tensor(args.ckpt["state_std"], device=device).squeeze(0)
        action_mean = torch.as_tensor(args.ckpt["action_mean"], device=device).squeeze(0)
        action_std = torch.as_tensor(args.ckpt["action_std"], device=device).squeeze(0)

    noise_scheduler = None
    if model_type == "unet_lowdim_diffusion":
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ckpt.get("num_diffusion_steps", 100),
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        )

    video_dir = os.path.dirname(args.video_path)
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)

    cam_h, cam_w = 512, 768

    successes = 0
    all_frames = []

    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep + 1}/{args.num_episodes} ---")
        env = create_env(
            env_name="OpenCabinet",
            render_onscreen=False,
            seed=args.seed + ep,
            camera_widths=cam_w,
            camera_heights=cam_h,
        )
        obs = env.reset()
        ep_meta = env.get_ep_meta()
        lang = ep_meta.get("lang", "")
        print(f"  Task:    {lang}")
        print(f"  Layout:  {env.layout_id}   Style: {env.style_id}")

        success = False
        hold_count = 0
        ep_frames = []
        action_queue = []
        steps_since_plan = 0
        state_hist = []
        img_hist = {k: [] for k in image_keys}

        for step in range(args.max_steps):
            state = extract_state(
                obs, state_dim, env=env,
                use_handle_pos=use_handle_pos,
                augmented_obs_keys=augmented_obs_keys,
            )

            if model_type in ("vision_diffusion_chunk", "unet_lowdim_diffusion", "bc_unet_lowdim"):
                state_hist.append(state.astype(np.float32))
                if len(state_hist) > n_obs_steps:
                    state_hist.pop(0)
                while len(state_hist) < n_obs_steps:
                    state_hist.insert(0, state_hist[0].copy())

            if model_type == "vision_diffusion_chunk":
                for cam in image_keys:
                    img_hist[cam].append(
                        preprocess_image_for_model(
                            obs[f"{cam}_image"], image_size
                        ).astype(np.float32)
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
                            num_inference_steps=args.ckpt.get("num_inference_steps", 16),
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
                        obs_dict = {
                            "state": (
                                torch.from_numpy(state_np).to(device) - state_mean.view(1, 1, -1)
                            )
                            / state_std.view(1, 1, -1)
                        }
                        for cam in image_keys:
                            obs_dict[cam] = torch.from_numpy(
                                np.stack(img_hist[cam], axis=0)[None]
                            ).to(device)
                        action_norm = model.sample_action_chunk(
                            obs_dict,
                            num_inference_steps=args.ckpt.get("num_inference_steps", 32),
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
                        num_inference_steps=args.ckpt.get("num_inference_steps", 20),
                    )
                    action = (
                        action_norm * action_std.unsqueeze(0) + action_mean.unsqueeze(0)
                    ).cpu().numpy().squeeze(0)
                else:
                    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
                    action = model(state_tensor).cpu().numpy().squeeze(0)

            action = remap_action_dataset_to_env(
                action,
                gripper_threshold=args.gripper_threshold,
                base_mode_threshold=args.base_mode_threshold,
            )

            env_dim = env.action_dim
            if len(action) < env_dim:
                action = np.pad(action, (0, env_dim - len(action)))
            elif len(action) > env_dim:
                action = action[:env_dim]

            obs, reward, done, info = env.step(action)

            frame = env.sim.render(
                height=cam_h, width=cam_w, camera_name="robot0_agentview_center"
            )[::-1]
            ep_frames.append(frame)

            if step % 20 == 0:
                is_open = (
                    check_any_door_open(env, threshold_rad=args.success_threshold_rad)
                    if use_relaxed_success
                    else env._check_success()
                )
                status = "cabinet OPEN" if is_open else "in progress"
                print(
                    f"  step {step:4d}  reward={reward:+.3f}  [{status}]"
                )

            is_success = (
                check_any_door_open(env, threshold_rad=args.success_threshold_rad)
                if use_relaxed_success
                else env._check_success()
            )
            if is_success:
                hold_count += 1
                if hold_count >= 15:
                    success = True
                    break
            else:
                hold_count = 0

        result = "SUCCESS" if success else "did not open cabinet"
        print(f"  Result: {result}  ({len(ep_frames)} frames)")
        if success:
            successes += 1

        all_frames.extend(ep_frames)
        env.close()

    print(f"\nWriting {len(all_frames)} frames to {args.video_path} ...")
    with imageio.get_writer(args.video_path, fps=args.fps) as writer:
        for frame in all_frames:
            writer.append_data(frame)
    print(f"Video saved: {args.video_path}")

    print(f"\nFinal: {successes}/{args.num_episodes} episodes succeeded.")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a trained policy rollout in OpenCabinet"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/tmp/cabinet_policy_checkpoints/best_policy.pt",
        help="Path to policy checkpoint (.pt) saved by 06_train_policy.py",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=300,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="Render to video file instead of opening a viewer window",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="/tmp/policy_rollout.mp4",
        help="Output video path (used with --offscreen)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second for the saved video",
    )
    parser.add_argument(
        "--max_fr",
        type=int,
        default=20,
        help="On-screen playback rate cap (frames/second); lower = slower",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for environment layout/style selection",
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

    print("=" * 60)
    print("  OpenCabinet - Policy Rollout Visualizer")
    print("=" * 60)
    print()

    # Load policy
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required.  Run: pip install torch")
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("Train a policy first with:  python 06_train_policy.py")
        sys.exit(1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model, state_dim, action_dim, ckpt = load_policy(args.checkpoint, device)
    args.ckpt = ckpt

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  Epoch {ckpt['epoch']}, loss {ckpt['loss']:.6f}")
    print(f"  State dim: {state_dim},  Action dim: {action_dim}")
    print(f"  Device: {device}")
    print()

    mode = "off-screen (video)" if args.offscreen else "on-screen (viewer window)"
    print(f"Mode:     {mode}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Max steps/ep: {args.max_steps}")
    if args.offscreen:
        print(f"Output:   {args.video_path}")
    print()

    if args.offscreen:
        run_offscreen(model, state_dim, action_dim, args)
    else:
        print("Opening viewer window...")
        print("  Tip: orbit the camera with the mouse to see the gripper.\n")
        run_onscreen(model, state_dim, action_dim, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
