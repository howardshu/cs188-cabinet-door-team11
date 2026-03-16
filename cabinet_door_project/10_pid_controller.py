"""
Step 10: Waypoint PID Controller Baseline
============================================
A simple scripted policy that uses proportional control to:
  1. Approach the cabinet door handle
  2. Grasp the handle
  3. Pull the door open

This establishes a no-learning baseline and validates which
observations are necessary to solve the OpenCabinet task.

Usage:
    # On-screen (Mac: use mjpython)
    mjpython 10_pid_controller.py

    # Off-screen with video
    python 10_pid_controller.py --offscreen --video_path /tmp/pid_rollout.mp4

    # Run multiple episodes
    python 10_pid_controller.py --num_episodes 10 --offscreen
"""

import argparse
import os
import sys
import time
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


_OFFSCREEN = "--offscreen" in sys.argv

if _OFFSCREEN:
    if sys.platform == "linux":
        os.environ.setdefault("MUJOCO_GL", "osmesa")
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
else:
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

ensure_local_dependency_paths()

import robocasa  # noqa: F401
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper


def quat_to_rot_matrix(quat):
    """Convert a [w, x, y, z] quaternion to a 3x3 rotation matrix."""
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ])


def find_handle_geom_names(sim):
    """Find all geom names containing 'handle'."""
    return [n for n in sim.model.geom_names if "handle" in n.lower()]


def get_handle_world_pos(env):
    """Get the cabinet handle position in world frame."""
    fxtr = env.fxtr
    fxtr_name = fxtr.name if hasattr(fxtr, "name") else ""

    candidates = [
        n for n in env.sim.model.geom_names
        if "handle" in n.lower() and fxtr_name in n
    ]
    if not candidates:
        candidates = find_handle_geom_names(env.sim)
    if not candidates:
        return None

    geom_id = env.sim.model.geom_name2id(candidates[0])
    return env.sim.data.geom_xpos[geom_id].copy()


def get_eef_world_pos(env):
    """Get the end-effector position in world frame."""
    robot = env.robots[0]
    eef_site_name = robot.gripper.important_sites.get("grip_site", None)
    if eef_site_name:
        site_id = env.sim.model.site_name2id(eef_site_name)
        return env.sim.data.site_xpos[site_id].copy()
    return None


def get_eef_base_pos(obs):
    """Get the end-effector position in base frame from observations."""
    return obs["robot0_base_to_eef_pos"].copy()


def world_to_base_frame(pos_world, base_pos, base_quat):
    """Transform a world-frame position into the robot base frame."""
    R = quat_to_rot_matrix(base_quat)
    return R.T @ (pos_world - base_pos)


def make_action(eef_delta, gripper=-1.0, rotation=None):
    """Build a 12-dim environment action vector.

    Args:
        eef_delta: (3,) position delta, clipped to [-1, 1] for OSC.
        gripper: -1 = open, 1 = close.
        rotation: (3,) rotation delta, or None for zero.
    """
    action = np.zeros(12)
    action[0:3] = np.clip(eef_delta, -1.0, 1.0)
    if rotation is not None:
        action[3:6] = np.clip(rotation, -1.0, 1.0)
    action[6] = 0.0      # torso
    action[7:10] = 0.0   # base motion
    action[10] = gripper  # -1 open, 1 close
    action[11] = -1.0     # arm control mode
    return action


class PIDController:
    """Simple waypoint PID controller for OpenCabinet."""

    PHASE_APPROACH = "approach"
    PHASE_ALIGN = "align"
    PHASE_GRASP = "grasp"
    PHASE_PULL = "pull"
    PHASE_DONE = "done"

    def __init__(
        self,
        approach_kp=8.0,
        approach_threshold=0.04,
        align_offset=np.array([0.0, 0.0, 0.0]),
        grasp_steps=15,
        pull_kp=6.0,
        pull_direction=np.array([-1.0, 0.0, 0.0]),
        pull_distance=0.25,
    ):
        self.approach_kp = approach_kp
        self.approach_threshold = approach_threshold
        self.align_offset = align_offset
        self.grasp_steps = grasp_steps
        self.pull_kp = pull_kp
        self.pull_direction = pull_direction / (np.linalg.norm(pull_direction) + 1e-8)
        self.pull_distance = pull_distance

        self.phase = self.PHASE_APPROACH
        self.grasp_counter = 0
        self.pull_start_pos = None

    def reset(self):
        self.phase = self.PHASE_APPROACH
        self.grasp_counter = 0
        self.pull_start_pos = None

    def step(self, obs, env):
        """Compute a single action given current observation and environment."""
        handle_world = get_handle_world_pos(env)
        if handle_world is None:
            return make_action(np.zeros(3)), self.phase

        base_pos = obs["robot0_base_pos"]
        base_quat = obs["robot0_base_quat"]
        eef_base = get_eef_base_pos(obs)
        handle_base = world_to_base_frame(handle_world, base_pos, base_quat)

        if self.phase == self.PHASE_APPROACH:
            target = handle_base + self.align_offset
            error = target - eef_base
            dist = np.linalg.norm(error)

            # OSC output_max is 0.05m; normalize so action=1 -> 0.05m delta
            delta = self.approach_kp * error / 0.05
            action = make_action(delta, gripper=-1.0)

            if dist < self.approach_threshold:
                self.phase = self.PHASE_GRASP
                self.grasp_counter = 0

        elif self.phase == self.PHASE_GRASP:
            error = handle_base - eef_base
            delta = self.approach_kp * error / 0.05
            action = make_action(delta, gripper=1.0)
            self.grasp_counter += 1

            if self.grasp_counter >= self.grasp_steps:
                self.phase = self.PHASE_PULL
                self.pull_start_pos = eef_base.copy()

        elif self.phase == self.PHASE_PULL:
            pull_target = self.pull_start_pos + self.pull_direction * self.pull_distance
            error = pull_target - eef_base
            delta = self.pull_kp * error / 0.05

            action = make_action(delta, gripper=1.0)

            pulled = np.linalg.norm(eef_base - self.pull_start_pos)
            if pulled >= self.pull_distance * 0.9:
                self.phase = self.PHASE_DONE

        else:
            action = make_action(np.zeros(3), gripper=1.0)

        return action, self.phase


def run_episode(env, controller, max_steps, verbose=True):
    """Run one PID-controlled episode."""
    obs = env.reset()
    controller.reset()

    ep_meta = env.get_ep_meta()
    lang = ep_meta.get("lang", "")
    if verbose:
        print(f"  Task: {lang}")
        if hasattr(env, "layout_id"):
            print(f"  Layout: {env.layout_id}  Style: {env.style_id}")

    success = False
    hold_count = 0
    frames = []

    for step in range(max_steps):
        action, phase = controller.step(obs, env)

        env_dim = env.action_dim
        if len(action) < env_dim:
            action = np.pad(action, (0, env_dim - len(action)))
        elif len(action) > env_dim:
            action = action[:env_dim]

        obs, reward, done, info = env.step(action)

        if hasattr(env, "sim") and hasattr(env.sim, "render"):
            try:
                frame = env.sim.render(
                    height=512, width=768,
                    camera_name="robot0_agentview_center"
                )[::-1]
                frames.append(frame)
            except Exception:
                pass

        if verbose and step % 20 == 0:
            eef = obs.get("robot0_base_to_eef_pos", np.zeros(3))
            handle_world = get_handle_world_pos(env)
            base_pos = obs["robot0_base_pos"]
            base_quat = obs["robot0_base_quat"]
            handle_base = world_to_base_frame(handle_world, base_pos, base_quat) if handle_world is not None else np.zeros(3)
            dist = np.linalg.norm(eef - handle_base)
            is_open = env._check_success()
            print(
                f"  step {step:4d}  phase={phase:10s}  "
                f"dist={dist:.4f}  reward={reward:+.3f}  "
                f"{'OPEN' if is_open else ''}"
            )

        if env._check_success():
            hold_count += 1
            if hold_count >= 15:
                success = True
                break
        else:
            hold_count = 0

    return success, step + 1, frames


def main():
    parser = argparse.ArgumentParser(
        description="PID controller baseline for OpenCabinet"
    )
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--video_path", type=str, default="/tmp/pid_rollout.mp4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--approach_kp", type=float, default=8.0)
    parser.add_argument("--pull_kp", type=float, default=6.0)
    parser.add_argument("--approach_threshold", type=float, default=0.04)
    parser.add_argument("--grasp_steps", type=int, default=15)
    parser.add_argument("--pull_distance", type=float, default=0.25)
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenCabinet - PID Controller Baseline")
    print("=" * 60)

    if args.offscreen:
        from robocasa.utils.env_utils import create_env

        all_frames = []
        successes = 0

        for ep in range(args.num_episodes):
            print(f"\n--- Episode {ep + 1}/{args.num_episodes} ---")
            env = create_env(
                env_name="OpenCabinet",
                render_onscreen=False,
                seed=args.seed + ep,
                camera_widths=768,
                camera_heights=512,
            )
            controller = PIDController(
                approach_kp=args.approach_kp,
                pull_kp=args.pull_kp,
                approach_threshold=args.approach_threshold,
                grasp_steps=args.grasp_steps,
                pull_distance=args.pull_distance,
            )
            ok, steps, frames = run_episode(env, controller, args.max_steps)
            status = "SUCCESS" if ok else "FAIL"
            print(f"  Result: {status} ({steps} steps)")
            if ok:
                successes += 1
            all_frames.extend(frames)
            env.close()

        if all_frames and args.video_path:
            import imageio
            video_dir = os.path.dirname(args.video_path)
            if video_dir:
                os.makedirs(video_dir, exist_ok=True)
            print(f"\nWriting {len(all_frames)} frames to {args.video_path} ...")
            with imageio.get_writer(args.video_path, fps=20) as writer:
                for frame in all_frames:
                    writer.append_data(frame)
            print(f"Video saved: {args.video_path}")

        print(f"\nFinal: {successes}/{args.num_episodes} episodes succeeded "
              f"({100 * successes / max(1, args.num_episodes):.0f}%)")

    else:
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
            controller = PIDController(
                approach_kp=args.approach_kp,
                pull_kp=args.pull_kp,
                approach_threshold=args.approach_threshold,
                grasp_steps=args.grasp_steps,
                pull_distance=args.pull_distance,
            )
            ok, steps, _ = run_episode(env, controller, args.max_steps)
            status = "SUCCESS" if ok else "FAIL"
            print(f"  Result: {status} ({steps} steps)")
            if ok:
                successes += 1
            time.sleep(0.5)

        env.close()
        print(f"\nFinal: {successes}/{args.num_episodes} episodes succeeded "
              f"({100 * successes / max(1, args.num_episodes):.0f}%)")


if __name__ == "__main__":
    main()
