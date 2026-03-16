"""
Step 9: Cache Cabinet Handle Positions from Demonstrations
============================================================
Replays all demonstration episodes from the LeRobot dataset and
extracts the 3D position of the cabinet door handle at every timestep.

Saves per-episode ``handle_pos.npy`` files (shape [T, 3]) inside
each episode's extras directory.  The positions are stored in the
**robot base frame** so they can be concatenated directly with the
existing state vector (which already uses base-relative coordinates
for the end-effector).

Usage:
    python 09_cache_handle_positions.py
    python 09_cache_handle_positions.py --dataset_path /path/to/lerobot
    python 09_cache_handle_positions.py --num_episodes 10  # process only first N
"""

import argparse
import gzip
import json
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


# Must set rendering env vars before importing MuJoCo-backed libraries
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

ensure_local_dependency_paths()

import robocasa  # noqa: F401
import robocasa.utils.lerobot_utils as LU
import robosuite
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to


def quat_to_rot_matrix(quat):
    """Convert a [w, x, y, z] quaternion to a 3x3 rotation matrix."""
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ])


def world_to_base_frame(pos_world, base_pos, base_quat):
    """Transform a world-frame position into the robot base frame."""
    R = quat_to_rot_matrix(base_quat)
    return R.T @ (pos_world - base_pos)


def find_handle_geom_names(sim, fixture_name):
    """Find geom names containing both the fixture name and 'handle'."""
    matches = [
        n for n in sim.model.geom_names
        if fixture_name in n and "handle" in n
    ]
    return matches


def find_handle_site_names(sim, fixture_name):
    """Find site names containing both the fixture name and 'handle'."""
    matches = [
        n for n in sim.model.site_names
        if fixture_name in n and "handle" in n
    ]
    return matches


def get_robot_base_pose(env):
    """Get robot base position and quaternion from the sim."""
    robot = env.robots[0]
    base_body_id = env.sim.model.body_name2id(robot.robot_model.root_body)
    base_pos = env.sim.data.body_xpos[base_body_id].copy()
    base_quat = env.sim.data.body_xquat[base_body_id].copy()
    return base_pos, base_quat


def get_dataset_path(override=None):
    if override and os.path.exists(override):
        return Path(override)
    from robocasa.utils.dataset_registry_utils import get_ds_path
    path = get_ds_path("OpenCabinet", source="human")
    if path is None or not os.path.exists(path):
        print("ERROR: Dataset not found. Run 04_download_dataset.py first.")
        sys.exit(1)
    return Path(path)


def main():
    parser = argparse.ArgumentParser(
        description="Cache handle positions from demonstration replays"
    )
    parser.add_argument(
        "--dataset_path", type=str, default=None,
        help="Path to the LeRobot dataset root",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=None,
        help="Only process the first N episodes (default: all)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-cache even if handle_pos.npy already exists",
    )
    args = parser.parse_args()

    dataset = get_dataset_path(args.dataset_path)
    print("=" * 60)
    print("  Cache Handle Positions from Demonstrations")
    print("=" * 60)
    print(f"Dataset: {dataset}")

    env_meta = LU.get_env_metadata(dataset)
    env_kwargs = dict(env_meta["env_kwargs"])
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["use_camera_obs"] = False
    env_kwargs["renderer"] = "mjviewer"

    print("Creating environment...")
    env = robosuite.make(**env_kwargs)

    episodes = LU.get_episodes(dataset)
    if args.num_episodes is not None:
        episodes = episodes[:args.num_episodes]
    print(f"Processing {len(episodes)} episodes\n")

    cached = 0
    skipped = 0
    for idx, ep_dir in enumerate(episodes):
        ep_num = int(ep_dir.name.split("_")[1])
        out_path = ep_dir / "handle_pos.npy"

        if out_path.exists() and not args.force:
            skipped += 1
            continue

        states = LU.get_episode_states(dataset, ep_num)
        model_xml = LU.get_episode_model_xml(dataset, ep_num)
        ep_meta = LU.get_episode_meta(dataset, ep_num)

        initial_state = {
            "states": states[0],
            "model": model_xml,
            "ep_meta": json.dumps(ep_meta),
        }
        reset_to(env, initial_state)

        fxtr_name = ep_meta.get("fixture_refs", {}).get("fxtr", "")
        handle_sites = find_handle_site_names(env.sim, fxtr_name)
        if not handle_sites:
            handle_sites = find_handle_site_names(env.sim, "handle")
        handle_geoms = find_handle_geom_names(env.sim, fxtr_name)
        if not handle_geoms:
            handle_geoms = find_handle_geom_names(env.sim, "handle")

        handle_site = handle_sites[0] if handle_sites else None
        handle_geom = handle_geoms[0] if handle_geoms else None
        if handle_site is None and handle_geom is None:
            print(f"  Episode {ep_num:06d}: WARNING no handle site/geom found, skipping")
            continue

        handle_positions = []
        for t in range(len(states)):
            env.sim.set_state_from_flattened(states[t])
            env.sim.forward()

            if handle_site is not None:
                site_id = env.sim.model.site_name2id(handle_site)
                handle_world = env.sim.data.site_xpos[site_id].copy()
            else:
                geom_id = env.sim.model.geom_name2id(handle_geom)
                handle_world = env.sim.data.geom_xpos[geom_id].copy()

            base_pos, base_quat = get_robot_base_pose(env)
            handle_base = world_to_base_frame(handle_world, base_pos, base_quat)
            handle_positions.append(handle_base)

        handle_positions = np.array(handle_positions, dtype=np.float32)
        np.save(out_path, handle_positions)
        cached += 1

        if (idx + 1) % 10 == 0 or idx == 0:
            print(
                f"  Episode {ep_num:06d}: {len(states)} steps, "
                f"handle={'site' if handle_site else 'geom'}='{handle_site or handle_geom}', "
                f"saved {out_path.name} shape={handle_positions.shape}"
            )

    print(f"\nDone. Cached: {cached}, Skipped (existing): {skipped}")
    env.close()


if __name__ == "__main__":
    main()
