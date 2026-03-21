# Cabinet Door Opening Robot - CS 188 Final Project

## Overview

In this project, we built infrastructure for a robot to learn to open kitchen cabinet doors
using **RoboCasa365**, a large-scale simulation benchmark for everyday robot
tasks. We progressed from understanding the simulation environment, to
collecting demonstrations, to training a neural-network policy that controls the robot autonomously.

This repository still includes the **simple MLP** and **low-dim / vision diffusion**
baselines in `cabinet_door_project/06_train_policy.py`. For the OpenCabinet demo
scale, the team’s stronger **state + handle-feature** policies live in
`06c_train_bc_unet.py` (compact U-Net), `06d_train_highdim_bc_unet.py` (wider
U-Net), and `06b_train_diffusion_unet.py` (diffusion U-Net).

### Where to run commands

- **Steps 0–8** (`00_*.py` … `08_*.py`): run from `cabinet_door_project/` with the
  venv activated (`source ../.venv/bin/activate` from that folder, or activate
  then `cd cabinet_door_project`).
- **`09_run_ablations.py`**: run from the **repository root** (paths inside the
  script are `cabinet_door_project/...`).
- **Colab**: use the paths shown in the Colab section (typically repo root as cwd).

### Overview of Codebase Goals

1. Demonstrate how robotic manipulation environments are structured (MuJoCo + robosuite + RoboCasa)
2. Demonstrate how the `OpenCabinet` task works -- sensors, actions, success criteria
3. Show how to collect and use demonstration datasets (human + MimicGen)
4. Train a behavior-cloning policy from demonstrations
5. Evaluate your trained policy in simulation

### The robot

We use the **PandaOmron** mobile manipulator -- a Franka Panda 7-DOF arm
mounted on an Omron wheeled base with a torso lift joint. This is the default
and best-supported robot in RoboCasa.

---

## Installation

Run the install script (works on **macOS** and **WSL/Linux**):

```bash
./install.sh
```

This will:
- Create a Python virtual environment (`.venv`)
- Clone and install robosuite and robocasa
- Install all Python dependencies (PyTorch, numpy, matplotlib, etc.)
- Download RoboCasa kitchen assets (~10 GB)

After installation, activate the environment:

```bash
source .venv/bin/activate
```

Then verify everything works:

```bash
cd cabinet_door_project
python 00_verify_installation.py
```

> **macOS note:** Scripts that open a rendering window (03, 05) require
> `mjpython` instead of `python`. The install script will remind you of this.

---

## Project Structure

```
cabinet_door_project/
  00_verify_installation.py      # Check that everything is installed correctly
  01_explore_environment.py      # Create the OpenCabinet env, inspect observations/actions
  02_random_rollouts.py          # Run random actions, save video, understand the task
  03_teleop_collect_demos.py     # Teleoperate the robot to collect your own demonstrations
  04_download_dataset.py         # Download the pre-collected OpenCabinet dataset
  05_playback_demonstrations.py  # Play back demonstrations to see expert behavior
  05b_augment_handle_data.py     # Augment dataset with handle features (recommended before 06b–06d)
  06_train_policy.py             # Train vision-diffusion-chunk (default), low-dim diffusion, or simple MLP
  06b_train_diffusion_unet.py    # State + handle features: Diffusion + 1D U-Net (~15M params)
  06c_train_bc_unet.py           # State + handle features: BC 1D U-Net (compact, fast)
  06d_train_highdim_bc_unet.py   # State + handle features: wider BC 1D U-Net
  07_evaluate_policy.py          # Evaluate checkpoints (auto-detects model type)
  08_visualize_policy_rollout.py # Visualize rollouts
  09_run_ablations.py            # Ablation suite (06c BC U-Net); run from repo root
  policy_models.py               # Shared model definitions used by train/eval
  colab_setup.sh               # Colab dependency install
  configs/
    diffusion_policy.yaml        # Starter vision-diffusion hyperparameters
  notebook.ipynb                 # Interactive Jupyter notebook companion
install.sh                       # Installation script (macOS + WSL/Linux)
ablation_results/                # Optional: example ablation outputs (checkpoints + CSV)
README.md                        # This file
```

### Policy training at a glance

| Script | Inputs | Default checkpoint directory |
|--------|--------|------------------------------|
| `06_train_policy.py` | Images + state (vision) or state only | `/tmp/cabinet_policy_checkpoints` |
| `06c_train_bc_unet.py` | Low-dim state + handle features | `bc_unet_checkpoints/` (under cwd) |
| `06d_train_highdim_bc_unet.py` | Full state vector + handle features | `highdim_bc_unet_checkpoints/` |
| `06b_train_diffusion_unet.py` | Low-dim state + handle features | `diffusion_unet_policy_checkpoints/` |

Training scripts resolve the OpenCabinet dataset via RoboCasa’s registry unless you
pass `--dataset_path` to a LeRobot-format root (`data/`, `videos/`, optionally
nested under `lerobot/`).

---

## Step-by-Step Guide

### Step 0: Verify Installation

```bash
python 00_verify_installation.py
```

This checks that MuJoCo, robosuite, RoboCasa, and all dependencies are
correctly installed and that the `OpenCabinet` environment can be created.

### Step 1: Explore the Environment

```bash
python 01_explore_environment.py
```

This script creates the `OpenCabinet` environment and prints detailed
information about:
- **Observation space**: what the robot sees (camera images, joint positions,
  gripper state, base pose)
- **Action space**: what the robot can do (arm movement, gripper open/close,
  base motion, control mode)
- **Task description**: the natural language instruction for the episode
- **Success criteria**: how the environment determines task completion

### Step 2: Random Rollouts

```bash
python 02_random_rollouts.py
```

Runs the robot with random actions to see what happens (spoiler: nothing
useful, but it helps you understand the action space). By default saves a video
to `cabinet_door_project/videos/cabinet_random_rollouts.mp4` (override with
`--video_path`).

### Step 3: Teleoperate and Collect Demonstrations

```bash
# Mac users: use mjpython instead of python
python 03_teleop_collect_demos.py
```

Control the robot yourself using the keyboard to open cabinet doors. This
gives you intuition for the task difficulty and generates demonstration data.

**Keyboard controls:**
| Key | Action |
|-----|--------|
| Ctrl+q | Reset simulation |
| spacebar | Toggle gripper (open/close) |
| up-right-down-left | Move horizontally in x-y plane |
| .-; | Move vertically |
| o-p | Rotate (yaw) |
| y-h | Rotate (pitch) |
| e-r | Rotate (roll) |
| b | Toggle arm/base mode (if applicable) |
| s | Switch active arm (if multi-armed robot) |
| = | Switch active robot (if multi-robot environment) |              

### Step 4: Download Pre-collected Dataset

```bash
python 04_download_dataset.py
```

Downloads the official OpenCabinet demonstration dataset from the RoboCasa
servers. This includes both human demonstrations and MimicGen-expanded data
across diverse kitchen scenes.

### Step 5: Play Back Demonstrations

```bash
python 05_playback_demonstrations.py
```

Visualize the downloaded demonstrations to see how an expert opens cabinet
doors. This is the data your policy will learn from.

### Step 6: Train a Policy

#### Recommended (Working Setups)

For the OpenCabinet dataset size (~100 demos), the strongest results come from
behavior cloning with input state vectors augmented with handle features and action chunking:

```bash
# 1) Augment the dataset with handle features
python 05b_augment_handle_data.py

# 2a) Very fast training: low-dim BC 1D U-Net (defaults: handle_pos + handle_to_eef, H=16, execute=8)
python 06c_train_bc_unet.py --checkpoint_dir /path/to/checkpoints/folder

# 2b) Larger BC U-Net: high-dim state + handle features (same chunk defaults unless overridden)
python 06d_train_highdim_bc_unet.py --checkpoint_dir /path/to/checkpoints/folder

# 2c) Slower: Diffusion 1D U-Net on low-dim state + handle features
python 06b_train_diffusion_unet.py --checkpoint_dir /path/to/checkpoints/folder

# 3) Evaluate (see "Evaluation success criteria" below; relaxed is default).
#    Match --execute_steps to what the checkpoint was trained with (default 8 for the commands above).
python 07_evaluate_policy.py \
  --checkpoint /path/to/checkpoints/folder/best_policy.pt \
  --num_rollouts 50 \
  --execute_steps 8
```

To run ablations for the low-dimensional BC U-Net (`06c`), from the **repository root**:

```bash
python cabinet_door_project/09_run_ablations.py --output_root /tmp/cabinet_ablations --suite minimal
```

Use `--suite full` for two extra experiments (larger model + gripper-threshold eval).
Add `--dry_run` to print train/eval commands without running them. Optional:
`--dataset_path`, `--split pretrain|target`, `--skip_train` / `--skip_eval`.

#### Vision Diffusion (slowest training)

```bash
python 06_train_policy.py --policy vision_diffusion_chunk --config configs/diffusion_policy.yaml
```

`06_train_policy.py` defaults to `--policy vision_diffusion_chunk`. Other choices
are `diffusion` (low-dimensional DDPM from state) and `simple` (MLP BC). Trains
the vision-conditioned diffusion policy that uses all three cameras + robot state
and predicts action chunks. Checkpoints are saved as `.pt` files and include model
type + normalization statistics so evaluation and visualization scripts can load
them automatically.

Profile for training Vision Diffusion within a few hours (now default in `configs/diffusion_policy.yaml`):

```bash
python 06_train_policy.py \
  --policy vision_diffusion_chunk \
  --epochs 20 \
  --batch_size 16 \
  --max_train_steps_per_epoch 150 \
  --image_size 64 \
  --hidden_dim 512 \
  --vision_feature_dim 128 \
  --n_obs_steps 2 \
  --n_action_steps 8 \
  --num_diffusion_steps 100 \
  --num_inference_steps 16
```

#### Simple MLP baseline

You can still run the simple MLP baseline for comparison:

```bash
python 06_train_policy.py --policy simple
```



For higher final quality (longer runs), could increase epochs / model size and remove
the steps-per-epoch cap (`max_train_steps_per_epoch: null`).

### Step 7: Evaluate Your Policy

```bash
python 07_evaluate_policy.py --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt
```

If you run into an error with `robocasa` not found, you may have to set `PYTHONPATH`:
```bash
export PYTHONPATH="$PWD/robocasa:$PWD/robosuite:$PYTHONPATH"
```

Runs your trained policy in simulation and reports success rate across episodes
and scene splits (`pretrain` / `target`). The evaluator auto-detects whether
the checkpoint is from the simple MLP, low-dim diffusion, or vision-chunk
diffusion model.

**Success criterion (pick one):** RoboCasa’s built-in task success
(`env._check_success()` → `fixture.is_open`) requires **every** door joint on
the cabinet to reach a high normalized openness (default **0.9** per joint).
On **double-door** cabinets that is stricter than “open one door.”
`07_evaluate_policy.py` therefore supports three modes:

| Mode | Flags | Meaning |
|------|--------|---------|
| **Relaxed** (default) | *(none)* | Any single door past a loose hinge threshold (`--success_threshold_rad`, default **0.3** rad). Good for quick iteration; not identical to RoboCasa’s fixture math. |
| **Strict (RoboCasa)** | `--strict_success` | Same as `env._check_success()`: **all** doors must satisfy the fixture openness check. |
| **Fixture threshold, one door** | `--fixture_any_door_success` | Same **normalized** per-door definition as `Fixture.is_open` (default `--fixture_open_threshold 0.90`), but success if **any one** door passes — matches “open a cabinet door” without requiring both sides of a double door. |

Examples:

```bash
# Default: relaxed
python 07_evaluate_policy.py --checkpoint path/to/best_policy.pt --num_rollouts 50

# RoboCasa-native: all doors open per fixture
python 07_evaluate_policy.py --checkpoint path/to/best_policy.pt --strict_success

# RoboCasa-style openness (0.9) on a single door only (recommended vs. strict for double doors)
python 07_evaluate_policy.py --checkpoint path/to/best_policy.pt --fixture_any_door_success

# Optional: tune the fixture threshold (still “any one door”)
python 07_evaluate_policy.py --checkpoint path/to/best_policy.pt \
  --fixture_any_door_success --fixture_open_threshold 0.90
```

`--strict_success` and `--fixture_any_door_success` cannot be used together.

### Step 8: Visualize a Rollout

```bash
python 08_visualize_policy_rollout.py --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt --offscreen
```

Visualizes policy behavior with the same chunked control logic used during
evaluation.

---

### Google Colab (Training)

Use this copy-paste sequence in a fresh Colab GPU notebook:

```python
# Cell 1: clone repo and enter it
!git clone <YOUR_REPO_URL>
%cd cs188-cabinet-door-team11
```

```python
# Cell 2: install Colab dependencies
!bash cabinet_door_project/colab_setup.sh
```

```python
# Cell 3 (optional): mount Drive for dataset + checkpoints
from google.colab import drive
drive.mount('/content/drive')
```

```python
# Cell 4: train (set paths for your Drive layout)
DATASET_PATH = "/content/drive/MyDrive/OpenCabinet/lerobot"
CKPT_DIR = "/content/drive/MyDrive/cabinet_policy_checkpoints"

# if training vision diffusion policy
!python cabinet_door_project/06_train_policy.py \
  --policy vision_diffusion_chunk \
  --config cabinet_door_project/configs/diffusion_policy.yaml \
  --dataset_path "$DATASET_PATH" \
  --checkpoint_dir "$CKPT_DIR"
```

Notes:
- `DATASET_PATH` must point to a LeRobot dataset root containing `data/` and `videos/`.
- If your repo already lives on Drive, skip `git clone` and `%cd` into that folder instead.
- On Colab, prefer offscreen evaluation/visualization (no on-screen viewer).

---

## Key Concepts

### The OpenCabinet Task

- **Goal**: Open a kitchen cabinet door
- **Fixture**: `HingeCabinet` (a cabinet with hinged doors)
- **Initial state**: Cabinet door is closed; robot is positioned nearby
- **Success (RoboCasa / `env._check_success`)**: For door “open” tasks, success calls
  `fixture.is_open(env)`: **each** door joint on that fixture must have
  normalized openness ≥ **0.9** (see `robocasa` `Fixture.is_open`). On a
  two-door cabinet, that typically means **both** doors must be nearly fully
  open.
- **Evaluation in this repo**: `07_evaluate_policy.py` defaults to a **relaxed**
  “any door” check for easier reporting; use `--strict_success` for the exact
  RoboCasa criterion above, or `--fixture_any_door_success` for the same
  per-door **0.9** threshold but success when **any one** door qualifies (see
  Step 7).
- **Horizon**: 500 timesteps at 20 Hz control frequency (25 seconds)
- **Scene variety**: 2,500+ kitchen layouts/styles for generalization

### Observation Space (PandaOmron)

| Key | Shape | Description |
|-----|-------|-------------|
| `robot0_agentview_left_image` | (256, 256, 3) | Left shoulder camera |
| `robot0_agentview_right_image` | (256, 256, 3) | Right shoulder camera |
| `robot0_eye_in_hand_image` | (256, 256, 3) | Wrist-mounted camera |
| `robot0_gripper_qpos` | (2,) | Gripper finger positions |
| `robot0_base_pos` | (3,) | Base position (x, y, z) |
| `robot0_base_quat` | (4,) | Base orientation quaternion |
| `robot0_base_to_eef_pos` | (3,) | End-effector pos relative to base |
| `robot0_base_to_eef_quat` | (4,) | End-effector orientation relative to base |

### Action Space (PandaOmron)

| Index | Key | Dim | Description |
|-------|-----|-----|-------------|
| 0-2 | `end_effector_position` | 3 | Delta (dx, dy, dz) for the end-effector |
| 3-5 | `end_effector_rotation` | 3 | Delta rotation (axis-angle) |
| 6 | `torso` | 1 | Torso lift joint |
| 7-9 | `base_motion` | 3 | (forward, side, yaw) |
| 10 | `gripper_close` | 1 | -1 = open, 1 = close |
| 11 | `control_mode` | 1 | -1 = arm control, 1 = base control |

### Dataset Format (LeRobot)

Datasets are stored in LeRobot format:
```
dataset/
  meta/           # Episode metadata (task descriptions, camera info)
  videos/         # MP4 videos from each camera
  data/           # Parquet files with actions, states, rewards
  extras/         # Per-episode metadata
```

---

## Architecture Diagram

```
                    RoboCasa Stack
                    ==============

  +-------------------+     +-------------------+
  |   Kitchen Scene   |     |   OpenCabinet     |
  |  (2500+ layouts)  |     |   (Task Logic)    |
  +--------+----------+     +--------+----------+
           |                         |
           v                         v
  +------------------------------------------------+
  |              Kitchen Base Class                 |
  |  - Fixture management (cabinets, fridges, etc)  |
  |  - Object placement (bowls, cups, etc)          |
  |  - Robot positioning                            |
  +------------------------+-----------------------+
                           |
                           v
  +------------------------------------------------+
  |              robosuite (Backend)                |
  |  - MuJoCo physics simulation                   |
  |  - Robot models (PandaOmron, GR1, Spot, ...)   |
  |  - Controller framework                        |
  +------------------------+-----------------------+
                           |
                           v
  +------------------------------------------------+
  |              MuJoCo 3.3.1 (Physics)            |
  |  - Contact dynamics, rendering, sensors        |
  +------------------------------------------------+
```

---

## Research Directions

This repository now includes a vision-conditioned diffusion baseline with action
chunking. Good next steps are extending it toward stronger performance:

### Stronger Visual Backbone

Current model uses a lightweight CNN encoder for each camera. Consider
upgrading to a stronger pretrained visual backbone and adding temporal fusion
across observation steps.

### DAgger (Online Correction)

Script 03 provides keyboard teleoperation. We have set it up with a DAgger mode. Use it to close the loop:
train a policy, roll it out, then have a human take over and correct the robot
whenever it fails. Aggregate these corrections into the training set and
retrain. This directly attacks distribution shift — the fundamental reason
offline BC degrades at test time — by collecting data in the states the policy
actually visits. Even one or two rounds of DAgger can dramatically improve
robustness. See [Ross et al., 2011](https://arxiv.org/abs/1011.0686).

### Longer Action Horizon

Instead of predicting one action per timestep, predict the next *K* actions at
once and execute them open-loop before re-planning. This improves temporal
coherence and often stabilizes manipulation rollouts. Practical sweeps:
`n_action_steps = 8, 12, 16` and `n_obs_steps = 2, 4`.

### Other Ideas
- Gaussian Mixture Model for output logits. Can ameliorate the MSE multimodality issue.
- Vision Transformer. Will need a beefier computer to see benefits but definitely can improve policy at scale.
- Hooking in an existing VLM and experimenting with zero-shot inference.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `MuJoCo version must be 3.3.1` | `pip install mujoco==3.3.1` |
| `numpy version must be 2.2.5` | `pip install numpy==2.2.5` |
| Rendering crashes on Mac | Use `mjpython` instead of `python` |
| `GLFW error` on headless server | Set `export MUJOCO_GL=egl` or `osmesa` |
| Out of GPU / MPS memory during training | Reduce `batch_size`, `image_size`, `hidden_dim`, and `num_inference_steps` in `configs/diffusion_policy.yaml` |
| Kitchen assets not found | Run `python -m robocasa.scripts.download_kitchen_assets` |
| `09_run_ablations.py` cannot find `cabinet_door_project/...` | Run the script from the **repository root**, not from inside `cabinet_door_project/` |

---

## References

- [RoboCasa Paper & Website](https://robocasa.ai/)
- [RoboCasa GitHub](https://github.com/robocasa/robocasa)
- [robosuite Documentation](https://robosuite.ai/)
- [Diffusion Policy Paper](https://diffusion-policy.cs.columbia.edu/)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [LeRobot Dataset Format](https://github.com/huggingface/lerobot)
