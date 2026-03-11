# Cabinet Door Opening Robot - CS 188 Starter Project

### Disclaimer

This project was designed for CS 188 - Intro to Robotics as a template starter project. If you have any issues with the codebase, please email me at holdengs @ cs.ucla.edu!

## Overview

In this project you will build a robot that learns to open kitchen cabinet doors
using **RoboCasa365**, a large-scale simulation benchmark for everyday robot
tasks. You will progress from understanding the simulation environment, to
collecting demonstrations, to training a neural-network policy that controls the
robot autonomously.

This fork includes a **native diffusion-policy baseline** directly in
`cabinet_door_project/06_train_policy.py` (no external Hydra training repo
required for core training / eval / visualization).

### What you will learn

1. How robotic manipulation environments are structured (MuJoCo + robosuite + RoboCasa)
2. How the `OpenCabinet` task works -- sensors, actions, success criteria
3. How to collect and use demonstration datasets (human + MimicGen)
4. How to train a behavior-cloning policy from demonstrations
5. How to evaluate your trained policy in simulation

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

## Project Structure

```
cabinet_door_project/
  00_verify_installation.py      # Check that everything is installed correctly
  01_explore_environment.py      # Create the OpenCabinet env, inspect observations/actions
  02_random_rollouts.py          # Run random actions, save video, understand the task
  03_teleop_collect_demos.py     # Teleoperate the robot to collect your own demonstrations
  04_download_dataset.py         # Download the pre-collected OpenCabinet dataset
  05_playback_demonstrations.py  # Play back demonstrations to see expert behavior
  06_train_policy.py             # Train simple MLP or diffusion policy (default)
  07_evaluate_policy.py          # Evaluate checkpoints (simple or diffusion)
  08_visualize_policy_rollout.py # Visualize rollouts (simple or diffusion)
  policy_models.py               # Shared model definitions used by train/eval
  configs/
    diffusion_policy.yaml        # Starter diffusion hyperparameters
  notebook.ipynb                 # Interactive Jupyter notebook companion
install.sh                       # Installation script (macOS + WSL/Linux)
README.md                        # This file
```

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
useful, but it helps you understand the action space). Saves a video to
`/tmp/cabinet_random_rollouts.mp4`.

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

```bash
python 06_train_policy.py --policy vision_diffusion_chunk --config configs/diffusion_policy.yaml
```

Trains the built-in vision-conditioned diffusion policy that uses all three
cameras + robot state and predicts action chunks. Checkpoints are saved as
`.pt` files and include model type + normalization statistics so evaluation and
visualization scripts can load them automatically.

You can still run the simple MLP baseline for comparison:

```bash
python 06_train_policy.py --policy simple
```

Few-hours profile (now default in `configs/diffusion_policy.yaml`):

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

For higher final quality (longer runs), increase epochs / model size and remove
the steps-per-epoch cap (`max_train_steps_per_epoch: null`).

### Step 7: Evaluate Your Policy

```bash
python 07_evaluate_policy.py --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt
```

Runs your trained policy in simulation and reports success rate across episodes
and scene splits (`pretrain` / `target`). The evaluator auto-detects whether
the checkpoint is from the simple MLP, low-dim diffusion, or vision-chunk
diffusion model.

### Step 8: Visualize a Rollout

```bash
python 08_visualize_policy_rollout.py --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt --offscreen
```

Visualizes policy behavior with the same chunked control logic used during
evaluation.

---

## Key Concepts

### The OpenCabinet Task

- **Goal**: Open a kitchen cabinet door
- **Fixture**: `HingeCabinet` (a cabinet with hinged doors)
- **Initial state**: Cabinet door is closed; robot is positioned nearby
- **Success**: `fixture.is_open(env)` returns `True`
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

| Key | Dim | Description |
|-----|-----|-------------|
| `end_effector_position` | 3 | Delta (dx, dy, dz) for the end-effector |
| `end_effector_rotation` | 3 | Delta rotation (axis-angle) |
| `gripper_close` | 1 | 0 = open, 1 = close |
| `base_motion` | 4 | (forward, side, yaw, torso) |
| `control_mode` | 1 | 0 = arm control, 1 = base control |

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

Script 03 already provides keyboard teleoperation. I have it set up with a DAgger mode that may or may not be kinda buggy. Use it to close the loop:
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

I'll continually update this section as students find bugs in the system. Please, let me know if you encounter issues!

| Problem | Solution |
|---------|----------|
| `MuJoCo version must be 3.3.1` | `pip install mujoco==3.3.1` |
| `numpy version must be 2.2.5` | `pip install numpy==2.2.5` |
| Rendering crashes on Mac | Use `mjpython` instead of `python` |
| `GLFW error` on headless server | Set `export MUJOCO_GL=egl` or `osmesa` |
| Out of GPU / MPS memory during training | Reduce `batch_size`, `image_size`, `hidden_dim`, and `num_inference_steps` in `configs/diffusion_policy.yaml` |
| Kitchen assets not found | Run `python -m robocasa.scripts.download_kitchen_assets` |

---

## References

- [RoboCasa Paper & Website](https://robocasa.ai/)
- [RoboCasa GitHub](https://github.com/robocasa/robocasa)
- [robosuite Documentation](https://robosuite.ai/)
- [Diffusion Policy Paper](https://diffusion-policy.cs.columbia.edu/)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [LeRobot Dataset Format](https://github.com/huggingface/lerobot)
