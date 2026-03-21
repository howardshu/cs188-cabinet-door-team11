"""
Step 9: Run Ablations for OpenCabinet
======================================
Runs a small suite of ablations around the low-dim BC U-Net (``06c_train_bc_unet.py``).
Orchestrates training + evaluation and writes ``ablation_results.csv`` under ``--output_root``.

Run from the **repository root** (subprocess paths use ``cabinet_door_project/...``).

Usage:
    python cabinet_door_project/09_run_ablations.py --output_root /tmp/cabinet_ablations --dry_run
    python cabinet_door_project/09_run_ablations.py --output_root /tmp/cabinet_ablations --suite minimal
    python cabinet_door_project/09_run_ablations.py --output_root /tmp/cabinet_ablations --suite full
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


def build_train_cmd(script_path, checkpoint_dir, dataset_path, overrides):
    cmd = [
        sys.executable,
        script_path,
        "--checkpoint_dir",
        checkpoint_dir,
    ]
    if dataset_path:
        cmd += ["--dataset_path", dataset_path]
    for key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(key)
        else:
            cmd += [key, str(value)]
    return cmd


def build_eval_cmd(checkpoint_path, split, num_rollouts, max_steps, overrides):
    cmd = [
        sys.executable,
        "cabinet_door_project/07_evaluate_policy.py",
        "--checkpoint",
        checkpoint_path,
        "--split",
        split,
        "--num_rollouts",
        str(num_rollouts),
        "--max_steps",
        str(max_steps),
    ]
    for key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(key)
        else:
            cmd += [key, str(value)]
    return cmd


def run_cmd(cmd, dry_run=False):
    print(" ".join(cmd))
    if dry_run:
        return 0, ""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output


def parse_success_rate(output):
    for line in output.splitlines():
        if "Success rate:" in line:
            try:
                return float(line.split("Success rate:")[1].strip().strip("%"))
            except Exception:
                return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Run OpenCabinet ablation suite")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="target", choices=["pretrain", "target"])
    parser.add_argument("--num_rollouts", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--suite", type=str, default="minimal", choices=["minimal", "full"])
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_script = "cabinet_door_project/06c_train_bc_unet.py"

    base_train = {
        "--epochs": 100,
        "--batch_size": 128,
        "--lr": 1e-3,
        "--n_obs_steps": 2,
        "--n_action_steps": 16,
        "--execute_steps": 8,
        "--base_channels": 32,
        "--channel_mults": "1,2",
        "--kernel_size": 5,
        "--cond_dim": 256,
        "--val_fraction": 0.15,
        "--patience": 25,
        "--min_epochs": 2,
    }

    base_eval = {
        "--execute_steps": 8,
        "--success_threshold_rad": 0.3,
        "--gripper_threshold": 0.0,
        "--base_mode_threshold": 0.0,
    }

    experiments = []

    # experiments.append(
    #     {
    #         "name": "full_handle",
    #         "train_overrides": {},
    #         "eval_overrides": {},
    #     }
    # )
    # experiments.append(
    #     {
    #         "name": "handle_to_eef_only",
    #         "train_overrides": {"--no_handle_pos": True},
    #         "eval_overrides": {},
    #     }
    # )
    # experiments.append(
    #     {
    #         "name": "handle_pos_only",
    #         "train_overrides": {"--no_handle_to_eef": True},
    #         "eval_overrides": {},
    #     }
    # )
    # experiments.append(
    #     {
    #         "name": "no_handle_features",
    #         "train_overrides": {"--no_handle_pos": True, "--no_handle_to_eef": True},
    #         "eval_overrides": {},
    #     }
    # )
    experiments.append(
        {
            "name": "quaternions",
            "train_overrides": {"--no_drop_quaternions": True},
            "eval_overrides": {},
        }
    )

    # experiments.append(
    #     {
    #         "name": "highdim_bc_unet",
    #         "script": "cabinet_door_project/06d_train_highdim_bc_unet.py",
    #         "train_overrides": {
    #             "--base_channels": 128,
    #             "--channel_mults": "1,2,4",
    #             "--cond_dim": 512,
    #         },
    #         "eval_overrides": {},
    #     }
    # )
    # experiments.append(
    #     {
    #         "name": "highdim_quaternions",
    #         "script": "cabinet_door_project/06d_train_highdim_bc_unet.py",
    #         "train_overrides": {
    #             "--base_channels": 128,
    #             "--channel_mults": "1,2,4",
    #             "--cond_dim": 512,
    #             "--no_drop_quaternions": True,
    #         },
    #         "eval_overrides": {},
    #     }
    # )

    experiments.append(
        {
            "name": "chunk_H16_exec8",
            "train_overrides": {"--n_action_steps": 16, "--execute_steps": 8},
            "eval_overrides": {"--execute_steps": 8},
        }
    )
    experiments.append(
        {
            "name": "chunk_H4_exec2",
            "train_overrides": {"--n_action_steps": 4, "--execute_steps": 2},
            "eval_overrides": {"--execute_steps": 2},
        }
    )
    experiments.append(
        {
            "name": "chunk_H1_exec1",
            "train_overrides": {
                "--n_action_steps": 1,
                "--execute_steps": 1,
                "--channel_mults": "1",
            },
            "eval_overrides": {"--execute_steps": 1},
        }
    )

    if args.suite == "full":
        experiments += [
            {
                "name": "bigger_model",
                "train_overrides": {"--base_channels": 128, "--channel_mults": "1,2,4"},
                "eval_overrides": {},
            },
            {
                "name": "gripper_thresh_0p5",
                "train_overrides": {},
                "eval_overrides": {"--gripper_threshold": 0.5},
            },
        ]

    results_path = output_root / "ablation_results.csv"
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "name",
                "train_cmd",
                "eval_cmd",
                "exit_code",
                "success_rate",
            ]
        )

    for exp in experiments:
        name = exp["name"]
        ckpt_dir = output_root / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        train_overrides = dict(base_train)
        train_overrides.update(exp["train_overrides"])
        eval_overrides = dict(base_eval)
        eval_overrides.update(exp["eval_overrides"])

        script = exp.get("script", train_script)

        train_cmd = build_train_cmd(
            script, str(ckpt_dir), args.dataset_path, train_overrides
        )
        eval_cmd = build_eval_cmd(
            str(ckpt_dir / "best_policy.pt"),
            args.split,
            args.num_rollouts,
            args.max_steps,
            eval_overrides,
        )

        exit_code = 0
        success_rate = None
        if not args.skip_train:
            exit_code, _ = run_cmd(train_cmd, dry_run=args.dry_run)
        if exit_code == 0 and not args.skip_eval:
            exit_code, output = run_cmd(eval_cmd, dry_run=args.dry_run)
            success_rate = parse_success_rate(output)

        with open(results_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    name,
                    " ".join(train_cmd),
                    " ".join(eval_cmd),
                    exit_code,
                    success_rate,
                ]
            )

    print(f"\nResults written to: {results_path}")


if __name__ == "__main__":
    main()
