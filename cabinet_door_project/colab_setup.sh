#!/usr/bin/env bash
set -euo pipefail

echo "Updating apt packages..."
apt-get -y update
apt-get -y install ffmpeg libgl1-mesa-glx libglib2.0-0

echo "Installing Python dependencies..."
python -m pip install --upgrade pip
python -m pip install \
  torch torchvision torchaudio \
  diffusers imageio imageio-ffmpeg pyarrow pandas numpy pyyaml

echo "Colab setup complete."
echo "Next: set dataset_path in configs/diffusion_policy.yaml or pass --dataset_path."
