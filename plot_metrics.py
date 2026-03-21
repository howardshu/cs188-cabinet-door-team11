import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

paths = {
    'vision_diffusion_chunk': ('Vision Diffusion Chunk', 'vision_diffusion_chunk_colab_logs/training_metrics.csv'),
    'diffusion_unet_200': ('Diffusion UNet (200 ep)', 'lowdim_unet_policy_checkpoints/training_metrics.csv'),
    'diffusion_unet_400': ('Diffusion UNet (400 ep)', 'lowdim_unet_policy_checkpoints3/training_metrics.csv'),
    'low_dim_bc_unet': ('Low-dim Simple UNet', 'bc_unet_checkpoints/training_metrics.csv'),
    'high_dim_bc_unet': ('High-dim Simple UNet', 'bc_unet_highdim_checkpoints/training_metrics.csv')
}

for key, (name, path) in paths.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        plt.figure(figsize=(8, 5))
        if 'train_denoise_loss' in df.columns:
            plt.plot(df['epoch'], df['train_denoise_loss'], label='Train Loss')
        elif 'train_loss' in df.columns:
            plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
            
        if 'val_denoise_loss' in df.columns:
            df_val = df.dropna(subset=['val_denoise_loss'])
            plt.plot(df_val['epoch'], df_val['val_denoise_loss'], label='Val Loss', marker='o', markersize=3)
        elif 'val_loss' in df.columns:
            df_val = df.dropna(subset=['val_loss'])
            plt.plot(df_val['epoch'], df_val['val_loss'], label='Val Loss', marker='o', markersize=3)
            
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{name} Loss vs Epochs')
        plt.legend()
        plt.grid(True)
        
        if 'diffusion_unet' in key:
            plt.yscale('log')
            plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())
            plt.gca().yaxis.set_minor_formatter(ticker.ScalarFormatter())
            plt.gca().tick_params(axis='y', which='minor', labelsize=8)
        else:
            plt.yscale('linear')
            
        plt.tight_layout()
        plt.savefig(f'{key}_plot.png')
        plt.close()

# Simple MLP
mlp_epochs = [1, 10, 20, 30, 40, 50]
mlp_loss = [0.123414, 0.061720, 0.043680, 0.037650, 0.033248, 0.030390]

plt.figure(figsize=(8, 5))
plt.plot(mlp_epochs, mlp_loss, label='Train Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Baseline MLP Loss vs Epochs')
plt.legend()
plt.grid(True)
plt.yscale('linear')
plt.tight_layout()
plt.savefig('baseline_mlp_plot.png')
plt.close()
print("Plots saved.")
