from typing import Dict, List

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class SimplePolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.net(state)


class DiffusionActionMLP(nn.Module):
    def __init__(self, state_dim, action_dim, num_diffusion_steps=100, hidden_dim=512):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.time_embed = nn.Embedding(num_diffusion_steps, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, noisy_action, timesteps, state):
        t_emb = self.time_embed(timesteps)
        x = torch.cat([noisy_action, state, t_emb], dim=-1)
        return self.net(x)

    def sample_actions(self, state, num_inference_steps=20):
        scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        )
        scheduler.set_timesteps(num_inference_steps)
        x = torch.randn(
            (state.shape[0], self.action_dim),
            device=state.device,
            dtype=state.dtype,
        )
        for t in scheduler.timesteps:
            t_batch = torch.full(
                (state.shape[0],),
                int(t.item()),
                device=state.device,
                dtype=torch.long,
            )
            pred_noise = self.forward(x, t_batch, state)
            x = scheduler.step(pred_noise, t, x).prev_sample
        return x


class ImageEncoder(nn.Module):
    def __init__(self, in_channels=3, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class VisionDiffusionChunkPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        image_keys: List[str],
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        num_diffusion_steps: int = 100,
        vision_feature_dim: int = 256,
        hidden_dim: int = 768,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_diffusion_steps = num_diffusion_steps
        self.image_keys = image_keys
        self.action_chunk_dim = n_action_steps * action_dim

        self.image_encoders = nn.ModuleDict(
            {k: ImageEncoder(out_dim=vision_feature_dim) for k in image_keys}
        )
        cond_dim = (n_obs_steps * state_dim) + (
            len(image_keys) * n_obs_steps * vision_feature_dim
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.time_embed = nn.Embedding(num_diffusion_steps, hidden_dim)
        self.denoiser = nn.Sequential(
            nn.Linear(self.action_chunk_dim + hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.action_chunk_dim),
        )

    def _encode_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        # state: [B, n_obs_steps, state_dim]
        state = obs_dict["state"].reshape(obs_dict["state"].shape[0], -1)
        feats = [state]
        for key in self.image_keys:
            # image: [B, n_obs_steps, C, H, W]
            img = obs_dict[key]
            b, t, c, h, w = img.shape
            img = img.reshape(b * t, c, h, w)
            img_feat = self.image_encoders[key](img).reshape(b, t, -1).reshape(b, -1)
            feats.append(img_feat)
        return self.cond_proj(torch.cat(feats, dim=-1))

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        obs_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # noisy_actions: [B, n_action_steps, action_dim]
        bsz = noisy_actions.shape[0]
        noisy_flat = noisy_actions.reshape(bsz, -1)
        cond = self._encode_obs(obs_dict)
        t_emb = self.time_embed(timesteps)
        x = torch.cat([noisy_flat, cond, t_emb], dim=-1)
        pred = self.denoiser(x)
        return pred.reshape(bsz, self.n_action_steps, self.action_dim)

    def sample_action_chunk(
        self, obs_dict: Dict[str, torch.Tensor], num_inference_steps: int = 32
    ) -> torch.Tensor:
        scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        )
        scheduler.set_timesteps(num_inference_steps)
        device = obs_dict["state"].device
        dtype = obs_dict["state"].dtype
        bsz = obs_dict["state"].shape[0]
        x = torch.randn(
            (bsz, self.n_action_steps, self.action_dim), device=device, dtype=dtype
        )
        for t in scheduler.timesteps:
            t_batch = torch.full((bsz,), int(t.item()), device=device, dtype=torch.long)
            pred_noise = self.forward(x, t_batch, obs_dict)
            x = scheduler.step(pred_noise, t, x).prev_sample
        return x
