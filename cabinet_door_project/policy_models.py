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


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, groups=8):
        super().__init__()
        padding = kernel_size // 2
        num_groups = min(groups, out_ch)
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
            nn.SiLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        base_channels=64,
        channel_mults=(1, 2, 4),
        kernel_size=5,
    ):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.skip_channels = []
        ch = in_channels

        for mult in channel_mults:
            out_ch = base_channels * mult
            self.downs.append(ConvBlock1D(ch, out_ch, kernel_size=kernel_size))
            self.skip_channels.append(out_ch)
            self.downs.append(
                nn.Conv1d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
            )
            ch = out_ch

        self.mid = ConvBlock1D(ch, ch, kernel_size=kernel_size)

        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            skip_ch = self.skip_channels.pop() if self.skip_channels else out_ch
            self.ups.append(
                nn.ConvTranspose1d(ch, out_ch, kernel_size=4, stride=2, padding=1)
            )
            self.ups.append(ConvBlock1D(out_ch + skip_ch, out_ch, kernel_size=kernel_size))
            ch = out_ch

        self.final = nn.Conv1d(ch, out_channels, kernel_size=1)

    @staticmethod
    def _match_length(x, target_len):
        if x.shape[-1] > target_len:
            return x[..., :target_len]
        if x.shape[-1] < target_len:
            pad = target_len - x.shape[-1]
            return nn.functional.pad(x, (0, pad))
        return x

    def forward(self, x):
        skips = []
        for i in range(0, len(self.downs), 2):
            x = self.downs[i](x)
            skips.append(x)
            x = self.downs[i + 1](x)

        x = self.mid(x)

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips.pop()
            x = self._match_length(x, skip.shape[-1])
            x = torch.cat([x, skip], dim=1)
            x = self.ups[i + 1](x)

        return self.final(x)


class BCUnet1DPolicy(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        n_obs_steps=2,
        n_action_steps=16,
        base_channels=32,
        channel_mults=(1, 2),
        kernel_size=5,
        cond_dim=256,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.cond_dim = cond_dim

        self.cond_proj = nn.Sequential(
            nn.Linear(n_obs_steps * state_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
        )
        self.pos_embed = nn.Parameter(torch.zeros(n_action_steps, cond_dim))
        self.unet = UNet1D(
            in_channels=cond_dim,
            out_channels=action_dim,
            base_channels=base_channels,
            channel_mults=channel_mults,
            kernel_size=kernel_size,
        )

    def forward(self, obs_hist):
        # obs_hist: [B, n_obs_steps, state_dim]
        bsz = obs_hist.shape[0]
        cond = self.cond_proj(obs_hist.reshape(bsz, -1))
        tokens = cond[:, None, :] + self.pos_embed[None, :, :]
        x = tokens.permute(0, 2, 1)
        out = self.unet(x)
        return out.permute(0, 2, 1)
