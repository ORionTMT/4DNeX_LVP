#!/usr/bin/env python3
import argparse
from pathlib import Path

import imageio
import numpy as np
import torch
from diffusers import AutoencoderKLWan

from core.finetune.constants import ENCODED_PM_MEAN, ENCODED_PM_STD
from core.finetune.datasets.utils import preprocess_video_with_resize


def parse_resolution(value: str):
    parts = value.lower().split("x")
    if len(parts) != 3:
        raise ValueError(f"train_resolution must be like 49x480x480, got: {value}")
    frames, height, width = (int(p) for p in parts)
    return frames, height, width


def to_uint8(frames: torch.Tensor) -> torch.Tensor:
    return frames.clamp(0, 255).byte().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Round-trip xyz video through VAE + pm normalization.")
    parser.add_argument("--xyz_video", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="pretrained/Wan2.1-I2V-14B-480P-Diffusers-lvp")
    parser.add_argument("--train_resolution", type=str, default="49x480x480")
    parser.add_argument("--out_dir", type=str, default="results/xyz_roundtrip")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--sample_mode", type=str, choices=["mode", "sample"], default="mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    frames, height, width = parse_resolution(args.train_resolution)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and resize video (0..255)
    video = preprocess_video_with_resize(args.xyz_video, frames, height, width)
    video = video.float()

    # Save resized input for reference
    input_uint8 = to_uint8(video)
    input_np = input_uint8.permute(0, 2, 3, 1).numpy()
    imageio.mimwrite(out_dir / "input_resized.mp4", input_np, fps=args.fps)
    imageio.imwrite(out_dir / "input_first.png", input_np[0])

    # Normalize to [-1,1] like training
    video_norm = video / 255.0 * 2.0 - 1.0
    video_norm = video_norm.permute(1, 0, 2, 3).unsqueeze(0)

    vae = AutoencoderKLWan.from_pretrained(args.model_path, subfolder="vae", torch_dtype=dtype).to(args.device)
    vae.eval()

    with torch.no_grad():
        latent_dist = vae.encode(video_norm.to(args.device, dtype=dtype)).latent_dist
        if args.sample_mode == "sample":
            latent = latent_dist.sample()
        else:
            latent = latent_dist.mode()

        latents_mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(latent.device, latent.dtype)
        )
        latents_std = (
            1.0
            / torch.tensor(vae.config.latents_std)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(latent.device, latent.dtype)
        )

        latent_scaled = (latent - latents_mean) * latents_std

        # Apply pointmap normalization (right-half normalization in training)
        latent_pm = (latent_scaled - ENCODED_PM_MEAN) / ENCODED_PM_STD

        # Reverse pointmap normalization
        latent_scaled_back = latent_pm * ENCODED_PM_STD + ENCODED_PM_MEAN
        latent_back = latent_scaled_back / latents_std + latents_mean

        decoded = vae.decode(latent_back).sample

    decoded = ((decoded + 1.0) / 2.0).clamp(0, 1).float()
    decoded_np = (decoded[0].permute(1, 2, 3, 0).cpu().numpy() * 255.0).clip(0, 255).astype("uint8")

    imageio.mimwrite(out_dir / "decoded_roundtrip.mp4", decoded_np, fps=args.fps)
    imageio.imwrite(out_dir / "decoded_first.png", decoded_np[0])

    input_float = (video / 255.0).permute(0, 2, 3, 1)
    decoded_float = torch.from_numpy(decoded_np).float() / 255.0
    mse = (input_float - decoded_float).pow(2).mean().item()
    print(f"mse: {mse:.6f}")
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
