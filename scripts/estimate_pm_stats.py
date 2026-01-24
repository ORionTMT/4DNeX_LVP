#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Tuple

import torch
from tqdm import tqdm

from diffusers import AutoencoderKLWan

from core.finetune.datasets.utils import preprocess_video_with_resize


def parse_resolution(value: str) -> Tuple[int, int, int]:
    parts = value.lower().split("x")
    if len(parts) != 3:
        raise ValueError(f"train_resolution must be like 49x480x480, got: {value}")
    frames, height, width = (int(p) for p in parts)
    return frames, height, width


def load_metadata(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def resolve_path(path: str, data_root: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (data_root / p).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate encoded pointmap mean/std for Wan VAE latents.")
    parser.add_argument("--metadata", type=str, required=True, help="Path to meta.json (jsonl).")
    parser.add_argument("--data_root", type=str, default=".", help="Root for relative paths in metadata.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="pretrained/Wan2.1-I2V-14B-480P-Diffusers-lvp",
        help="Base model path containing VAE subfolder.",
    )
    parser.add_argument("--train_resolution", type=str, default="49x480x480")
    parser.add_argument("--latent_frames", type=int, default=13, help="Number of latent frames to include.")
    parser.add_argument("--max_samples", type=int, default=0, help="If >0, limit to this many samples.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    frames, height, width = parse_resolution(args.train_resolution)
    data_root = Path(args.data_root)
    metadata_path = Path(args.metadata)

    vae = AutoencoderKLWan.from_pretrained(args.model_path, subfolder="vae", torch_dtype=dtype).to(args.device)
    vae.eval()

    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(device=args.device, dtype=dtype)
    )
    latents_std = (
        1.0
        / torch.tensor(vae.config.latents_std)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(device=args.device, dtype=dtype)
    )

    total_sum = torch.zeros((), dtype=torch.float64, device="cpu")
    total_sum_sq = torch.zeros((), dtype=torch.float64, device="cpu")
    total_count = 0

    samples = list(load_metadata(metadata_path))
    if args.max_samples and args.max_samples < len(samples):
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(samples), generator=generator)[: args.max_samples].tolist()
        samples = [samples[i] for i in indices]

    for sample in tqdm(samples, desc="Encoding pointmap videos"):
        xyz_path = sample.get("xyz_video_path") or sample.get("pointmap_video_path")
        if not xyz_path:
            continue
        xyz_path = resolve_path(xyz_path, data_root)
        if not xyz_path.exists():
            continue

        try:
            video = preprocess_video_with_resize(xyz_path, frames, height, width)
        except Exception:
            continue

        # Normalize to [-1, 1] like training
        video = video / 255.0 * 2.0 - 1.0
        video = video.permute(1, 0, 2, 3).unsqueeze(0)
        video = video.to(device=args.device, dtype=dtype)

        with torch.no_grad():
            latent_dist = vae.encode(video).latent_dist
            latent = latent_dist.sample()
            latent = (latent - latents_mean) * latents_std

        latent = latent.squeeze(0)
        latent = latent[:, : min(args.latent_frames, latent.shape[1])]

        latent_cpu = latent.detach().to(dtype=torch.float64, device="cpu")
        total_sum += latent_cpu.sum()
        total_sum_sq += (latent_cpu * latent_cpu).sum()
        total_count += latent_cpu.numel()

        del video, latent, latent_cpu
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    if total_count == 0:
        raise RuntimeError("No valid samples processed. Check metadata paths.")

    mean = total_sum / total_count
    var = total_sum_sq / total_count - mean * mean
    std = torch.sqrt(var)

    print(f"encoded_pm_mean: {mean.item():.6f}")
    print(f"encoded_pm_std:  {std.item():.6f}")


if __name__ == "__main__":
    main()
