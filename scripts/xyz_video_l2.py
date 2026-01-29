import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def _read_video(path: Path, max_frames: int | None) -> np.ndarray:
    reader = imageio.get_reader(str(path))
    frames = []
    try:
        for i, frame in enumerate(reader):
            if max_frames is not None and i >= max_frames:
                break
            frames.append(frame)
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"No frames read from {path}")
    return np.stack(frames, axis=0)


def _write_mask_video(mask: np.ndarray, out_path: Path, fps: int) -> None:
    # mask: [T, H, W] boolean or 0/1
    frames = (mask.astype(np.uint8) * 255)
    frames = np.repeat(frames[..., None], 3, axis=-1)
    imageio.mimwrite(str(out_path), frames, fps=fps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-pixel 3D Euclidean distance between two videos.")
    parser.add_argument("--gt", type=str, required=True, help="Ground-truth video path.")
    parser.add_argument("--pred", type=str, required=True, help="Predicted video path.")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit number of frames to read.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor applied to pixel values.")
    parser.add_argument("--mask_out", type=str, default=None, help="Optional binary mask video output path.")
    parser.add_argument(
        "--mask_percent",
        type=float,
        default=20.0,
        help="Top-percent error pixels to mark as 1 in the mask (per-frame).",
    )
    parser.add_argument(
        "--mask_thresh",
        type=float,
        default=None,
        help="Absolute L2 threshold; if set, use this instead of mask_percent.",
    )
    parser.add_argument("--fps", type=int, default=15, help="FPS for mask video output.")
    args = parser.parse_args()

    gt_path = Path(args.gt)
    pred_path = Path(args.pred)

    gt = _read_video(gt_path, args.max_frames).astype(np.float32) * args.scale
    pred = _read_video(pred_path, args.max_frames).astype(np.float32) * args.scale

    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gt {gt.shape} vs pred {pred.shape}")
    if gt.shape[-1] < 3:
        raise ValueError(f"Expected at least 3 channels, got shape {gt.shape}")

    diff = gt[..., :3] - pred[..., :3]
    per_pixel_l2 = np.sqrt(np.sum(diff * diff, axis=-1))  # [T, H, W]

    per_frame_mean = per_pixel_l2.reshape(per_pixel_l2.shape[0], -1).mean(axis=1)
    mean = per_pixel_l2.mean()
    median = np.median(per_pixel_l2)
    p95 = np.percentile(per_pixel_l2, 95)
    maxv = per_pixel_l2.max()

    print(f"frames: {gt.shape[0]}  size: {gt.shape[2]}x{gt.shape[1]}")
    print(f"mean_l2: {mean:.6f}")
    print(f"median_l2: {median:.6f}")
    print(f"p95_l2: {p95:.6f}")
    print(f"max_l2: {maxv:.6f}")
    print("per_frame_mean_l2:", " ".join(f"{v:.6f}" for v in per_frame_mean))

    if args.mask_out is not None:
        if args.mask_thresh is not None:
            mask = per_pixel_l2 >= args.mask_thresh
        else:
            if not (0.0 < args.mask_percent <= 100.0):
                raise ValueError("mask_percent must be in (0, 100].")
            # Compute per-frame threshold so each frame keeps the top-k% errors.
            thresholds = np.percentile(per_pixel_l2, 100.0 - args.mask_percent, axis=(1, 2))
            mask = per_pixel_l2 >= thresholds[:, None, None]
        _write_mask_video(mask, Path(args.mask_out), args.fps)
        print(f"Saved mask video to: {args.mask_out}")


if __name__ == "__main__":
    main()
