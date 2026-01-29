import argparse
import shutil
import subprocess
from pathlib import Path

import torch
from PIL import Image
from diffusers.utils import export_to_video, load_image
from peft import LoraConfig, set_peft_model_state_dict
from transformers import CLIPVisionModel

from core.finetune.datasets.utils import preprocess_image_with_resize
from core.finetune.constants import ENCODED_PM_MEAN, ENCODED_PM_STD
from core.finetune.models.wan_i2v.demb_samerope_trainer import (
    WanSameRopeWBWImageToVideoPipeline,
    WanTransformer3DModelDembSameRope,
)


def _read_nonempty_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _resolve_path(base_dir: Path, entry: str) -> Path:
    entry_path = Path(entry)
    if entry_path.is_absolute():
        return entry_path
    return base_dir / entry


def _load_pipeline(args) -> WanSameRopeWBWImageToVideoPipeline:
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    image_encoder = CLIPVisionModel.from_pretrained(
        args.base_model, subfolder="image_encoder", torch_dtype=torch.float32
    )
    transformer = WanTransformer3DModelDembSameRope.from_pretrained(
        args.base_model, subfolder="transformer", torch_dtype=dtype
    )
    transformer.domain_embedding_scale = float(args.domain_embedding_scale)

    pipe = WanSameRopeWBWImageToVideoPipeline.from_pretrained(
        args.base_model,
        image_encoder=image_encoder,
        transformer=transformer,
        torch_dtype=dtype,
    )

    # PEFT-style LoRA loading to match training.
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights=True,
        target_modules=args.target_modules,
    )
    pipe.transformer.add_adapter(transformer_lora_config)
    lora_state_dict = pipe.__class__.lora_state_dict(args.lora_path)
    transformer_state_dict = {
        k.replace("transformer.", ""): v for k, v in lora_state_dict.items() if k.startswith("transformer.")
    }
    set_peft_model_state_dict(pipe.transformer, transformer_state_dict, adapter_name="default")

    emb_path = Path(args.lora_path) / "learnable_domain_embeddings.pt"
    if emb_path.exists():
        emb = torch.load(emb_path, map_location="cpu")
        pipe.transformer.learnable_domain_embeddings.data = emb.to(
            pipe.transformer.device, pipe.transformer.dtype
        )

    pipe = pipe.to("cuda", dtype=dtype)
    pipe.transformer.eval()
    pipe.text_encoder.eval()
    pipe.image_encoder.eval()
    pipe.vae.eval()
    return pipe


def _crop_and_pad_right(raw_path: Path, out_path: Path, height: int, width: int, crop_size: int) -> None:
    if crop_size > min(height, width):
        raise ValueError(f"crop_size {crop_size} exceeds min(height, width)={min(height, width)}")
    # Right half starts at x=width.
    x0 = width + (width - crop_size) // 2
    y0 = (height - crop_size) // 2
    pad = (width - crop_size) // 2
    if shutil.which("ffmpeg"):
        vf = f"crop={crop_size}:{crop_size}:{x0}:{y0},pad={width}:{height}:{pad}:{pad}"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_path),
            "-vf",
            vf,
            "-c:a",
            "copy",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
        return

    frames = imageio.mimread(str(raw_path))
    cropped = []
    for frame in frames:
        crop = frame[y0 : y0 + crop_size, x0 : x0 + crop_size]
        padded = np.pad(crop, ((pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=0)
        cropped.append(padded)
    imageio.mimwrite(str(out_path), cropped, fps=15)


def _crop_left(raw_path: Path, out_path: Path, height: int, width: int) -> None:
    if shutil.which("ffmpeg"):
        vf = f"crop={width}:{height}:0:0"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_path),
            "-vf",
            vf,
            "-c:a",
            "copy",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
        return

    frames = imageio.mimread(str(raw_path))
    cropped = [frame[:, :width] for frame in frames]
    imageio.mimwrite(str(out_path), cropped, fps=15)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PEFT inference and crop right half to 470 then pad to 480.")
    parser.add_argument("--validation_dir", type=str, default="results/infer_inputs")
    parser.add_argument("--prompts_file", type=str, default="prompt.txt")
    parser.add_argument("--images_file", type=str, default="image.txt")
    parser.add_argument("--xyz_images_file", type=str, default="xyz_image.txt")
    parser.add_argument("--line_index", type=int, default=1, help="1-based line index")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True, help="Final cropped+padded mp4 output path.")
    parser.add_argument("--left_output", type=str, default=None, help="Optional left-half 480x480 mp4 output path.")
    parser.add_argument("--keep_raw", action="store_true", help="Keep the intermediate raw mp4.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--crop_size", type=int, default=470)
    parser.add_argument(
        "--denorm_pointmap",
        action="store_true",
        help="Decode latent and denormalize xyz half using ENCODED_PM_MEAN/STD (outputs 960x480).",
    )
    parser.add_argument("--domain_embedding_scale", type=float, default=1.0)
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["to_q", "to_k", "to_v", "to_out.0"],
    )
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    validation_dir = Path(args.validation_dir)
    prompts = _read_nonempty_lines(validation_dir / args.prompts_file)
    images = _read_nonempty_lines(validation_dir / args.images_file)
    xyz_images = _read_nonempty_lines(validation_dir / args.xyz_images_file)

    idx = args.line_index - 1
    if idx < 0 or idx >= len(prompts):
        raise SystemExit(f"line_index {args.line_index} out of range (prompts={len(prompts)})")
    if idx >= len(images) or idx >= len(xyz_images):
        raise SystemExit(
            f"line_index {args.line_index} out of range (images={len(images)}, xyz_images={len(xyz_images)})"
        )

    prompt = prompts[idx]
    image_path = _resolve_path(validation_dir, images[idx])
    xyz_image_path = _resolve_path(validation_dir, xyz_images[idx])

    pipe = _load_pipeline(args)

    image_tensor = preprocess_image_with_resize(image_path, args.height, args.width)
    image_tensor = image_tensor.to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    image = Image.fromarray(image_tensor)

    pointmap_image = load_image(str(xyz_image_path)).resize((args.width, args.height))
    pointmap_tensor = pipe.video_processor.preprocess(pointmap_image, height=args.height, width=args.width)
    setattr(pipe, "pointmap_image", pointmap_tensor)

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    if args.denorm_pointmap:
        latents = pipe(
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            prompt=prompt,
            image=image,
            generator=generator,
            guidance_scale=args.guidance_scale,
            attention_kwargs={"scale": args.lora_scale},
            output_type="latent",
        ).frames[0]
        if latents.ndim == 4:
            latents = latents.unsqueeze(0)
        pm_mean = torch.tensor(ENCODED_PM_MEAN, device=latents.device, dtype=latents.dtype)
        pm_std = torch.tensor(ENCODED_PM_STD, device=latents.device, dtype=latents.dtype)
        latents[..., latents.shape[-1] // 2 :] = latents[..., latents.shape[-1] // 2 :] * pm_std + pm_mean
        with torch.no_grad():
            latents = latents.to(pipe.vae.dtype)
            latents_mean = (
                torch.tensor(pipe.vae.config.latents_mean)
                .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = pipe.vae.decode(latents, return_dict=False)[0]
            video = pipe.video_processor.postprocess_video(video, output_type="np")[0]
    else:
        video = pipe(
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            prompt=prompt,
            image=image,
            generator=generator,
            guidance_scale=args.guidance_scale,
            attention_kwargs={"scale": args.lora_scale},
        ).frames[0]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_name(f"{output_path.stem}_raw.mp4")
    export_to_video(video, str(raw_path), fps=args.fps)

    _crop_and_pad_right(raw_path, output_path, args.height, args.width, args.crop_size)

    if args.left_output:
        left_path = Path(args.left_output)
        left_path.parent.mkdir(parents=True, exist_ok=True)
        _crop_left(raw_path, left_path, args.height, args.width)

    if not args.keep_raw and raw_path.exists():
        raw_path.unlink()

    print(f"Saved cropped+padded video to: {output_path}")


if __name__ == "__main__":
    main()
