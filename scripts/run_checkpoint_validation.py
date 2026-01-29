import argparse
from pathlib import Path

import torch
from core.finetune.models.utils import get_model_cls
from core.finetune.schemas import Args


def _parse_step(checkpoint_dir: Path) -> int:
    name = checkpoint_dir.name
    if not name.startswith("checkpoint-"):
        raise ValueError(f"Expected a checkpoint dir like 'checkpoint-5100', got: {name}")
    try:
        return int(name.split("-", 1)[1])
    except ValueError as exc:
        raise ValueError(f"Could not parse step from checkpoint dir: {name}") from exc


def _lora_stats(transformer) -> dict:
    lora_norm = 0.0
    lora_params = 0
    for name, param in transformer.named_parameters():
        if "lora_" in name:
            lora_norm += param.detach().float().norm().item()
            lora_params += 1
    emb = getattr(transformer, "learnable_domain_embeddings", None)
    emb_norm = emb.detach().float().norm().item() if emb is not None else float("nan")
    return {
        "lora_param_count": lora_params,
        "lora_norm_sum": lora_norm,
        "domain_emb_norm": emb_norm,
        "active_adapters": getattr(transformer, "active_adapters", None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the trainer's checkpoint validation for a saved checkpoint.")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--validation_dir", type=str, default="results/infer_inputs")
    parser.add_argument("--validation_prompts", type=str, default="prompt.txt")
    parser.add_argument("--validation_images", type=str, default="image.txt")
    parser.add_argument("--validation_xyz_images", type=str, default="xyz_image.txt")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to write validation outputs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--domain_embedding_scale", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    step = _parse_step(checkpoint_dir)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir / "validation_from_script"

    # Build Args directly so we can run the exact trainer code-path without launching training.
    trainer_args = Args(
        model_path=Path(args.base_model),
        model_name="wan-i2v-demb-samerope",
        model_type="wan-i2v",
        training_type="lora",
        output_dir=checkpoint_dir.parent,
        report_to="tensorboard",
        data_root=Path("."),
        caption_column=Path("prompts.txt"),
        image_column=None,
        video_column=Path("videos.txt"),
        pointmap_column=Path("pointmap_videos.txt"),
        raw_metadata=Path("data/lvp/meta.json"),
        raw_data=True,
        dummy_data=False,
        dummy_num_samples=8,
        use_xyz_first_frame=True,
        log_data_paths=False,
        log_data_paths_limit=10,
        resume_from_checkpoint=checkpoint_dir,
        init_lora_path=None,
        domain_embedding_scale=args.domain_embedding_scale,
        init_domain_embeddings_path=None,
        seed=args.seed,
        train_epochs=1,
        train_steps=step + 1,
        checkpointing_steps=100,
        checkpointing_limit=2,
        batch_size=1,
        gradient_accumulation_steps=1,
        train_resolution=(49, 480, 480),
        mixed_precision=args.mixed_precision,
        learning_rate=5e-4,
        optimizer="adamw",
        beta1=0.9,
        beta2=0.95,
        beta3=0.98,
        epsilon=1e-8,
        weight_decay=1e-4,
        max_grad_norm=1.0,
        lr_scheduler="constant_with_warmup",
        lr_warmup_steps=0,
        lr_num_cycles=1,
        lr_power=1.0,
        num_workers=0,
        pin_memory=True,
        xyz_loss_weight=1.0,
        gradient_checkpointing=True,
        enable_slicing=True,
        enable_tiling=True,
        nccl_timeout=1800,
        rank=128,
        lora_alpha=64,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        do_validation=False,
        checkpoint_validation=True,
        validation_steps=None,
        validation_dir=Path(args.validation_dir),
        validation_prompts=args.validation_prompts,
        validation_images=args.validation_images,
        validation_videos=None,
        validation_xyz_images=args.validation_xyz_images,
        gen_fps=15,
    )

    trainer_cls = get_model_cls(trainer_args.model_name, trainer_args.training_type)
    trainer = trainer_cls(trainer_args)

    # Mirror the setup sequence needed for accelerator.load_state to work.
    trainer.check_setting()
    trainer.prepare_models()
    trainer.prepare_dataset()
    trainer.prepare_trainable_parameters()
    trainer.prepare_optimizer()
    trainer.prepare_for_training()
    trainer.prepare_for_validation()

    before = _lora_stats(trainer.components.transformer)
    print("LoRA/domain stats before load_state:", before)

    # We only need model weights for validation; skip optimizer/scheduler state loading.
    trainer.accelerator._optimizers = []
    trainer.accelerator._schedulers = []
    trainer.accelerator.load_state(str(checkpoint_dir))
    trainer.accelerator.wait_for_everyone()

    # learnable_domain_embeddings is saved separately from accelerate state.
    emb_path = checkpoint_dir / "learnable_domain_embeddings.pt"
    if emb_path.exists():
        emb = torch.load(emb_path, map_location="cpu")
        param = getattr(trainer.components.transformer, "learnable_domain_embeddings", None)
        if param is not None:
            param.data = emb.to(param.device, param.dtype)
            print(
                "Loaded learnable_domain_embeddings from checkpoint:",
                str(emb_path),
                "norm=",
                param.detach().float().norm().item(),
            )

    after = _lora_stats(trainer.components.transformer)
    print("LoRA/domain stats after load_state:", after)

    # Call the exact checkpoint validation implementation used during training.
    trainer._Trainer__checkpoint_validate(step, output_dir)
    print(f"Wrote validation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
