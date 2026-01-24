# save as convert_ckpt_to_diffusers.py and run it
import torch
from pathlib import Path
from diffusers import WanTransformer3DModel

ckpt_path = "/vast/projects/jgu32/lab/mutian/large-video-planner/data/ckpts/lvp_14B.ckpt"
base = "pretrained/Wan2.1-I2V-14B-480P-Diffusers"
out = "pretrained/Wan2.1-I2V-14B-480P-Diffusers-lvp"

ckpt = torch.load(ckpt_path, map_location="cpu")
state = ckpt.get("state_dict", ckpt)
state = {k: v for k, v in state.items() if k.startswith("model.")}
state = {k.replace("model._orig_mod.", "").replace("model.", "", 1): v for k, v in state.items()}

model = WanTransformer3DModel.from_pretrained(base, subfolder="transformer")
model_state = model.state_dict()

mapped = {}


def copy_param(src_key, dst_key):
    if src_key in state:
        mapped[dst_key] = state[src_key]


def copy_linear(src_prefix, dst_prefix):
    copy_param(f"{src_prefix}.weight", f"{dst_prefix}.weight")
    copy_param(f"{src_prefix}.bias", f"{dst_prefix}.bias")


def copy_norm(src_prefix, dst_prefix):
    copy_param(f"{src_prefix}.weight", f"{dst_prefix}.weight")
    copy_param(f"{src_prefix}.bias", f"{dst_prefix}.bias")


# Patch embedding
copy_linear("patch_embedding", "patch_embedding")

# Condition embeddings
copy_linear("time_embedding.0", "condition_embedder.time_embedder.linear_1")
copy_linear("time_embedding.2", "condition_embedder.time_embedder.linear_2")
copy_linear("time_projection.1", "condition_embedder.time_proj")
copy_linear("text_embedding.0", "condition_embedder.text_embedder.linear_1")
copy_linear("text_embedding.2", "condition_embedder.text_embedder.linear_2")

# Image embedding (I2V only)
copy_norm("img_emb.proj.0", "condition_embedder.image_embedder.norm1")
copy_linear("img_emb.proj.1", "condition_embedder.image_embedder.ff.net.0.proj")
copy_linear("img_emb.proj.3", "condition_embedder.image_embedder.ff.net.2")
copy_norm("img_emb.proj.4", "condition_embedder.image_embedder.norm2")

# Transformer blocks
num_layers = model.config.num_layers
for i in range(num_layers):
    src = f"blocks.{i}"
    dst = f"blocks.{i}"

    # Self-attention
    copy_linear(f"{src}.self_attn.q", f"{dst}.attn1.to_q")
    copy_linear(f"{src}.self_attn.k", f"{dst}.attn1.to_k")
    copy_linear(f"{src}.self_attn.v", f"{dst}.attn1.to_v")
    copy_linear(f"{src}.self_attn.o", f"{dst}.attn1.to_out.0")
    copy_param(f"{src}.self_attn.norm_q.weight", f"{dst}.attn1.norm_q.weight")
    copy_param(f"{src}.self_attn.norm_k.weight", f"{dst}.attn1.norm_k.weight")

    # Cross-attention
    copy_linear(f"{src}.cross_attn.q", f"{dst}.attn2.to_q")
    copy_linear(f"{src}.cross_attn.k", f"{dst}.attn2.to_k")
    copy_linear(f"{src}.cross_attn.v", f"{dst}.attn2.to_v")
    copy_linear(f"{src}.cross_attn.o", f"{dst}.attn2.to_out.0")
    copy_param(f"{src}.cross_attn.norm_q.weight", f"{dst}.attn2.norm_q.weight")
    copy_param(f"{src}.cross_attn.norm_k.weight", f"{dst}.attn2.norm_k.weight")
    copy_linear(f"{src}.cross_attn.k_img", f"{dst}.attn2.add_k_proj")
    copy_linear(f"{src}.cross_attn.v_img", f"{dst}.attn2.add_v_proj")
    copy_param(f"{src}.cross_attn.norm_k_img.weight", f"{dst}.attn2.norm_added_k.weight")

    # Norms (LVP norm3 == diffusers norm2)
    copy_norm(f"{src}.norm3", f"{dst}.norm2")

    # FFN
    copy_linear(f"{src}.ffn.0", f"{dst}.ffn.net.0.proj")
    copy_linear(f"{src}.ffn.2", f"{dst}.ffn.net.2")

    # Modulation -> scale_shift_table
    copy_param(f"{src}.modulation", f"{dst}.scale_shift_table")

# Output projection + scale shift
copy_linear("head.head", "proj_out")
copy_param("head.modulation", "scale_shift_table")

# Filter mapped keys by shape and existence
filtered = {}
skipped_shape = []
skipped_missing = []
for k, v in mapped.items():
    if k not in model_state:
        skipped_missing.append(k)
        continue
    if model_state[k].shape != v.shape:
        skipped_shape.append((k, tuple(v.shape), tuple(model_state[k].shape)))
        continue
    filtered[k] = v

missing, unexpected = model.load_state_dict(filtered, strict=False)
print(
    f"mapped {len(mapped)} keys, loaded {len(filtered)}; "
    f"missing {len(missing)}, unexpected {len(unexpected)}, "
    f"skipped_missing {len(skipped_missing)}, skipped_shape {len(skipped_shape)}"
)

del ckpt, state, mapped, filtered, model_state
import gc
gc.collect()

Path(out).mkdir(parents=True, exist_ok=True)
model.save_pretrained(Path(out) / "transformer", safe_serialization=True)

# copy other components from base
import shutil
for name in ["vae","text_encoder","tokenizer","image_encoder","image_processor","scheduler","model_index.json"]:
    src = Path(base) / name
    dst = Path(out) / name
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

print("saved to", out)
