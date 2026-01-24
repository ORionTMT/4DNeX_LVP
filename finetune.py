import os
import sys


def _prepend_ld_library_path(paths):
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    existing_parts = [p for p in existing.split(":") if p]
    new_parts = []
    for path in paths:
        if path and path not in existing_parts and path not in new_parts:
            new_parts.append(path)
    if new_parts:
        os.environ["LD_LIBRARY_PATH"] = ":".join(new_parts + existing_parts)


def _configure_nvrtc_env():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    old_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")

    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    base = os.path.join(conda_prefix, "lib", py_version, "site-packages", "nvidia")

    nv_cu13_lib = os.environ.get("NV_CU13_LIB") or os.path.join(base, "cu13", "lib")
    nv_cudnn_lib = os.environ.get("NV_CUDNN_LIB") or os.path.join(base, "cudnn", "lib")
    nv_cuda_rt_lib = os.environ.get("NV_CUDA_RT_LIB") or os.path.join(base, "cuda_runtime", "lib")
    nv_nvjitlink_lib = os.environ.get("NV_NVCUDA_NVJITLINK_LIB") or os.path.join(base, "nvjitlink", "lib")

    if os.path.isdir(nv_cu13_lib):
        os.environ.setdefault("NV_CU13_LIB", nv_cu13_lib)
    if os.path.isdir(nv_cudnn_lib):
        os.environ.setdefault("NV_CUDNN_LIB", nv_cudnn_lib)
    if os.path.isdir(nv_cuda_rt_lib):
        os.environ.setdefault("NV_CUDA_RT_LIB", nv_cuda_rt_lib)
    if os.path.isdir(nv_nvjitlink_lib):
        os.environ.setdefault("NV_NVCUDA_NVJITLINK_LIB", nv_nvjitlink_lib)

    _prepend_ld_library_path(
        [
            os.environ.get("NV_CU13_LIB"),
            os.environ.get("NV_CUDNN_LIB"),
            os.environ.get("NV_CUDA_RT_LIB"),
            os.environ.get("NV_NVCUDA_NVJITLINK_LIB"),
        ]
    )
    new_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if (
        new_ld_library_path != old_ld_library_path
        and os.environ.get("FINETUNE_NVRTC_REEXEC") != "1"
    ):
        os.environ["FINETUNE_NVRTC_REEXEC"] = "1"
        os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)


_configure_nvrtc_env()


from core.finetune.models.utils import get_model_cls
from core.finetune.schemas import Args


def main():
    args = Args.parse_args()
    trainer_cls = get_model_cls(args.model_name, args.training_type)
    trainer = trainer_cls(args)
    trainer.fit()


if __name__ == "__main__":
    main()
