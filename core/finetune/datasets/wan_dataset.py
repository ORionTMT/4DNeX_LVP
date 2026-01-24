import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import torch
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override
import PIL

from core.finetune.constants import ENCODED_PM_MEAN, ENCODED_PM_STD, LOG_LEVEL, LOG_NAME

from .utils import (
    load_images,
    load_images_from_videos,
    load_raw_metadata,
    load_prompts,
    load_videos,
    preprocess_image_with_resize,
    preprocess_video_with_buckets,
    preprocess_video_with_resize,
    generate_uniform_pointmap,
)


if TYPE_CHECKING:
    from core.finetune.trainer import Trainer

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(LOG_NAME, LOG_LEVEL)


class BaseWanDataset(Dataset):
    """
    Base dataset class for Image-to-Video (I2V) training.

    This dataset loads prompts, videos and corresponding conditioning images for I2V training.

    Args:
        data_root (str): Root directory containing the dataset files
        caption_column (str): Path to file containing text prompts/captions
        video_column (str): Path to file containing video paths
        image_column (str): Path to file containing image paths
        device (torch.device): Device to load the data on
        encode_video_fn (Callable[[torch.Tensor], torch.Tensor], optional): Function to encode videos
    """

    def __init__(
        self,
        data_root: str,
        caption_column: str,
        video_column: str,
        image_column: str | None,
        device: torch.device,
        trainer: "Trainer" = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        self.trainer = trainer
        self.device = device
        self.encode_video = trainer.encode_video
        self.encode_text = trainer.encode_text
        self.encode_image = getattr(trainer, "encode_image", None)
        self._dummy_data = bool(trainer and getattr(trainer.args, "dummy_data", False))
        self._raw_data = bool(trainer and getattr(trainer.args, "raw_data", False))
        self._raw_samples: List[Dict[str, Any]] = []
        if self._dummy_data:
            self._dummy_num_samples = max(1, int(getattr(trainer.args, "dummy_num_samples", 8)))
            self.prompts = ["dummy prompt"] * self._dummy_num_samples
            self.videos = [Path(f"dummy_{i}.mp4") for i in range(self._dummy_num_samples)]
            self.images = [Path(f"dummy_{i}.png") for i in range(self._dummy_num_samples)]
        else:
            data_root = Path(data_root)
            if self._raw_data and getattr(self.trainer.args, "raw_metadata", None):
                raw_metadata_path = Path(self.trainer.args.raw_metadata)
                if not raw_metadata_path.is_absolute():
                    # Prefer explicit path if it exists relative to cwd, otherwise resolve under data_root.
                    if not raw_metadata_path.exists():
                        raw_metadata_path = data_root / raw_metadata_path
                self._raw_samples = load_raw_metadata(raw_metadata_path)
                self.prompts = [sample["caption"] for sample in self._raw_samples]
                self.videos = [self._resolve_sample_path(data_root, sample["rgb_video_path"]) for sample in self._raw_samples]
                self.pointmap_videos = [
                    self._resolve_sample_path(data_root, sample["xyz_video_path"]) for sample in self._raw_samples
                ]
                self.images = []
            else:
                self.prompts = load_prompts(data_root / caption_column)
                self.videos = load_videos(data_root / video_column)
                if self._raw_data:
                    pointmap_column = getattr(self.trainer.args, "pointmap_column", None)
                    if pointmap_column is None:
                        raise ValueError("pointmap_column must be specified when raw_data is enabled")
                    self.pointmap_videos = load_videos(data_root / pointmap_column)
                    self.images = []
                elif image_column is not None:
                    self.images = load_images(data_root / image_column)
                else:
                    self.images = load_images_from_videos(self.videos)

        uniform_pointmap = torch.from_numpy(
            generate_uniform_pointmap(self.trainer.args.train_resolution[1], self.trainer.args.train_resolution[2])
        ).permute(2, 0, 1)
        self.uniform_pointmap = uniform_pointmap * 2 - 1
        self.use_xyz_first_frame = getattr(self.trainer.args, "use_xyz_first_frame", False)
        self.encoded_pm_mean = ENCODED_PM_MEAN
        self.encoded_pm_std = ENCODED_PM_STD
        self.log_data_paths = getattr(self.trainer.args, "log_data_paths", False)
        self.log_data_paths_limit = getattr(self.trainer.args, "log_data_paths_limit", 10)
        self._log_data_paths_count = 0

        if self._dummy_data:
            self._dummy_shapes = self._compute_dummy_shapes()
            return

        # Check if number of prompts matches number of videos and images
        if self._raw_data:
            if not (len(self.videos) == len(self.prompts) == len(self.pointmap_videos)):
                raise ValueError(
                    "Expected length of prompts, rgb videos and pointmap videos to be the same but found "
                    f"{len(self.prompts)}, {len(self.videos)} and {len(self.pointmap_videos)}."
                )
        elif not (len(self.videos) == len(self.prompts) == len(self.images)):
            raise ValueError(
                f"Expected length of prompts, videos and images to be the same but found {len(self.prompts)}, {len(self.videos)} and {len(self.images)}. Please ensure that the number of caption prompts, videos and images match in your dataset."
            )

        # Check if all video files exist
        if any(not path.is_file() for path in self.videos):
            raise ValueError(
                f"Some video files were not found. Please ensure that all video files exist in the dataset directory. Missing file: {next(path for path in self.videos if not path.is_file())}"
            )

        if self._raw_data:
            if any(not path.is_file() for path in self.pointmap_videos):
                raise ValueError(
                    "Some pointmap video files were not found. Please ensure that all pointmap video files exist in the dataset directory. "
                    f"Missing file: {next(path for path in self.pointmap_videos if not path.is_file())}"
                )
        else:
            # Check if all image files exist
            if any(not path.is_file() for path in self.images):
                raise ValueError(
                    f"Some image files were not found. Please ensure that all image files exist in the dataset directory. Missing file: {next(path for path in self.images if not path.is_file())}"
                )

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self._dummy_data:
            return self._dummy_getitem(index)
        if self._raw_data:
            return self._raw_getitem(index)
        while True:
            try:
                ret = self.getitem(index)
                if ret['video_metadata']['num_frames'] != 13:
                    raise ValueError("Not enough frames")
                break
            except Exception as e:
                # print(e)
                index = (index + 1) % len(self.videos)
        return ret

    def _compute_dummy_shapes(self) -> Dict[str, int]:
        frames, height, width = self.trainer.args.train_resolution
        vae_config = self.trainer.components.vae.config
        temporal_downsample = getattr(vae_config, "temperal_downsample", None)
        if temporal_downsample is None:
            vae_scale_factor_temporal = 4
        else:
            vae_scale_factor_temporal = 2 ** sum(1 for v in temporal_downsample if v)

        dim_mult = getattr(vae_config, "dim_mult", None)
        if dim_mult:
            vae_scale_factor_spatial = 2 ** (len(dim_mult) - 1)
        else:
            vae_scale_factor_spatial = 8

        num_latent_frames = (frames - 1) // vae_scale_factor_temporal + 1
        num_latent_frames = min(num_latent_frames, 13)
        latent_height = height // vae_scale_factor_spatial
        latent_width = width * 2 // vae_scale_factor_spatial

        num_channels_latents = int(getattr(vae_config, "z_dim", 16))

        text_config = self.trainer.components.text_encoder.config
        text_hidden_size = int(getattr(text_config, "d_model", getattr(text_config, "hidden_size", 4096)))
        prompt_seq_len = 512

        image_config = self.trainer.components.image_encoder.config
        image_hidden_size = int(getattr(image_config, "hidden_size", 1280))
        image_size = getattr(image_config, "image_size", 224)
        if isinstance(image_size, (list, tuple)):
            image_size = image_size[0]
        patch_size = getattr(image_config, "patch_size", 14)
        if isinstance(patch_size, (list, tuple)):
            patch_size = patch_size[0]
        if patch_size <= 0:
            patch_size = 14
        num_image_tokens = (image_size // patch_size) ** 2 + 1

        return {
            "frames": frames,
            "height": height,
            "width": width,
            "latent_channels": num_channels_latents,
            "latent_frames": num_latent_frames,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "prompt_seq_len": prompt_seq_len,
            "text_hidden_size": text_hidden_size,
            "image_seq_len": num_image_tokens,
            "image_hidden_size": image_hidden_size,
        }

    def _dummy_getitem(self, index: int) -> Dict[str, Any]:
        shapes = self._dummy_shapes
        encoded_video = torch.randn(
            shapes["latent_channels"],
            shapes["latent_frames"],
            shapes["latent_height"],
            shapes["latent_width"],
        )
        prompt_embedding = torch.randn(shapes["prompt_seq_len"], shapes["text_hidden_size"])
        image_embedding = torch.randn(shapes["image_seq_len"], shapes["image_hidden_size"])

        image = torch.rand(3, shapes["height"], shapes["width"]) * 2 - 1
        if self.use_xyz_first_frame:
            image_pm = torch.rand(3, shapes["height"], shapes["width"]) * 2 - 1
            image = torch.concat([image, image_pm], dim=-1)
        else:
            image = torch.concat([image, self.uniform_pointmap], dim=-1)

        return {
            "image": image,
            "prompt_embedding": prompt_embedding,
            "encoded_video": encoded_video,
            "image_embedding": image_embedding,
            "video_metadata": {
                "num_frames": shapes["latent_frames"],
                "height": shapes["latent_height"],
                "width": shapes["latent_width"],
            },
        }

    def _resolve_sample_path(self, base_dir: Path, path_str: str) -> Path:
        path = Path(path_str)
        if not path.is_absolute():
            path = base_dir / path
        return path

    def _raw_getitem(self, index: int) -> Dict[str, Any]:
        if self.encode_image is None:
            raise RuntimeError("encode_image is not available for raw_data mode")

        if self._raw_samples:
            sample = self._raw_samples[index]
            prompt = sample["caption"]
            video = self.videos[index]
            pointmap_video = self.pointmap_videos[index]
        else:
            prompt = self.prompts[index]
            video = self.videos[index]
            pointmap_video = self.pointmap_videos[index]
        if (
            self.log_data_paths
            and self._log_data_paths_count < self.log_data_paths_limit
            and self.trainer.accelerator.is_main_process
        ):
            logger.info(f"raw sample {index}: rgb={video} xyz={pointmap_video}")
            self._log_data_paths_count += 1

        # HACK: add suffix prompt
        suffix = "POINTMAP_STYLE."
        prompt = prompt + " " + suffix

        cache_dir = self.trainer.args.data_root / "cache"
        prompt_embeddings_dir = cache_dir / "prompt_embeddings"
        prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)

        prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
        prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")

        if prompt_embedding_path.exists():
            prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
        else:
            with torch.no_grad():
                prompt_embedding = self.encode_text(prompt)
            prompt_embedding_cpu = prompt_embedding[0].to("cpu")
            save_file({"prompt_embedding": prompt_embedding_cpu}, prompt_embedding_path)
            prompt_embedding = prompt_embedding_cpu

        # Load raw videos and normalize to [-1, 1]
        frames_rgb, _ = self.preprocess(video, None)
        if frames_rgb is None:
            raise RuntimeError(f"Failed to load rgb video: {video}")
        first_frame_tensor = frames_rgb[0]
        frames_rgb = self.video_transform(frames_rgb)
        rgb_video = frames_rgb.permute(1, 0, 2, 3).unsqueeze(0)

        frames_pm, _ = self.preprocess(pointmap_video, None)
        if frames_pm is None:
            raise RuntimeError(f"Failed to load xyz video: {pointmap_video}")
        first_frame_pm = frames_pm[0]
        frames_pm = self.video_transform(frames_pm)
        pm_video = frames_pm.permute(1, 0, 2, 3).unsqueeze(0)

        with torch.no_grad():
            encoded_video = self.encode_video(rgb_video)[0]
            encoded_pm = self.encode_video(pm_video)[0]

        # Encode first frame to image embedding
        first_frame_pil = first_frame_tensor.permute(1, 2, 0).cpu().numpy()
        first_frame_pil = first_frame_pil.clip(0, 255).astype("uint8")
        first_frame_pil = PIL.Image.fromarray(first_frame_pil, mode="RGB")
        with torch.no_grad():
            image_embedding = self.encode_image(first_frame_pil)[0].to("cpu")

        image = self.image_transform(first_frame_tensor)
        if self.use_xyz_first_frame:
            image_pm = self.image_transform(first_frame_pm)
            image = torch.concat([image, image_pm], -1)
        else:
            image = torch.concat([image, self.uniform_pointmap], -1)

        encoded_pm = (encoded_pm - self.encoded_pm_mean) / self.encoded_pm_std

        # HACK: train on the first 49 frames
        encoded_video = torch.concat([encoded_video[:, :13, :, :], encoded_pm[:, :13, :, :]], -1)
        encoded_video = encoded_video.to("cpu")

        return {
            "image": image,
            "prompt_embedding": prompt_embedding,
            "encoded_video": encoded_video,
            "image_embedding": image_embedding,
            "video_metadata": {
                "num_frames": encoded_video.shape[1],
                "height": encoded_video.shape[2],
                "width": encoded_video.shape[3],
            },
        }

    def getitem(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            return index

        prompt = self.prompts[index]
        # HACK: add suffix prompt
        suffix = 'POINTMAP_STYLE.'
        prompt = prompt + ' ' + suffix
        video = self.videos[index]
        image = self.images[index]
        # Use the correct resolution string for WAN dataset
        train_resolution = self.trainer.args.train_resolution
        if isinstance(train_resolution, (list, tuple)):
            train_resolution_str = f"{train_resolution[0]}x{train_resolution[1]}x{train_resolution[2]}"
        else:
            train_resolution_str = str(train_resolution)
        cache_dir = self.trainer.args.data_root / "cache"
        video_latent_dir = cache_dir / "video_latent" / "wan-i2v" / train_resolution_str
        prompt_embeddings_dir = cache_dir / "prompt_embeddings"
        video_latent_dir.mkdir(parents=True, exist_ok=True)
        prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)

        prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
        prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")
        video_name = video.stem
        encoded_video_path = video_latent_dir / (video_name + ".safetensors")
        pm_latent_dir = self.trainer.args.data_root / "pointmap_latents"
        encoded_pm_path = pm_latent_dir / (video_name + ".pt")

        # Load prompt embedding
        if prompt_embedding_path.exists():
            prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
            logger.debug(
                f"process {self.trainer.accelerator.process_index}: Loaded prompt embedding from {prompt_embedding_path}",
                main_process_only=False,
            )
        else:
            prompt_embedding = self.encode_text(prompt)
            prompt_embedding = prompt_embedding.to("cpu")
            # [1, seq_len, hidden_size] -> [seq_len, hidden_size]
            prompt_embedding = prompt_embedding[0]
            save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)
            logger.info(f"Saved prompt embedding to {prompt_embedding_path}", main_process_only=False)

        # Load encoded video and image embedding
        if encoded_video_path.exists():
            loaded = load_file(encoded_video_path)
            encoded_video = loaded["encoded_video"]
            image_embedding = loaded["image_embedding"]
            logger.debug(f"Loaded encoded video and image embedding from {encoded_video_path}", main_process_only=False)
        else:
            raise FileNotFoundError(f"Encoded video file not found: {encoded_video_path}")

        # Load encoded pointmap
        if encoded_pm_path.exists():
            encoded_pm = torch.load(encoded_pm_path, map_location='cpu')
            logger.debug(f"Loaded encoded point map from {encoded_pm_path}", main_process_only=False)
        else:
            raise FileNotFoundError(f"Encoded pointmap file not found: {encoded_pm_path}")

        # Load first frame image
        _, image = self.preprocess(None, self.images[index])    # resize image
        image = self.image_transform(image)

        encoded_pm = (encoded_pm - self.encoded_pm_mean) / self.encoded_pm_std
        # HACK: train on the first 49 frames
        encoded_video = torch.concat([encoded_video[:, :13, :, :], encoded_pm[:, :13, :, :]], -1)
        image = torch.concat([image, self.uniform_pointmap], -1)

        return {
            "image": image,
            "prompt_embedding": prompt_embedding,
            "encoded_video": encoded_video,
            "image_embedding": image_embedding,
            "video_metadata": {
                "num_frames": encoded_video.shape[1],
                "height": encoded_video.shape[2],
                "width": encoded_video.shape[3],
            },
        }

    def preprocess(self, video_path: Path | None, image_path: Path | None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads and preprocesses a video and an image.
        If either path is None, no preprocessing will be done for that input.

        Args:
            video_path: Path to the video file to load
            image_path: Path to the image file to load

        Returns:
            A tuple containing:
                - video(torch.Tensor) of shape [F, C, H, W] where F is number of frames,
                  C is number of channels, H is height and W is width
                - image(torch.Tensor) of shape [C, H, W]
        """
        raise NotImplementedError("Subclass must implement this method")

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to a video.

        Args:
            frames (torch.Tensor): A 4D tensor representing a video
                with shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed video tensor
        """
        raise NotImplementedError("Subclass must implement this method")

    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to an image.

        Args:
            image (torch.Tensor): A 3D tensor representing an image
                with shape [C, H, W] where:
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed image tensor
        """
        raise NotImplementedError("Subclass must implement this method")


class WanI2VDatasetWithResize(BaseWanDataset):
    """
    A dataset class for image-to-video generation that resizes inputs to fixed dimensions.

    This class preprocesses videos and images by resizing them to specified dimensions:
    - Videos are resized to max_num_frames x height x width
    - Images are resized to height x width

    Args:
        max_num_frames (int): Maximum number of frames to extract from videos
        height (int): Target height for resizing videos and images
        width (int): Target width for resizing videos and images
    """

    def __init__(self, max_num_frames: int, height: int, width: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width

        self.__frame_transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])
        self.__image_transforms = self.__frame_transforms

    @override
    def preprocess(self, video_path: Path | None, image_path: Path | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if video_path is not None:
            video = preprocess_video_with_resize(video_path, self.max_num_frames, self.height, self.width)
        else:
            video = None
        if image_path is not None:
            image = preprocess_image_with_resize(image_path, self.height, self.width)
        else:
            image = None
        return video, image

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    @override
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)
