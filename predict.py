# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
# Model cache configuration
MODEL_CACHE = "weights"
# BASE_URL = "https://weights.replicate.delivery/default/bytedance-bagel/model_cache/"

# Set up environment variables for model caching
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

import sys
import subprocess
import tempfile
import imageio
import torch
import numpy as np
from PIL import Image
from einops import rearrange
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from transformers import WhisperModel, AutoFeatureExtractor
from loguru import logger
from pathlib import Path

import time
from typing import Optional
from cog import BasePredictor, Input, Path as CogPath


def data_preprocess_server(args, image_path, audio_path, prompts, feature_extractor):
    """Data preprocessing function adapted from tool_for_end2end.py"""
    import math
    
    llava_transform = transforms.Compose([
        transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.BILINEAR), 
        transforms.ToTensor(), 
        transforms.Normalize((0.48145466, 0.4578275, 0.4082107), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
    # Generate prompt
    if prompts is None:
        prompts = "Authentic, Realistic, Natural, High-quality, Lens-Fixed." 
    else:
        prompts = "Authentic, Realistic, Natural, High-quality, Lens-Fixed, " + prompts

    fps = 25
    img_size = args.image_size
    ref_image = Image.open(image_path).convert('RGB')
    
    # Resize reference image
    w, h = ref_image.size
    scale = img_size / min(w, h)
    new_w = round(w * scale / 64) * 64
    new_h = round(h * scale / 64) * 64

    if img_size == 704:
        img_size_long = 1216
    if new_w * new_h > img_size * img_size_long:
        scale = math.sqrt(img_size * img_size_long / w / h)
        new_w = round(w * scale / 64) * 64
        new_h = round(h * scale / 64) * 64

    ref_image = ref_image.resize((new_w, new_h), Image.LANCZOS)
    ref_image = np.array(ref_image)
    ref_image = torch.from_numpy(ref_image)
        
    # Import here to avoid issues during setup
    from hymm_sp.data_kits.audio_dataset import get_audio_feature
    audio_input, audio_len = get_audio_feature(feature_extractor, audio_path)
    audio_prompts = audio_input[0]
    
    motion_bucket_id_heads = np.array([25] * 4)
    motion_bucket_id_exps = np.array([30] * 4)
    motion_bucket_id_heads = torch.from_numpy(motion_bucket_id_heads)
    motion_bucket_id_exps = torch.from_numpy(motion_bucket_id_exps)
    fps = torch.from_numpy(np.array(fps))
    
    to_pil = ToPILImage()
    pixel_value_ref = rearrange(ref_image.clone().unsqueeze(0), "b h w c -> b c h w")
    
    pixel_value_ref_llava = [llava_transform(to_pil(image)) for image in pixel_value_ref]
    pixel_value_ref_llava = torch.stack(pixel_value_ref_llava, dim=0)

    batch = {
        "text_prompt": [prompts],
        "audio_path": [audio_path],
        "image_path": [image_path],
        "fps": fps.unsqueeze(0).to(dtype=torch.float16),
        "audio_prompts": audio_prompts.unsqueeze(0).to(dtype=torch.float16),
        "audio_len": [audio_len],
        "motion_bucket_id_exps": motion_bucket_id_exps.unsqueeze(0),
        "motion_bucket_id_heads": motion_bucket_id_heads.unsqueeze(0),
        "pixel_value_ref": pixel_value_ref.unsqueeze(0).to(dtype=torch.float16),
        "pixel_value_ref_llava": pixel_value_ref_llava.unsqueeze(0).to(dtype=torch.float16)
    }

    return batch


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Setup following weights/README.md and scripts pattern"""
        
        print("[+] Setting up HunyuanVideo-Avatar...")
        
        # Set up environment following scripts pattern
        weights_dir = "./weights"
        os.environ["MODEL_BASE"] = weights_dir
        os.environ["PYTHONPATH"] = "./"
        os.environ["DISABLE_SP"] = "1"  # Single GPU mode
        os.environ["CPU_OFFLOAD"] = "1"  # For lower VRAM
        
        # Add current directory to Python path
        sys.path.insert(0, ".")
        
        # Check for specific checkpoint file instead of just weights directory
        checkpoint_path = f"{weights_dir}/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt"
        
        # Download weights following weights/README.md if checkpoint doesn't exist
        if not os.path.exists(checkpoint_path):
            print("[+] Downloading weights using huggingface-cli...")
            os.makedirs(weights_dir, exist_ok=True)
            
            # Install huggingface-hub if not available
            try:
                import huggingface_hub
            except ImportError:
                print("[+] Installing huggingface_hub...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub[cli]"])
            
            # Download the model weights
            subprocess.check_call([
                "huggingface-cli", "download", "tencent/HunyuanVideo-Avatar", 
                "--local-dir", weights_dir
            ])
            print("[+] Weights downloaded successfully!")
        else:
            print("[+] Weights already exist, skipping download")
        
        # Verify checkpoint exists after download
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")
        
        # Configure logger
        logger.remove()
        logger.add(lambda msg: print(msg), level="INFO")
        
        # Import HunyuanVideo components after setting up environment
        from hymm_sp.config import parse_args
        from hymm_sp.sample_inference_audio import HunyuanVideoSampler
        from hymm_sp.data_kits.face_align import AlignImage
        
        # Parse arguments following the scripts pattern (run_single_poor.sh)
        args_list = [
            "--ckpt", checkpoint_path,
            "--sample-n-frames", "129",
            "--seed", "128", 
            "--image-size", "704",
            "--cfg-scale", "7.5",
            "--infer-steps", "50",
            "--use-deepcache", "1",
            "--flow-shift-eval-video", "5.0",
            "--use-fp8",
            "--cpu-offload",
            "--infer-min"
        ]
        
        print("[+] Parsing arguments...")
        # Fix: Pass args_list properly by modifying sys.argv temporarily
        old_argv = sys.argv[:]
        sys.argv = ["predict.py"] + args_list
        self.args = parse_args()
        sys.argv = old_argv
        
        # Verify models root path exists
        models_root_path = Path(self.args.ckpt)
        if not models_root_path.exists():
            raise ValueError(f"Models root not exists: {models_root_path}")

        # Set up device (single GPU as per scripts)
        device = torch.device("cuda")
        self.device = device
        
        print("[+] Loading HunyuanVideoSampler...")
        # Load HunyuanVideoSampler following sample_gpu_poor.py pattern
        self.hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
            self.args.ckpt, args=self.args, device=device
        )
        # Get the updated args
        self.args = self.hunyuan_video_sampler.args
        
        # Apply CPU offloading if enabled (following sample_gpu_poor.py)
        if self.args.cpu_offload:
            print("[+] Applying CPU offloading...")
            from diffusers.hooks import apply_group_offloading
            onload_device = torch.device("cuda")
            apply_group_offloading(
                self.hunyuan_video_sampler.pipeline.transformer, 
                onload_device=onload_device, 
                offload_type="block_level", 
                num_blocks_per_group=1
            )

        print("[+] Loading Whisper model...")
        # Load audio processing models (following scripts pattern)
        MODEL_BASE = os.environ.get('MODEL_BASE')
        self.wav2vec = WhisperModel.from_pretrained(
            f"{MODEL_BASE}/ckpts/whisper-tiny/"
        ).to(device=device, dtype=torch.float32)
        self.wav2vec.requires_grad_(False)
        
        print("[+] Loading face alignment model...")
        # Load face alignment model
        BASE_DIR = f'{MODEL_BASE}/ckpts/det_align/'
        det_path = os.path.join(BASE_DIR, 'detface.pt')
        self.align_instance = AlignImage("cuda", det_path=det_path)
        
        # Load feature extractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            f"{MODEL_BASE}/ckpts/whisper-tiny/"
        )

        print("[+] Setup completed successfully!")

    def predict(
        self,
        image: CogPath = Input(description="Reference image for lip syncing"),
        audio: CogPath = Input(description="Audio file to drive the lip sync"),
        prompt: str = Input(
            description="Text prompt to guide the generation",
            default="a person is speaking naturally"
        ),
        fps: int = Input(description="Output video FPS", default=25, ge=20, le=30),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=20, le=100, default=50
        ),
        guidance_scale: float = Input(
            description="Guidance scale for generation", ge=1.0, le=20.0, default=7.5
        ),
        seed: Optional[int] = Input(
            description="Random seed for reproducible results", default=None
        ),
    ) -> CogPath:
        """Generate a lip-synced video following sample_gpu_poor.py pattern"""
        
        # Set seed if provided
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        
        # Update args with user inputs
        self.args.infer_steps = num_inference_steps
        self.args.cfg_scale = guidance_scale
        self.args.seed = seed

        # Save inputs to temporary files
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
            img = Image.open(image).convert("RGB")
            img.save(tmp_img.name)
            image_path = tmp_img.name
            
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            # Copy audio file
            with open(audio, "rb") as src:
                tmp_audio.write(src.read())
            audio_path = tmp_audio.name

        try:
            print("[+] Preprocessing inputs...")
            # Preprocess data
            batch = data_preprocess_server(
                self.args, image_path, audio_path, prompt, self.feature_extractor
            )

            # Apply infer_min setting from scripts (following sample_gpu_poor.py)
            if self.args.infer_min:
                batch["audio_len"][0] = 129

            print("[+] Running inference...")
            # Run inference following sample_gpu_poor.py pattern exactly
            with torch.no_grad():
                samples = self.hunyuan_video_sampler.predict(
                    self.args, batch, self.wav2vec, self.feature_extractor, self.align_instance
                )

            print("[+] Processing output...")
            # Process output following sample_gpu_poor.py exactly
            sample = samples['samples'][0].unsqueeze(0)  # denoised latent, (bs, 16, t//4, h//8, w//8)
            sample = sample[:, :, :batch["audio_len"][0]]
            
            # Convert latent to video frames (exact sample_gpu_poor.py pattern)
            video = rearrange(sample[0], "c f h w -> f h w c")
            video = (video * 255.).data.cpu().numpy().astype(np.uint8)  # (f h w c)
            
            torch.cuda.empty_cache()

            # Prepare final frames (exact sample_gpu_poor.py pattern)
            final_frames = []
            for frame in video:
                final_frames.append(frame)
            final_frames = np.stack(final_frames, axis=0)
            
            # Save video without audio first (following sample_gpu_poor.py)
            output_path = "/tmp/output.mp4"
            imageio.mimsave(output_path, final_frames, fps=fps)
            
            # Add audio to video using ffmpeg (exact sample_gpu_poor.py pattern)
            output_with_audio = "/tmp/output_with_audio.mp4"
            os.system(f"ffmpeg -i '{output_path}' -i '{audio_path}' -shortest '{output_with_audio}' -y -loglevel quiet; rm '{output_path}'")
            
            return CogPath(output_with_audio)
            
        finally:
            # Clean up temporary files
            try:
                os.unlink(image_path)
                os.unlink(audio_path)
            except:
                pass
