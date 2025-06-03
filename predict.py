# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import sys
import subprocess
import tempfile
import imageio
import torch
import numpy as np
import math
from PIL import Image
from einops import rearrange
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from transformers import WhisperModel, AutoFeatureExtractor
from loguru import logger
import time
from typing import Optional
from cog import BasePredictor, Input, Path as CogPath

# Environment setup for single GPU
os.environ["MODEL_BASE"] = "./weights"
os.environ["PYTHONPATH"] = "./"
os.environ["DISABLE_SP"] = "1"
os.environ["HF_HOME"] = "./weights"
os.environ["TORCH_HOME"] = "./weights"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def data_preprocess_server(args, image_path, audio_path, prompts, feature_extractor):
    """Exact copy from tool_for_end2end.py"""
    from hymm_sp.data_kits.audio_dataset import get_audio_feature
    
    llava_transform = transforms.Compose(
            [
                transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.BILINEAR), 
                transforms.ToTensor(), 
                transforms.Normalize((0.48145466, 0.4578275, 0.4082107), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
    
    """ 生成prompt """
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
        
    audio_input, audio_len = get_audio_feature(feature_extractor, audio_path)
    audio_prompts = audio_input[0]
    
    motion_bucket_id_heads = np.array([25] * 4)
    motion_bucket_id_exps = np.array([30] * 4)
    motion_bucket_id_heads = torch.from_numpy(motion_bucket_id_heads)
    motion_bucket_id_exps = torch.from_numpy(motion_bucket_id_exps)
    fps = torch.from_numpy(np.array(fps))
    
    to_pil = ToPILImage()
    pixel_value_ref = rearrange(ref_image.clone().unsqueeze(0), "b h w c -> b c h w")   # (b c h w)
    
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
        """Setup for single GPU like sample_gpu_poor.py"""
        
        print("[+] Setting up HunyuanVideo-Avatar...")
        
        # Add current directory to Python path first
        sys.path.insert(0, ".")
        
        # Import after setting up environment
        from hymm_sp.config import parse_args
        from hymm_sp.sample_inference_audio import HunyuanVideoSampler
        from hymm_sp.data_kits.face_align import AlignImage
        
        # Check for fp8 checkpoint
        weights_dir = "./weights"
        checkpoint_path = f"{weights_dir}/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt"
        fp8_checkpoint_path = f"{weights_dir}/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt"
        
        if os.path.exists(fp8_checkpoint_path):
            actual_checkpoint = fp8_checkpoint_path
            use_fp8 = True
            print("[+] Using fp8 checkpoint")
        elif os.path.exists(checkpoint_path):
            actual_checkpoint = checkpoint_path
            use_fp8 = False
            print("[+] Using full precision checkpoint")
        else:
            print("[+] Downloading weights...")
            os.makedirs(weights_dir, exist_ok=True)
            subprocess.check_call([
                "huggingface-cli", "download", "tencent/HunyuanVideo-Avatar", 
                "--local-dir", weights_dir
            ])
            if os.path.exists(fp8_checkpoint_path):
                actual_checkpoint = fp8_checkpoint_path
                use_fp8 = True
            else:
                actual_checkpoint = checkpoint_path
                use_fp8 = False
        
        # Parse arguments for single GPU
        args_list = [
            "--ckpt", actual_checkpoint,
            "--sample-n-frames", "129",
            "--seed", "128", 
            "--image-size", "704",
            "--cfg-scale", "7.5",
            "--infer-steps", "50",
            "--use-deepcache", "1",
            "--flow-shift-eval-video", "5.0"
        ]
        
        if use_fp8:
            args_list.extend(["--use-fp8"])
        
        print(f"[+] Parsing arguments...")
        old_argv = sys.argv[:]
        sys.argv = ["predict.py"] + args_list
        self.args = parse_args()
        sys.argv = old_argv
        
        # Single GPU setup like sample_gpu_poor.py
        self.rank = 0
        self.device = torch.device("cuda")
        
        # Verify checkpoint exists
        models_root_path = CogPath(self.args.ckpt)
        if not models_root_path.exists():
            raise ValueError(f"Models root not exists: {models_root_path}")
        
        print(f"[+] Loading HunyuanVideoSampler...")
        # Load model like sample_gpu_poor.py
        self.hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
            self.args.ckpt, args=self.args, device=self.device
        )
        # Get the updated args
        self.args = self.hunyuan_video_sampler.args

        print(f"[+] Loading additional models...")
        # Load exactly like sample_gpu_poor.py
        MODEL_OUTPUT_PATH = os.environ.get('MODEL_BASE')
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(f"{MODEL_OUTPUT_PATH}/ckpts/whisper-tiny/")
        self.wav2vec = WhisperModel.from_pretrained(f"{MODEL_OUTPUT_PATH}/ckpts/whisper-tiny/").to(device=self.device, dtype=torch.float32)
        self.wav2vec.requires_grad_(False)

        BASE_DIR = f'{MODEL_OUTPUT_PATH}/ckpts/det_align/'
        det_path = os.path.join(BASE_DIR, 'detface.pt')    
        self.align_instance = AlignImage("cuda", det_path=det_path)

        print(f"[+] Setup completed successfully!")

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
            description="Number of denoising steps", 
            ge=1, le=100, default=50
        ),
        guidance_scale: float = Input(
            description="Guidance scale for generation", ge=1.0, le=20.0, default=7.5
        ),
        seed: Optional[int] = Input(
            description="Random seed for reproducible results", default=None
        ),
        image_size: int = Input(
            description="Output image size", 
            default=704,
            ge=512, le=704
        ),
    ) -> CogPath:
        """Generate video using single GPU setup with flask_audio.py tensor processing"""
        
        # Clear cache before starting
        torch.cuda.empty_cache()
        
        # Set seed if provided
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        # Update args with user inputs
        self.args.infer_steps = num_inference_steps
        self.args.cfg_scale = guidance_scale
        self.args.seed = seed
        self.args.image_size = image_size

        # Save inputs to temporary files
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
            img = Image.open(image).convert("RGB")
            img.save(tmp_img.name)
            image_path = tmp_img.name
            
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            with open(audio, "rb") as src:
                tmp_audio.write(src.read())
            audio_path = tmp_audio.name

        print("[+] Preprocessing inputs...")
        
        # Preprocess data like sample_gpu_poor.py
        batch = data_preprocess_server(
            self.args, image_path, audio_path, prompt, self.feature_extractor
        )

        print(f"[+] Generation settings:")
        print(f"    - Steps: {num_inference_steps}")
        print(f"    - Image size: {image_size}")
        print(f"    - Audio frames: {batch['audio_len'][0]}")

        print("[+] Running inference...")
        
        # Run inference like sample_gpu_poor.py
        samples = self.hunyuan_video_sampler.predict(
            self.args, batch, self.wav2vec, self.feature_extractor, self.align_instance
        )

        print("[+] Processing output...")
        
        # Use exact flask_audio.py tensor processing pattern - this is the key fix
        sample = samples['samples'][0].unsqueeze(0)
        sample = sample[:, :, :batch["audio_len"][0]]
        
        # This is the crucial difference - using flask_audio.py's exact pattern
        video = sample[0].permute(1, 2, 3, 0).clamp(0, 1).numpy()
        video = (video * 255.).astype(np.uint8)

        torch.cuda.empty_cache()

        # Save video exactly like tool_for_end2end.py
        import uuid
        TEMP_DIR = "/tmp"
        
        uuid_string = str(uuid.uuid4())
        temp_video_path = f'{TEMP_DIR}/{uuid_string}.mp4'
        imageio.mimsave(temp_video_path, video, fps=fps)

        # Add audio
        output_path = temp_video_path
        save_path = temp_video_path.replace(".mp4", "_audio.mp4")
        print('='*100)
        print(f"output_path = {output_path}\n audio_path = {audio_path}\n save_path = {save_path}")
        os.system(f"ffmpeg -i '{output_path}' -i '{audio_path}' -shortest '{save_path}' -y -loglevel quiet; rm '{output_path}'")
        
        # Clean up temporary files
        os.unlink(image_path)
        os.unlink(audio_path)
        
        print(f"[+] Generated video successfully")
        return CogPath(save_path)
