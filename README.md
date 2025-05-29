# HunyuanVideo-Avatar Replicate Cog

This is a Replicate Cog implementation of **HunyuanVideo-Avatar**, a state-of-the-art audio-driven lip sync model that generates high-quality, dynamic videos with precise emotion alignment and character consistency.

## Overview

HunyuanVideo-Avatar is capable of:
- **High-Dynamic Video Generation**: Creates realistic lip-sync videos with natural head and body movements
- **Emotion-Controllable**: Transfers emotional cues from audio to visual output
- **Multi-Character Support**: Can animate various avatar styles (photorealistic, cartoon, 3D-rendered)
- **Multi-Scale Generation**: Supports portrait, upper-body, and full-body video generation

## API Usage

### Inputs

- **`image`** (Path): Reference image for lip syncing - the face that will be animated
- **`audio`** (Path): Audio file to drive the lip sync (.wav, .mp3, etc.)
- **`prompt`** (str): Text prompt to guide the generation (default: "a person is speaking naturally")
- **`fps`** (int): Output video frame rate (default: 25, range: 20-30)
- **`num_inference_steps`** (int): Number of denoising steps (default: 50, range: 20-100)
- **`guidance_scale`** (float): Guidance scale for generation quality (default: 7.5, range: 1.0-20.0)
- **`seed`** (int, optional): Random seed for reproducible results

### Output

- **Video file**: MP4 video with the animated face lip-synced to the provided audio

### Example Usage

```python
import replicate

output = replicate.run(
    "your-username/hunyuan-avatar:version-hash",
    input={
        "image": open("portrait.jpg", "rb"),
        "audio": open("speech.wav", "rb"),
        "prompt": "a person speaking with natural expressions",
        "fps": 25,
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "seed": 12345
    }
)

# Download the generated video
video_url = output
```

## Technical Details

### Model Architecture

This implementation is based on the HunyuanVideo-Avatar paper and includes:

1. **Character Image Injection Module**: Ensures dynamic motion while maintaining character consistency
2. **Audio Emotion Module (AEM)**: Extracts and transfers emotional cues from audio
3. **Face-Aware Audio Adapter (FAA)**: Enables precise audio-driven character animation
4. **Multi-Modal Diffusion Transformer (MM-DiT)**: Core generation architecture

### Key Features

- **High-Quality Output**: Generates videos up to 704x1216 resolution
- **Long Video Support**: Can generate up to 129 frames (~5 seconds at 25fps)
- **Optimized Inference**: Uses FP8 quantization and GPU optimizations for faster generation
- **Memory Efficient**: Supports CPU offloading for lower VRAM requirements

## Input Guidelines

### Image Requirements
- **Format**: JPG, PNG, or other common image formats
- **Content**: Clear frontal or near-frontal face
- **Quality**: High resolution recommended (will be resized to optimal dimensions)
- **Styles**: Supports photorealistic, cartoon, 3D-rendered, and stylized characters

### Audio Requirements
- **Format**: WAV, MP3, or other common audio formats
- **Sample Rate**: Automatically converted to 16kHz
- **Length**: Up to ~5 seconds for optimal results
- **Content**: Clear speech audio for best lip sync results

### Prompt Guidelines
- Use descriptive text that matches the desired emotion and context
- Examples:
  - "a person speaking confidently in a business meeting"
  - "someone telling an exciting story with animated expressions"
  - "a character delivering a calm, reassuring message"

## Performance Considerations

- **Generation Time**: ~30-60 seconds depending on video length and settings
- **GPU Memory**: Requires ~24GB VRAM for full quality, optimized for 80GB
- **Quality vs Speed**: Higher `num_inference_steps` = better quality but longer generation time

## Limitations

- Optimal results with clear, front-facing portraits
- Audio length affects generation time and memory usage
- Very long audio clips may be truncated
- Complex backgrounds or extreme poses may affect quality

## Citation

If you use this model, please cite the original paper:

```bibtex
@misc{hu2025HunyuanVideo-Avatar,
      title={HunyuanVideo-Avatar: High-Fidelity Audio-Driven Human Animation for Multiple Characters}, 
      author={Yi Chen and Sen Liang and Zixiang Zhou and Ziyao Huang and Yifeng Ma and Junshu Tang and Qin Lin and Yuan Zhou and Qinglin Lu},
      year={2025},
      eprint={2505.20156},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/pdf/2505.20156}, 
}
```

## License

This implementation follows the original HunyuanVideo-Avatar license terms. Please refer to the original repository for detailed licensing information.

## Development

To build and test this Cog locally:

```bash
# Install cog
curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
chmod +x /usr/local/bin/cog

# Build the model
cog build -t hunyuan-avatar

# Test prediction
cog predict -i image=@portrait.jpg -i audio=@speech.wav -i prompt="a person speaking naturally"
```

## Support

For issues specific to this Replicate implementation, please open an issue in this repository. For questions about the underlying HunyuanVideo-Avatar model, refer to the [original repository](https://github.com/Tencent/HunyuanVideo-Avatar).
