# HunyuanVideo-Avatar Replicate Cog

This is a **working** Replicate Cog implementation of **HunyuanVideo-Avatar**, a state-of-the-art audio-driven lip sync model that generates high-quality, dynamic videos with precise emotion alignment and character consistency.

## 🚨 Important Reality Check

**This thing works, but it's SLOW AS HELL.** I'm talking 13+ minutes for a 2-step inference with ~5 seconds of audio. The model is absolutely incredible (seriously, the demos from Hunyuan are mind-blowing), but the inference time makes it impractical for most use cases.

I'm not pushing this to Replicate's platform because:
- 30-minute timeout limit (this easily exceeds that)
- Would cost users a fortune in GPU time
- Nobody wants to wait 20+ minutes for a lip-sync video

But hey, if you want to run it locally and have the patience of a saint, it's all here and working! 🎉

## Local Usage (The Real Deal)

Want to try it? Here's how:

```bash
# Clone this bad boy
git clone https://github.com/zsxkib/cog-Hunyuan-Avatar.git
cd hunyuan-avatar-cog

# Run a prediction (grab a coffee... or three)
sudo cog predict -i image=@assets/image/src1.png -i audio=@assets/audio/4.WAV -i prompt="a person delivering a calm, reassuring message" -i num_inference_steps=2
```

**Pro tip:** Start with `num_inference_steps=2` unless you want to age a few years waiting. Even with 2 steps, you're looking at 10-15+ minutes on decent hardware.

The script will automagically:
- Download all the weights (~30GB+ of models)
- Set up the environment 
- Process your inputs
- Generate your video (eventually)

## 🔥 Performance Reality

- **Generation Time**: 10-20+ minutes for ~5 seconds of video (yes, really)
- **GPU Memory**: Needs ~30GB+ VRAM (tested on A100-80GB)
- **Model Quality**: Actually incredible when it finishes
- **Patience Required**: Zen master level

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
- **`num_inference_steps`** (int): Number of denoising steps (default: 50, range: 2-100) - **START WITH 2!**
- **`guidance_scale`** (float): Guidance scale for generation quality (default: 7.5, range: 1.0-20.0)
- **`seed`** (int, optional): Random seed for reproducible results

### Output

- **Video file**: MP4 video with the animated face lip-synced to the provided audio

### Example Usage (Local Only)

```bash
# Basic usage
cog predict -i image=@path/to/face.jpg -i audio=@path/to/speech.wav

# With custom settings (still slow!)
cog predict \
  -i image=@assets/image/src1.png \
  -i audio=@assets/audio/4.WAV \
  -i prompt="a person delivering a calm, reassuring message" \
  -i num_inference_steps=2 \
  -i guidance_scale=7.5
```

## Technical Details

### Model Architecture

This implementation includes:

1. **Character Image Injection Module**: Ensures dynamic motion while maintaining character consistency
2. **Audio Emotion Module (AEM)**: Extracts and transfers emotional cues from audio
3. **Face-Aware Audio Adapter (FAA)**: Enables precise audio-driven character animation
4. **Multi-Modal Diffusion Transformer (MM-DiT)**: Core generation architecture

### Key Features

- **High-Quality Output**: Generates videos up to 704x1216 resolution
- **Long Video Support**: Can generate up to 129 frames (~5 seconds at 25fps)
- **FP8 Optimization**: Uses quantized weights for (slightly) better performance
- **Memory Efficient**: Supports single GPU inference

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

- **Generation Time**: Seriously, it's slow. Like, really slow. 
- **GPU Memory**: Requires ~30GB+ VRAM (A100-80GB recommended)
- **Quality vs Speed**: Higher `num_inference_steps` = better quality but exponentially longer wait times
- **Cost**: If this were on Replicate, it would cost $20+ per generation

## Limitations

- **Speed**: Did I mention it's slow?
- **Memory**: Needs beefy hardware
- **Audio length**: Longer audio = longer wait times
- Optimal results with clear, front-facing portraits
- Complex backgrounds or extreme poses may affect quality

## Contributing

**Want to make this faster?** PRs are absolutely welcome! Some ideas:
- Better memory optimization
- Multi-GPU support improvements
- TensorRT optimization
- Quantization improvements
- Any other speed optimizations you can think of

This model has incredible potential but needs some serious optimization love.

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

## Final Words

This is a working implementation of an absolutely incredible model. The quality is genuinely mind-blowing when you see the results. It's just... really, really slow right now. But hey, that's what makes it a fun challenge for optimization! 

If you've got ideas to make this faster, please contribute. The ML community needs more working implementations of cutting-edge models, even if they start out slower than we'd like.

Now go grab some coffee and generate some amazing lip-sync videos! ☕️
