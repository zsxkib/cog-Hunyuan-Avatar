import os
import math
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
from torch import distributed as dist
import loguru
import torch
import torch.nn as nn
import torch.distributed

RECOMMENDED_DTYPE = torch.float16

def mpi_comm():
    from mpi4py import MPI
    return MPI.COMM_WORLD

from torch import distributed as dist
def mpi_rank():
    return dist.get_rank()

def mpi_world_size():
    return dist.get_world_size()


class TorchIGather:
    def __init__(self):
        if not torch.distributed.is_initialized():
            rank = mpi_rank()
            world_size = mpi_world_size()
            os.environ['RANK'] = str(rank)
            os.environ['WORLD_SIZE'] = str(world_size)
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = str(29500)
            torch.cuda.set_device(rank)
            torch.distributed.init_process_group('nccl')

        self.handles = []
        self.buffers = []

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.groups_ids = []
        self.group = {}

        for i in range(self.world_size):
            self.groups_ids.append(tuple(range(i + 1)))

        for group in self.groups_ids:
            new_group = dist.new_group(group)
            self.group[group[-1]] = new_group


    def gather(self, tensor, n_rank=None):
        if n_rank is not None:
            group = self.group[n_rank - 1]
        else:
            group = None
        rank = self.rank
        tensor = tensor.to(RECOMMENDED_DTYPE)
        if rank == 0:
            buffer = [torch.empty_like(tensor) for i in range(n_rank)]
        else:
            buffer = None
        self.buffers.append(buffer)
        handle = torch.distributed.gather(tensor, buffer, async_op=True, group=group)
        self.handles.append(handle)

    def wait(self):
        for handle in self.handles:
            handle.wait()

    def clear(self):
        self.buffers = []
        self.handles = []


from diffusers.configuration_utils import ConfigMixin, register_to_config
try:
    # This diffusers is modified and packed in the mirror.
    from diffusers.loaders import FromOriginalVAEMixin
except ImportError:
    # Use this to be compatible with the original diffusers.
    from diffusers.loaders.single_file_model import FromOriginalModelMixin as FromOriginalVAEMixin
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from .vae import DecoderCausal3D, BaseOutput, DecoderOutput, DiagonalGaussianDistribution, EncoderCausal3D

"""
use trt need install polygraphy and onnx-graphsurgeon
python3 -m pip install --upgrade polygraphy>=0.47.0 onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com
"""
try:
    from polygraphy.backend.trt import ( TrtRunner, EngineFromBytes)
    from polygraphy.backend.common import BytesFromPath
except:
    print("TrtRunner or EngineFromBytes is not available, you can not use trt engine")

@dataclass
class DecoderOutput2(BaseOutput):
    sample: torch.FloatTensor
    posterior: Optional[DiagonalGaussianDistribution] = None


MODEL_OUTPUT_PATH = os.environ.get('MODEL_OUTPUT_PATH')
MODEL_BASE = os.environ.get('MODEL_BASE')

CPU_OFFLOAD = int(os.environ.get("CPU_OFFLOAD", 0))
DISABLE_SP = int(os.environ.get("DISABLE_SP", 0))
print(f'vae: cpu_offload={CPU_OFFLOAD}, DISABLE_SP={DISABLE_SP}')


class AutoencoderKLCausal3D(ModelMixin, ConfigMixin, FromOriginalVAEMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        force_upcast (`bool`, *optional*, default to `True`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
            can be fine-tuned / trained to a lower range without loosing too much precision in which case
            `force_upcast` can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlockCausal3D",),
        up_block_types: Tuple[str] = ("UpDecoderBlockCausal3D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        sample_tsize: int = 64,
        scaling_factor: float = 0.18215,
        force_upcast: float = True,
        spatial_compression_ratio: int = 8,
        time_compression_ratio: int = 4,
        disable_causal_conv: bool = False,
        mid_block_add_attention: bool = True,
        mid_block_causal_attn: bool = False,
        use_trt_engine: bool = False,
        nccl_gather: bool = True,
        engine_path: str = f"{MODEL_BASE}/HYVAE_decoder+conv_256x256xT_fp16_H20.engine",
    ):
        super().__init__()

        self.disable_causal_conv = disable_causal_conv
        self.time_compression_ratio = time_compression_ratio
        
        self.encoder = EncoderCausal3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            time_compression_ratio=time_compression_ratio,
            spatial_compression_ratio=spatial_compression_ratio,
            disable_causal=disable_causal_conv,
            mid_block_add_attention=mid_block_add_attention,
            mid_block_causal_attn=mid_block_causal_attn,
        )

        self.decoder = DecoderCausal3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            time_compression_ratio=time_compression_ratio,
            spatial_compression_ratio=spatial_compression_ratio,
            disable_causal=disable_causal_conv,
            mid_block_add_attention=mid_block_add_attention,
            mid_block_causal_attn=mid_block_causal_attn,
        )

        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, kernel_size=1)

        self.use_slicing = False
        self.use_spatial_tiling = False
        self.use_temporal_tiling = False


        # only relevant if vae tiling is enabled
        self.tile_sample_min_tsize = sample_tsize
        self.tile_latent_min_tsize = sample_tsize // time_compression_ratio

        self.tile_sample_min_size = self.config.sample_size
        sample_size = (
            self.config.sample_size[0]
            if isinstance(self.config.sample_size, (list, tuple))
            else self.config.sample_size
        )
        self.tile_latent_min_size = int(sample_size / (2 ** (len(self.config.block_out_channels) - 1)))
        self.tile_overlap_factor = 0.25

        use_trt_engine = False #if CPU_OFFLOAD else True
        # ============= parallism related code ===================
        self.parallel_decode = use_trt_engine
        self.nccl_gather = nccl_gather

        # only relevant if parallel_decode is enabled
        self.gather_to_rank0 = self.parallel_decode

        self.engine_path = engine_path
        
        self.use_trt_decoder = use_trt_engine

    @property
    def igather(self):
        assert self.nccl_gather and self.gather_to_rank0
        if hasattr(self, '_igather'):
            return self._igather
        else:
            self._igather = TorchIGather()
            return self._igather

    @property
    def use_padding(self):
        return (
            self.use_trt_decoder
            # dist.gather demands all processes possess to have the same tile shape.
            or (self.nccl_gather and self.gather_to_rank0)
        )

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (EncoderCausal3D, DecoderCausal3D)):
            module.gradient_checkpointing = value

    def enable_temporal_tiling(self, use_tiling: bool = True):
        self.use_temporal_tiling = use_tiling
    
    def disable_temporal_tiling(self):
        self.enable_temporal_tiling(False)
    
    def enable_spatial_tiling(self, use_tiling: bool = True):
        self.use_spatial_tiling = use_tiling
    
    def disable_spatial_tiling(self):
        self.enable_spatial_tiling(False)

    def enable_tiling(self, use_tiling: bool = True):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.enable_spatial_tiling(use_tiling)
        self.enable_temporal_tiling(use_tiling)

    def disable_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.disable_spatial_tiling()
        self.disable_temporal_tiling()

    def enable_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False


    def load_trt_decoder(self):
        self.use_trt_decoder = True
        self.engine = EngineFromBytes(BytesFromPath(self.engine_path))
        
        self.trt_decoder_runner = TrtRunner(self.engine)
        self.activate_trt_decoder()

    def disable_trt_decoder(self):
        self.use_trt_decoder = False
        del self.engine

    def activate_trt_decoder(self):
        self.trt_decoder_runner.activate()

    def deactivate_trt_decoder(self):
        self.trt_decoder_runner.deactivate()

    @property
    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]], _remove_lora=False
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor, _remove_lora=_remove_lora)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"), _remove_lora=_remove_lora)

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor, _remove_lora=True)

    @apply_forward_hook
    def encode(
        self, x: torch.FloatTensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of images into latents.

        Args:
            x (`torch.FloatTensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded images. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        assert len(x.shape) == 5, "The input tensor should have 5 dimensions"

        if self.use_temporal_tiling and x.shape[2] > self.tile_sample_min_tsize:
            return self.temporal_tiled_encode(x, return_dict=return_dict)
        
        if self.use_spatial_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
            return self.spatial_tiled_encode(x, return_dict=return_dict)
                            
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self.encoder(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self.encoder(x)

        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        assert len(z.shape) == 5, "The input tensor should have 5 dimensions"

        if self.use_temporal_tiling and z.shape[2] > self.tile_latent_min_tsize:
            return self.temporal_tiled_decode(z, return_dict=return_dict)
        
        if self.use_spatial_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
            return self.spatial_tiled_decode(z, return_dict=return_dict)
        
        if self.use_trt_decoder:
            # For unknown reason, `copy_outputs_to_host` must be set to True
            dec = self.trt_decoder_runner.infer({"input": z.to(RECOMMENDED_DTYPE).contiguous()}, copy_outputs_to_host=True)["output"].to(device=z.device, dtype=z.dtype)
        else:
            z = self.post_quant_conv(z)
            dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(
        self, z: torch.FloatTensor, return_dict: bool = True, generator=None
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        """
        Decode a batch of images.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.

        """

        if self.parallel_decode:
            if z.dtype != RECOMMENDED_DTYPE:
                loguru.logger.warning(
                    f'For better performance, using {RECOMMENDED_DTYPE} for both latent features and model parameters is recommended.'
                    f'Current latent dtype {z.dtype}. '
                    f'Please note that the input latent will be cast to {RECOMMENDED_DTYPE} internally when decoding.'
                )
                z = z.to(RECOMMENDED_DTYPE)

        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        if blend_extent == 0:
            return b

        a_region = a[..., -blend_extent:, :]
        b_region = b[..., :blend_extent, :]

        weights = torch.arange(blend_extent, device=a.device, dtype=a.dtype) / blend_extent
        weights = weights.view(1, 1, 1, blend_extent, 1)

        blended = a_region * (1 - weights) + b_region * weights

        b[..., :blend_extent, :] = blended
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        if blend_extent == 0:
            return b

        a_region = a[..., -blend_extent:]
        b_region = b[..., :blend_extent]

        weights = torch.arange(blend_extent, device=a.device, dtype=a.dtype) / blend_extent
        weights = weights.view(1, 1, 1, 1, blend_extent)

        blended = a_region * (1 - weights) + b_region * weights

        b[..., :blend_extent] = blended
        return b
    def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        if blend_extent == 0:
            return b

        a_region = a[..., -blend_extent:, :, :]
        b_region = b[..., :blend_extent, :, :]

        weights = torch.arange(blend_extent, device=a.device, dtype=a.dtype) / blend_extent
        weights = weights.view(1, 1, blend_extent, 1, 1)

        blended = a_region * (1 - weights) + b_region * weights

        b[..., :blend_extent, :, :] = blended
        return b

    def spatial_tiled_encode(self, x: torch.FloatTensor, return_dict: bool = True, return_moments: bool = False) -> AutoencoderKLOutput:
        r"""Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (`torch.FloatTensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoder_kl.AutoencoderKLOutput`] or `tuple`:
                If return_dict is True, a [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain
                `tuple` is returned.
        """
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split video into tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[-2], overlap_size):
            row = []
            for j in range(0, x.shape[-1], overlap_size):
                tile = x[:, :, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))

        moments = torch.cat(result_rows, dim=-2)
        if return_moments:
            return moments

        posterior = DiagonalGaussianDistribution(moments)
        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)


    def spatial_tiled_decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Decode a batch of images using a tiled decoder.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        if self.parallel_decode:

            rank = mpi_rank()
            torch.cuda.set_device(rank) # set device for trt_runner
            world_size = mpi_world_size()

            tiles = []
            afters_if_padding = []
            for i in range(0, z.shape[-2], overlap_size):
                for j in range(0, z.shape[-1], overlap_size):
                    tile = z[:, :, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]

                    if self.use_padding and (tile.shape[-2] < self.tile_latent_min_size or tile.shape[-1] < self.tile_latent_min_size):
                        from torch.nn import functional as F
                        after_h = tile.shape[-2] * 8
                        after_w = tile.shape[-1] * 8
                        padding = (0, self.tile_latent_min_size - tile.shape[-1], 0, self.tile_latent_min_size - tile.shape[-2], 0, 0)
                        tile = F.pad(tile, padding, "replicate").to(device=tile.device, dtype=tile.dtype)
                        afters_if_padding.append((after_h, after_w))
                    else:
                        afters_if_padding.append(None)

                    tiles.append(tile)


            # balance tasks
            ratio = math.ceil(len(tiles) / world_size)
            tiles_curr_rank = tiles[rank * ratio: None if rank == world_size - 1 else (rank + 1) * ratio]

            decoded_results = []


            total = len(tiles)
            n_task = ([ratio] * (total // ratio) + ([total % ratio] if total % ratio else []))
            n_task = n_task + [0] * (8 - len(n_task))

            for i, tile in enumerate(tiles_curr_rank):
                if self.use_trt_decoder:
                    # For unknown reason, `copy_outputs_to_host` must be set to True
                    decoded = self.trt_decoder_runner.infer(
                        {"input": tile.to(RECOMMENDED_DTYPE).contiguous()},
                        copy_outputs_to_host=True
                    )["output"].to(device=z.device, dtype=z.dtype)
                    decoded_results.append(decoded)
                else:
                    decoded_results.append(self.decoder(self.post_quant_conv(tile)))


                def find(n):
                    return next((i for i, task_n in enumerate(n_task) if task_n < n), len(n_task))


                if self.nccl_gather and self.gather_to_rank0:
                    self.igather.gather(decoded, n_rank=find(i + 1))

            if not self.nccl_gather:
                if self.gather_to_rank0:
                    decoded_results = mpi_comm().gather(decoded_results, root=0)
                    if rank != 0:
                        return DecoderOutput(sample=None)
                else:
                    decoded_results = mpi_comm().allgather(decoded_results)

                decoded_results = sum(decoded_results, [])
            else:
                # [Kevin]:
                # We expect all tiles obtained from the same rank have the same shape.
                # Shapes among ranks can differ due to the imbalance of task assignment.
                if self.gather_to_rank0:
                    if rank == 0:
                        self.igather.wait()
                        gather_results = self.igather.buffers
                    self.igather.clear()
                else:
                    raise NotImplementedError('The old `allgather` implementation is deprecated for nccl plan.')

                if rank != 0 and self.gather_to_rank0:
                    return DecoderOutput(sample=None)

                decoded_results = [col[i] for i in range(max([len(k) for k in gather_results])) for col in gather_results if i < len(col)]


            # Crop the padding region in pixel level
            if self.use_padding:
                new_decoded_results = []
                for after, dec in zip(afters_if_padding, decoded_results):
                    if after is not None:
                        after_h, after_w = after
                        new_decoded_results.append(dec[:, :, :, :after_h, :after_w])
                    else:
                        new_decoded_results.append(dec)
                decoded_results = new_decoded_results

            rows = []
            decoded_results_iter = iter(decoded_results)
            for i in range(0, z.shape[-2], overlap_size):
                row = []
                for j in range(0, z.shape[-1], overlap_size):
                    row.append(next(decoded_results_iter).to(rank))
                rows.append(row)
        else:
            rows = []
            for i in range(0, z.shape[-2], overlap_size):
                row = []
                for j in range(0, z.shape[-1], overlap_size):
                    tile = z[:, :, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                    tile = self.post_quant_conv(tile)
                    decoded = self.decoder(tile)
                    row.append(decoded)
                rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))

        dec = torch.cat(result_rows, dim=-2)
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def temporal_tiled_encode(self, x: torch.FloatTensor, return_dict: bool = True) -> AutoencoderKLOutput:
        assert not self.disable_causal_conv, "Temporal tiling is only compatible with causal convolutions."
    
        B, C, T, H, W = x.shape
        overlap_size = int(self.tile_sample_min_tsize * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_tsize * self.tile_overlap_factor)
        t_limit = self.tile_latent_min_tsize - blend_extent

        # Split the video into tiles and encode them separately.
        row = []
        for i in range(0, T, overlap_size):
            tile = x[:, :, i : i + self.tile_sample_min_tsize + 1, :, :]
            if self.use_spatial_tiling and (tile.shape[-1] > self.tile_sample_min_size or tile.shape[-2] > self.tile_sample_min_size):
                tile = self.spatial_tiled_encode(tile, return_moments=True)
            else:
                tile = self.encoder(tile)
                tile = self.quant_conv(tile)
            if i > 0:
                tile = tile[:, :, 1:, :, :]
            row.append(tile)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
                result_row.append(tile[:, :, :t_limit, :, :])
            else:
                result_row.append(tile[:, :, :t_limit+1, :, :])
        
        moments = torch.cat(result_row, dim=2)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)
    
    def temporal_tiled_decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        # Split z into overlapping tiles and decode them separately.
        assert not self.disable_causal_conv, "Temporal tiling is only supported with causal convolutions."
    
        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_tsize * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_tsize * self.tile_overlap_factor)
        t_limit = self.tile_sample_min_tsize - blend_extent
        rank = 0 if CPU_OFFLOAD or DISABLE_SP else mpi_rank()
        row = []
        for i in range(0, T, overlap_size):
            tile = z[:, :, i : i + self.tile_latent_min_tsize + 1, :, :]
            if self.use_spatial_tiling and (tile.shape[-1] > self.tile_latent_min_size or tile.shape[-2] > self.tile_latent_min_size):
                decoded = self.spatial_tiled_decode(tile, return_dict=True).sample
            else:
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
            if i > 0 and (not (self.parallel_decode and self.gather_to_rank0) or rank == 0):
                decoded = decoded[:, :, 1:, :, :]
            row.append(decoded)
        if not CPU_OFFLOAD and not DISABLE_SP and self.parallel_decode and self.gather_to_rank0 and rank != 0:
            return DecoderOutput(sample=None)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
                result_row.append(tile[:, :, :t_limit, :, :])
            else:
                result_row.append(tile[:, :, :t_limit+1, :, :])
        
        dec = torch.cat(result_row, dim=2)
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        return_posterior: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput2, torch.FloatTensor]:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            if return_posterior:
                return (dec, posterior)
            else:
                return (dec,)
        if return_posterior:
            return DecoderOutput2(sample=dec, posterior=posterior)
        else:
            return DecoderOutput2(sample=dec)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query,
        key, value) are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)
