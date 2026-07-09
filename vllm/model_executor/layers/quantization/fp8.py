# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Module
from torch.utils._python_dispatch import TorchDispatchMode

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    init_fp8_linear_kernel,
)
from vllm.model_executor.kernels.linear.scaled_mm import MarlinFP8ScaledMMLinearKernel
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.cpu_fused_moe import select_experts
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.moe_monokernel_interleave import (
    interleave_for_tma_wgmma_up_v2,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    convert_to_fp8_moe_kernel_format,
    make_fp8_moe_kernel,
    make_fp8_moe_quant_config,
    select_fp8_moe_backend,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_input_scale,
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
    process_fp8_input_tensor_strategy_moe,
    process_fp8_weight_tensor_strategy,
    process_fp8_weight_tensor_strategy_moe,
    validate_fp8_block_shape,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    create_fp8_quant_key,
    is_layer_skipped,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_block_fp8_supported,
    cutlass_fp8_supported,
    normalize_e4m3fn_to_e4m3fnuz,
)
from vllm.model_executor.model_loader.reload.layerwise import (
    initialize_online_processing,
)
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    is_deep_gemm_supported,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = init_logger(__name__)


class Fp8Config(QuantizationConfig):
    """Config class for FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
        weight_block_size: list[int] | None = None,
    ) -> None:
        super().__init__()

        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized

        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        if weight_block_size is not None:
            if not is_checkpoint_fp8_serialized:
                raise ValueError(
                    "The block-wise quantization only supports fp8-serialized "
                    "checkpoint for now."
                )
            if len(weight_block_size) != 2:
                raise ValueError(
                    "The quantization block size of weight must have 2 "
                    f"dimensions, but got {len(weight_block_size)} dimensions"
                )
            if activation_scheme != "dynamic":
                raise ValueError(
                    "The block-wise quantization only supports "
                    "dynamic activation scheme for now, but got "
                    f"{activation_scheme} activation scheme."
                )
        self.weight_block_size = weight_block_size
        self.use_deep_gemm: bool | None = None

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.ignored_layers is not None:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Fp8Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = "fp8" in quant_method
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
        return cls(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            if not self.is_checkpoint_fp8_serialized:
                online_method = Fp8OnlineLinearMethod(self)
                online_method.marlin_input_dtype = get_marlin_input_dtype(prefix)
                return online_method
            else:
                offline_method = Fp8LinearMethod(self)
                offline_method.marlin_input_dtype = get_marlin_input_dtype(prefix)
                return offline_method
        elif isinstance(layer, FusedMoE):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            if self.is_checkpoint_fp8_serialized:
                moe_quant_method = Fp8MoEMethod(self, layer)
            else:
                moe_quant_method = Fp8OnlineMoEMethod(self, layer)
            return moe_quant_method
        elif isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        return None

    def get_cache_scale(self, name: str) -> str | None:
        """
        Check whether the param name matches the format for k/v cache scales
        in compressed-tensors. If this is the case, return its equivalent
        param name expected by vLLM

        :param name: param name
        :return: matching param name for KV cache scale in vLLM
        """
        if name.endswith(".output_scale") and ".k_proj" in name:
            return name.replace(".k_proj.output_scale", ".attn.k_scale")
        if name.endswith(".output_scale") and ".v_proj" in name:
            return name.replace(".v_proj.output_scale", ".attn.v_scale")
        if name.endswith(".output_scale") and ".q_proj" in name:
            return name.replace(".q_proj.output_scale", ".attn.q_scale")
        if name.endswith("self_attn.prob_output_scale"):
            return name.replace(".prob_output_scale", ".attn.prob_scale")
        # If no matches, return None
        return None


class CopyNumelCounter(TorchDispatchMode):
    """
    Tracks total number of elements modified with `copy_`. Useful for keeping
    track of weight loading where underlying weights can be arbitrarily
    transformed (such as with `narrow`) before calling copy.
    """

    def __init__(self):
        super().__init__()
        self.copied_numel = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        out = func(*args, **kwargs)
        if func == torch.ops.aten.copy_.default:
            self.copied_numel += args[0].numel()
        return out


def _copy_missing_attrs(old: torch.Tensor, new: torch.Tensor) -> None:
    """Copies any attrs present in `old` but not in `new` to `new`"""
    new_attrs = set(dir(new))
    attrs_to_set = {}
    for attr in dir(old):
        if attr not in new_attrs:
            attrs_to_set[attr] = getattr(old, attr)
    set_weight_attrs(new, attrs_to_set)


class Fp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Limitations:
    1. Only support float8_e4m3fn data type due to the limitation of
       torch._scaled_mm (https://github.com/pytorch/pytorch/blob/2e48b39603411a41c5025efbe52f89560b827825/aten/src/ATen/native/cuda/Blas.cpp#L854-L856)

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.is_scale_e8m0 = getattr(quant_config, "is_scale_e8m0", False)
        self.cutlass_block_fp8_supported = cutlass_block_fp8_supported()
        self.out_dtype = torch.get_default_dtype()
        self.input_dtype = get_current_vllm_config().model_config.dtype

        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        self.marlin_input_dtype = None
        self.use_marlin = False

        if self.quant_config.use_deep_gemm is not None:
            self.use_deep_gemm = self.quant_config.use_deep_gemm
        else:
            self.use_deep_gemm = is_deep_gemm_supported()

        self.weight_block_size = self.quant_config.weight_block_size
        self.block_quant = self.weight_block_size is not None
        self.act_q_static = self.quant_config.activation_scheme == "static"

        if self.block_quant:
            assert not self.act_q_static
            assert self.weight_block_size is not None

            self.activation_quant_key = create_fp8_quant_key(
                static=self.act_q_static,
                group_shape=GroupShape(1, self.weight_block_size[0]),
            )
            self.weight_quant_key = create_fp8_quant_key(
                static=True, group_shape=GroupShape(*self.weight_block_size)
            )
        else:
            self.weight_quant_key = kFp8StaticTensorSym
            # Use per-token quantization for better perf if dynamic and cutlass
            if self.act_q_static:
                self.activation_quant_key = kFp8StaticTensorSym
            elif cutlass_fp8_supported():
                self.activation_quant_key = kFp8DynamicTokenSym
            else:
                self.activation_quant_key = kFp8DynamicTensorSym

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        if self.block_quant:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            validate_fp8_block_shape(
                layer,
                input_size,
                output_size,
                input_size_per_partition,
                output_partition_sizes,
                self.weight_block_size,
            )

        weight = create_fp8_weight_parameter(
            output_size_per_partition, input_size_per_partition, weight_loader
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        if not self.block_quant:
            scale = create_fp8_scale_parameter(
                PerTensorScaleParameter,
                output_partition_sizes,
                input_size_per_partition,
                None,
                weight_loader,
            )
            layer.register_parameter("weight_scale", scale)
        else:
            assert not self.act_q_static
            assert self.weight_block_size is not None
            scale = create_fp8_scale_parameter(
                BlockQuantScaleParameter,
                output_partition_sizes,
                input_size_per_partition,
                self.weight_block_size,
                weight_loader,
                scale_dtype=(torch.float8_e8m0fnu if self.is_scale_e8m0 else None),
            )
            # The weight_scale_inv name is intentional for deepseekv3
            layer.register_parameter("weight_scale_inv", scale)

        # INPUT ACTIVATION SCALE
        if self.act_q_static:
            scale = create_fp8_input_scale(output_partition_sizes, weight_loader)
            set_weight_attrs(scale, {"scale_type": "input_scale"})
            layer.register_parameter("input_scale", scale)

        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=self.activation_quant_key,
            weight_quant_key=self.weight_quant_key,
            weight_shape=layer.weight.shape,
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
            module_name=self.__class__.__name__,
        )

        self.use_marlin = isinstance(self.fp8_linear, MarlinFP8ScaledMMLinearKernel)

    def process_weights_after_loading(self, layer: Module) -> None:
        if self.use_marlin:
            # Only Marlin kernels support `marlin_input_dtype`; guard to avoid
            # AttributeError if backend selection changes.
            if hasattr(self.fp8_linear, "marlin_input_dtype"):
                self.fp8_linear.marlin_input_dtype = self.marlin_input_dtype
            self.fp8_linear.process_weights_after_loading(layer)
            return

        input_scale = None
        # TODO(rob): refactor block quant into separate class.
        if self.block_quant:
            assert not self.act_q_static

        # If checkpoint not serialized fp8, quantize the weights.
        else:
            # If checkpoint is fp8 per-tensor, handle that there are N scales for N
            # shards in a fused module
            weight = layer.weight
            weight_scale = layer.weight_scale

            # If using w8a8, torch._scaled_mm needs per tensor, so
            # requantize the logical shards as a single weight.
            weight, weight_scale, input_scale = process_fp8_weight_tensor_strategy(
                weight,
                weight_scale,
                layer.logical_widths,
                getattr(layer, "input_scale", None),
            )
            if self.act_q_static:
                assert input_scale is not None
                input_scale = input_scale.max()
            weight = weight.t()

            # Update layer with new values.
            replace_parameter(layer, "weight", weight.data)
            replace_parameter(layer, "weight_scale", weight_scale.data)

        if input_scale is not None:
            replace_parameter(layer, "input_scale", input_scale)
        else:
            layer.input_scale = None

        self.fp8_linear.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # if batch invariant mode is enabled, prefer DeepGEMM FP8 path
        # we will use BF16 dequant when DeepGEMM is not supported.
        if envs.VLLM_BATCH_INVARIANT:
            if self.block_quant:
                assert self.weight_block_size is not None
                return self.fp8_linear.apply_weights(
                    layer,
                    x,
                    bias,
                )
            else:
                # per-tensor/channel: dequant to BF16 and run GEMM
                weight_fp8 = layer.weight.to(torch.bfloat16)
                weight_scale = layer.weight_scale.to(torch.bfloat16)
                if weight_scale.numel() == 1:
                    # Per-tensor: simple scalar multiplication
                    weight_bf16 = weight_fp8 * weight_scale
                else:
                    # Multiple scales (fused modules like QKV)
                    # Try to infer correct broadcasting
                    # weight is [K, N], scale could be [num_logical_weights]
                    # Need to figure out how to broadcast - for now just try
                    # direct multiplication
                    if (
                        weight_scale.dim() == 1
                        and weight_scale.shape[0] == weight_fp8.shape[0]
                    ):
                        # Per-row scaling
                        weight_bf16 = weight_fp8 * weight_scale.unsqueeze(1)
                    else:
                        # Fallback
                        weight_bf16 = weight_fp8 * weight_scale
                return torch.nn.functional.linear(x, weight_bf16.t(), bias)

        if self.use_marlin:
            return self.fp8_linear.apply_weights(layer, x, bias)

        return self.fp8_linear.apply_weights(layer, x, bias)


# TODO(future PR): remove this class in favor of
# online/fp8.py::Fp8PerTensorOnlineLinearMethod
class Fp8OnlineLinearMethod(Fp8LinearMethod):
    """Online version of Fp8LinearMethod which loads a full precision checkpoint
    and quantizes weights during loading."""

    uses_meta_device: bool = True

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                device="meta",  # materialized and processed during loading
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        initialize_online_processing(layer)

        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=self.activation_quant_key,
            weight_quant_key=self.weight_quant_key,
            weight_shape=layer.weight.shape,
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
            module_name=self.__class__.__name__,
        )
        self.use_marlin = isinstance(self.fp8_linear, MarlinFP8ScaledMMLinearKernel)

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        # TODO(future): support block_quant in online quant path
        assert not self.block_quant

        layer.input_scale = None
        qweight, weight_scale = ops.scaled_fp8_quant(layer.weight, scale=None)

        # Update layer with new values.
        replace_parameter(layer, "weight", qweight.data)
        replace_parameter(layer, "weight_scale", weight_scale.data)

        if self.use_marlin:
            # Only Marlin kernels support `marlin_input_dtype`; guard to avoid
            # AttributeError if backend selection changes.
            if hasattr(self.fp8_linear, "marlin_input_dtype"):
                self.fp8_linear.marlin_input_dtype = self.marlin_input_dtype
            self.fp8_linear.process_weights_after_loading(layer)
        else:
            weight = qweight.t()
            replace_parameter(layer, "weight", weight.data)

        # Prevent duplicate processing (e.g., during weight reload)
        layer._already_called_process_weights_after_loading = True


class Fp8MoEMethod(FusedMoEMethodBase):
    """MoE method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config, layer: torch.nn.Module):
        super().__init__(layer.moe_config)
        self.quant_config = quant_config
        self.weight_block_size = self.quant_config.weight_block_size
        self.block_quant: bool = self.weight_block_size is not None
        self.weight_scale_name = (
            "weight_scale_inv" if self.block_quant else "weight_scale"
        )

        # Scratchpad for MoE monokernel fast path (Qwen3.5-35B FP8 block-wise)
        # Layout: BS x E x N fp8 + BS x E x N/2 fp8 + BS x HIDDEN fp16
        # with BS=1024: 4MB + <1MB + 10MB < 4M x 4byte
        #
        # Allocate ZEROED (not empty): the software grid/partial barriers in
        # the monokernel (src/moe_grid_barrier.h) spin on counter slots that
        # live at the tail of this scratchpad and MUST start at 0 on first
        # use (self-maintaining via ping-pong reset thereafter). The kernel
        # is launched standalone (non-cooperative) so it can be captured into
        # a CUDA Graph; the wrapper's one-shot cudaMemsetAsync would be
        # *recorded* (not executed) if it first runs during graph capture,
        # leaving the counters uninitialized and deadlocking the barrier
        # spin. Zeroing here guarantees valid counters before the first
        # (capture-time) launch. Matches test_monokernel_accuracy.py.
        self.moe_monokernel_scratchpad = torch.zeros(
            (1024, 4096),
            dtype=torch.float32,
            device="cpu",
        )
        # Whether this layer is eligible for the MoE monokernel fast path.
        # Determined after weight loading in process_weights_after_loading.
        self._use_moe_monokernel = False

        # Set weight key and activation key for kernel compatibility
        if self.block_quant:
            weight_key = kFp8Static128BlockSym
            activation_key = kFp8Dynamic128Sym
        else:
            weight_key = kFp8StaticTensorSym
            activation_key = (
                kFp8StaticTensorSym
                if self.quant_config.activation_scheme == "static"
                else kFp8DynamicTensorSym
            )

        # Select Fp8 MoE backend
        self.fp8_backend, self.experts_cls = select_fp8_moe_backend(
            config=self.moe,
            weight_key=weight_key,
            activation_key=activation_key,
            allow_vllm_cutlass=False,
        )

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        assert self.quant_config.is_checkpoint_fp8_serialized
        params_dtype = torch.float8_e4m3fn

        if self.block_quant:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            tp_size = get_tensor_model_parallel_world_size()
            block_n, block_k = (
                self.weight_block_size[0],
                self.weight_block_size[1],
            )
            # NOTE: To ensure proper alignment of the block-wise quantization
            # scales, the output_size of the weights for both the gate and up
            # layers must be divisible by block_n.
            # Required by column parallel or enabling merged weights
            if intermediate_size_per_partition % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )
            if tp_size > 1 and intermediate_size_per_partition % block_k != 0:
                # Required by row parallel
                raise ValueError(
                    f"The input_size of down's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}."
                )

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # BIASES (for models like GPT-OSS that have biased MoE)
        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    dtype=layer.orig_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=layer.orig_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

        # WEIGHT_SCALES
        if not self.block_quant:
            # For per-tensor quant, the scales are per expert and weight.
            w13_scale_data = torch.ones(num_experts, 2, dtype=torch.float32)
            w2_scale_data = torch.ones(num_experts, dtype=torch.float32)
        else:
            # For block quant, the scales are per block (typically 128x128).
            w13_scale_data = torch.ones(
                num_experts,
                2 * ((intermediate_size_per_partition + block_n - 1) // block_n),
                (hidden_size + block_k - 1) // block_k,
                dtype=torch.float32,
            )
            w2_scale_data = torch.ones(
                num_experts,
                (hidden_size + block_n - 1) // block_n,
                (intermediate_size_per_partition + block_k - 1) // block_k,
                dtype=torch.float32,
            )
        w13_weight_scale = torch.nn.Parameter(w13_scale_data, requires_grad=False)
        w2_weight_scale = torch.nn.Parameter(w2_scale_data, requires_grad=False)
        # Note: name is weight_scale for tensor, weight_scale_inv for block.
        layer.register_parameter(f"w13_{self.weight_scale_name}", w13_weight_scale)
        layer.register_parameter(f"w2_{self.weight_scale_name}", w2_weight_scale)

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
            if self.block_quant
            else {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.quant_config.activation_scheme == "static":
            assert not self.block_quant
            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)

        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def _setup_kernel(
        self,
        layer: FusedMoE,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_input_scale: torch.Tensor | None,
        w2_input_scale: torch.Tensor | None,
    ) -> None:
        # Shuffle weights to runtime format.
        w13, w2, w13_scale, w2_scale = convert_to_fp8_moe_kernel_format(
            fp8_backend=self.fp8_backend,
            layer=layer,
            w13=w13,
            w2=w2,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            w13_input_scale=w13_input_scale,
            w2_input_scale=w2_input_scale,
        )

        # Replace parameters with updated versions. Note that this helper
        # function ensures the replacement is compatible with RL weight reloads.
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, f"w13_{self.weight_scale_name}", w13_scale)
        replace_parameter(layer, f"w2_{self.weight_scale_name}", w2_scale)

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        if self.moe_quant_config:
            assert self.experts_cls is not None
            self.moe_kernel = make_fp8_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                fp8_backend=self.fp8_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._maybe_init_expert_routing_tables(),
                shared_experts=layer.shared_experts,
            )

    def process_weights_after_loading(self, layer: Module) -> None:
        # Allow for accessing weights and scales in standard way.
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = getattr(layer, f"w13_{self.weight_scale_name}")
        w2_scale = getattr(layer, f"w2_{self.weight_scale_name}")
        w13_input_scale = layer.w13_input_scale
        w2_input_scale = layer.w2_input_scale

        # MI300x and MI325x use FNUZ format for FP8. Convert if needed.
        if current_platform.is_fp8_fnuz():
            w13, w13_scale, w13_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                w13,
                w13_scale,
                w13_input_scale,
            )
            w2, w2_scale, w2_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                w2,
                w2_scale,
                w2_input_scale,
            )

        # Per tensor kernels require single activation scale. Use the max.
        if self.quant_config.activation_scheme == "static":
            assert not self.block_quant
            assert w13_input_scale is not None and w2_input_scale is not None
            w13_input_scale, w2_input_scale = process_fp8_input_tensor_strategy_moe(
                w13_input_scale, w2_input_scale
            )
            replace_parameter(layer, "w13_input_scale", w13_input_scale)
            replace_parameter(layer, "w2_input_scale", w2_input_scale)

        # Per tensor kernels require single weight scale for w13 per expert, but
        # on disk there is a scale for w1 and w3. Use the max to requantize.
        if not self.block_quant:
            shard_size = layer.intermediate_size_per_partition
            w13, w13_scale = process_fp8_weight_tensor_strategy_moe(
                w13, w13_scale, shard_size, layer.local_num_experts
            )

        # Shuffle weights to runtime format and setup kernel.
        self._setup_kernel(
            layer, w13, w2, w13_scale, w2_scale, w13_input_scale, w2_input_scale
        )

        # Decide whether the MoE monokernel fast path is eligible for this
        # layer. The monokernel only supports the Qwen3.5-35B FP8 block-wise
        # shape (E=256, K=2048, N=2*512=1024) with top_k>1, and requires the
        # raw (unshuffled) block-wise FP8 weight layout. Only the TRITON
        # backend preserves that layout (convert_to_fp8_moe_kernel_format is a
        # no-op there); backends like DEEPGEMM repack the weights and scales,
        # so they are not eligible. Larger batches (M>64) fall back to the
        # modular kernel inside apply_monolithic.
        #
        # Gated by VLLM_USE_MOE_MONOKERNEL (default on); set it to 0 to force
        # the standard TRITON fused-MoE backend for A/B benchmarking.
        self._use_moe_monokernel = (
            envs.VLLM_USE_MOE_MONOKERNEL
            and self.block_quant
            and self.fp8_backend == Fp8MoeBackend.TRITON
            and getattr(layer, "global_num_experts", 0) == 256
            and (layer.w13_weight.size(1) == 1024  # TP=1: 2*N=1024
                 or layer.w13_weight.size(1) == 512)  # TP=2: 2*(N/2)=512
            and layer.w13_weight.size(2) == 2048
            and getattr(layer, "top_k", 1) > 1
        )
        if self._use_moe_monokernel:
            logger.info(
                "MoE monokernel fast path ENABLED for %s "
                "(E=256, N=1024, K=2048, top_k=%d, backend=%s)",
                getattr(layer, "prefix", "<no-prefix>"),
                getattr(layer, "top_k", 1),
                self.fp8_backend.value,
            )
            # Pre-compute the BS8 (M<=8) gate/up PAIR interleave now, at
            # load time, so the ~512 MiB/layer repack is reserved BEFORE
            # vLLM's memory profiling sizes the KV cache. The repack is
            # cached on the weight tensor's `_tma_interleaved_up_v2`
            # attribute; the monokernel op checks that attribute first, so
            # the decode-time path becomes a cache hit with no allocation.
            # (The BS64 / prefill path uses the raw weights and needs no
            # interleave.)
            #
            # EP: detect expert parallelism. Under EP each rank holds only its
            # local expert slice [128, ...]; the monokernel EP path uses
            # global-id indexing into a [256] buffer + a routing filter that
            # computes only this rank's experts, so below we scatter the local
            # weights into their GLOBAL positions in a [256] buffer (rest zero,
            # never read). The per-layer non-EP interleave is skipped under EP.
            from vllm.distributed.parallel_state import get_ep_group
            _ep = get_ep_group()
            self._moe_ep_size = _ep.world_size
            self._is_ep_monokernel = self._moe_ep_size > 1
            if not self._is_ep_monokernel:
                up_interleaved = interleave_for_tma_wgmma_up_v2(
                    layer.w13_weight
                ).contiguous()
                with contextlib.suppress(AttributeError, RuntimeError):
                    layer.w13_weight._tma_interleaved_up_v2 = up_interleaved
            else:
                # Build the global [256] weight/scale buffers with this rank's
                # local experts at their global positions.
                GLOBAL_E = 256
                ep_rank = _ep.rank_in_group
                local_E = layer.w13_weight.size(0)
                self._moe_ep_expert_base = ep_rank * local_E
                sname = self.weight_scale_name

                def _scatter_global(local_t):
                    g = torch.zeros(
                        (GLOBAL_E,) + tuple(local_t.shape[1:]),
                        dtype=local_t.dtype, device=local_t.device,
                    )
                    g[self._moe_ep_expert_base:
                      self._moe_ep_expert_base + local_E] = local_t
                    return g

                self._ep_w13_global = _scatter_global(layer.w13_weight)
                self._ep_w2_global = _scatter_global(layer.w2_weight)
                self._ep_w13_scale_global = _scatter_global(
                    getattr(layer, f"w13_{sname}")
                )
                self._ep_w2_scale_global = _scatter_global(
                    getattr(layer, f"w2_{sname}")
                )
                self._ep_w13_interleaved = interleave_for_tma_wgmma_up_v2(
                    self._ep_w13_global
                ).contiguous()
                # Attach as the op's interleave cache so decode-time calls are
                # a cache hit (no per-call repack / allocation).
                with contextlib.suppress(AttributeError, RuntimeError):
                    self._ep_w13_global._tma_interleaved_up_v2 = (
                        self._ep_w13_interleaved
                    )

                # ── 3a: peer-mapped activation workspace for in-kernel EP
                # dispatch (all-to-all HIDING). Each rank writes its local
                # hidden into this VMM-mapped buffer; peers peer-read it inside
                # the monokernel instead of the NCCL all-gather. Reuses
                # MoELLWorkspace's VMM cross-process peer-mapping. Opt-in via
                # VLLM_MOE_EP_INKERNEL_DISPATCH=1 so the validated parity path
                # (stock all-gather + monokernel compute) is unaffected. ──
                import os as _os
                self._ep_inkernel_dispatch = (
                    _os.environ.get("VLLM_MOE_EP_INKERNEL_DISPATCH") == "1"
                )
                self._ep_act_ws = None
                if self._ep_inkernel_dispatch:
                    from vllm.distributed.moe_ll_workspace import MoELLWorkspace
                    # Allocate the peer-mapped workspace ONCE and share it
                    # across all MoE layers. process_weights_after_loading runs
                    # per layer; allocating one MoELLWorkspace per layer would
                    # do 40 VMM fd-exchanges over the SAME Unix-socket path with
                    # 40 EP-group collectives, desyncing the group and hanging a
                    # later step. Layers run sequentially, so one shared buffer
                    # is correct. First layer allocates (lockstep on all ranks);
                    # the rest reuse — keeping the collective count balanced.
                    cls = type(self)
                    if getattr(cls, "_ep_act_ws_singleton", None) is None:
                        # Use the EP CPU (gloo) group for the IPC-handle
                        # exchange. The handle bytes are host data, and running
                        # this ad-hoc object collective on the NCCL device_group
                        # (shared with vLLM's own EP collectives) mid-load can
                        # desync that communicator and hang the first forward.
                        # The gloo group has identical rank ordering, so kernel
                        # peer indices are unchanged.
                        _ep_ll_group = getattr(_ep, "cpu_group", None) or \
                            _ep.device_group
                        cls._ep_act_ws_singleton = MoELLWorkspace(
                            max_num_tokens=256,
                            hidden_dim=2048,
                            tp_group=_ep_ll_group,
                        )
                    self._ep_act_ws = cls._ep_act_ws_singleton
                    logger.info_once(
                        "MoE monokernel EP in-kernel dispatch ENABLED "
                        "(shared peer-mapped activation workspace)")
                logger.info(
                    "MoE monokernel EP path: rank %d owns experts [%d, %d) "
                    "of %d (global [256] buffer, ~2x MoE weight memory)",
                    ep_rank, self._moe_ep_expert_base,
                    self._moe_ep_expert_base + local_E, GLOBAL_E,
                )

            # Move the monokernel scratchpad to the weight device now, at
            # load time. Doing the host->device move lazily inside
            # apply_monolithic would run during CUDA graph capture, where a
            # pageable H2D copy synchronizes the stream (illegal during
            # capture -> hang) and the buffer would not have a stable
            # address for graph replay.
            self.moe_monokernel_scratchpad = self.moe_monokernel_scratchpad.to(
                layer.w13_weight.device
            )

            # Setup LL workspace for fused AR when tp_size > 1
            # Only initialized if VLLM_MOE_FUSED_AR=1 is set
            import os
            if os.environ.get("VLLM_MOE_FUSED_AR") == "1":
                from vllm.distributed.parallel_state import (
                    get_tensor_model_parallel_world_size,
                    get_tensor_model_parallel_rank,
                )
                tp_size = get_tensor_model_parallel_world_size()
                if tp_size > 1:
                    from vllm.distributed.moe_ll_workspace import MoELLWorkspace
                    from vllm.distributed.parallel_state import get_tp_group
                    self._moe_ll_workspace = MoELLWorkspace(
                        max_num_tokens=256,  # Must cover max cudagraph_capture_size
                        hidden_dim=2048,
                        tp_group=get_tp_group().device_group,
                    )
                    self._moe_tp_size = tp_size
                    self._moe_tp_rank = get_tensor_model_parallel_rank()
                else:
                    self._moe_ll_workspace = None
                    self._moe_tp_size = 1
                    self._moe_tp_rank = 0

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEPrepareAndFinalizeModular | None:
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel initialization "
            "logic. This function should not be called."
        )

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        w1_scale = getattr(layer, f"w13_{self.weight_scale_name}")
        w2_scale = getattr(layer, f"w2_{self.weight_scale_name}")
        a1_scale = layer.w13_input_scale
        a2_scale = layer.w2_input_scale

        quant_config = make_fp8_moe_quant_config(
            fp8_backend=self.fp8_backend,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=self.weight_block_size,
        )

        # Inject biases into the quant config if the model has them
        # (e.g. GPT-OSS biased MoE)
        if quant_config is not None and self.moe.has_bias:
            w13_bias = getattr(layer, "w13_bias", None)
            w2_bias = getattr(layer, "w2_bias", None)
            if w13_bias is not None:
                quant_config._w1.bias = w13_bias
            if w2_bias is not None:
                quant_config._w2.bias = w2_bias

        return quant_config

    @property
    def supports_eplb(self) -> bool:
        return True

    @property
    def is_monolithic(self) -> bool:
        # Route eligible layers through apply_monolithic so the MoE monokernel
        # fast path (which does its own routing from router_logits) can run.
        # Otherwise defer to the base-class decision (driven by the selected
        # modular kernel).
        if getattr(self, "_use_moe_monokernel", False):
            return True
        return super().is_monolithic

    @property
    def mk_owns_shared_expert(self) -> bool:
        # The MoE monokernel only computes routed experts — it does NOT run
        # the shared expert internally. When the monokernel path is active,
        # return False so the runner handles shared experts externally (via
        # NO_OVERLAP or MULTI_STREAM_OVERLAPPED). Without this override the
        # base class returns True (because self.moe_kernel was built with
        # shared_experts=layer.shared_experts), which makes the runner skip
        # shared expert execution entirely — a silent correctness bug for
        # models like Qwen3.5-35B that have shared experts.
        if getattr(self, "_use_moe_monokernel", False):
            # OPT-IN (VLLM_MOE_EP_SHARED_OVERLAP_COMBINE=1): on the EP path the
            # monokernel runs the shared expert ITSELF, on the aux stream,
            # ordered after the routed monokernel so it overlaps the combine
            # collective (NVLink/wait-bound, HBM idle) instead of contending
            # with the HBM-bound routed compute. Reporting True makes the runner
            # skip both its NO_OVERLAP and MULTI_STREAM_OVERLAPPED shared-expert
            # calls; apply_monolithic stores the result into
            # layer.shared_experts._output for the runner to pick up. Only valid
            # on the EP monokernel path (which always takes the EP branch), so
            # the shared expert is guaranteed to be computed there.
            import os as _os_mko
            if (
                _os_mko.environ.get("VLLM_MOE_EP_SHARED_OVERLAP_COMBINE") == "1"
                and getattr(self, "_is_ep_monokernel", False)
            ):
                return True
            return False
        return super().mk_owns_shared_expert

    def _apply_modular_fallback(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Route + run the standard modular kernel.

        Used when the monokernel fast path is enabled for this layer but the
        current batch does not satisfy the kernel's constraints (e.g. M > 64
        during prefill).
        """
        assert self.moe_kernel is not None
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=layer.top_k,
            use_grouped_topk=layer.use_grouped_topk,
            renormalize=layer.renormalize,
            topk_group=layer.topk_group,
            num_expert_group=layer.num_expert_group,
            custom_routing_function=layer.custom_routing_function,
            scoring_func=layer.scoring_func,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
        )
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts_input=None,
        )

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.is_monolithic

        # MoE monokernel fast path (Qwen3.5-35B FP8 block-wise, E=256,
        # N=1024, K=2048, top_k>1). The kernel only supports M<=64; larger
        # batches (e.g. prefill) fall back to the modular kernel.
        if getattr(self, "_use_moe_monokernel", False):
            # ── EP path: the monokernel computes this rank's local expert
            # slice over the globally-gathered token tile and emits a per-rank
            # PARTIAL; the reduce-scatter combine sums the partials back to each
            # rank's own tokens. Uses the global [256] weight buffer built at
            # load time.
            #
            # We run the monokernel for the WHOLE tile (prefill AND decode) by
            # CHUNKING the gathered tokens into <=8-token groups (the validated
            # BS8 EP kernel). We do NOT use the modular Triton fallback under
            # EP: with the monokernel OFF we verified stock Triton FP8-blockwise
            # EP is INCORRECT for this model (garbage generation), while the
            # FI-CUTLASS EP backend is coherent — but FI-CUTLASS repacks the
            # weights, breaking the monokernel's raw-layout assumption. So the
            # only correct route that keeps the raw Triton layout is to run the
            # monokernel itself for every M. The per-<=8-chunk partial is the
            # exact op validated in tests/moe/test_ep_realpath_weightprep.py.
            #
            # DEADLOCK SAFETY: the gathered tile x_g is IDENTICAL on every rank
            # (all_gatherv), so M_g and the chunk count are identical across
            # ranks. Both prefill and decode now follow the SAME structure
            # (dispatch -> per-chunk compute -> combine), issuing the SAME
            # collectives on every rank — there is no divergent-branch
            # deadlock (the earlier local-M vs global-M branch mismatch is
            # gone). No device->host sync is introduced, so CUDA-graph capture
            # of the decode path (single <=8 chunk) is unaffected.
            if getattr(self, "_is_ep_monokernel", False):
                from vllm.distributed.parallel_state import get_ep_group
                _ep = get_ep_group()
                scoring_func = getattr(layer, "scoring_func", "softmax")
                renormalize = getattr(layer, "renormalize", True)
                self._fused_ar_did_reduce = False
                CHUNK = 8

                # ── SHARED-EXPERT / COMBINE OVERLAP (opt-in via
                # VLLM_MOE_EP_SHARED_OVERLAP_COMBINE=1). The shared expert is the
                # only compute in a decode MoE layer that is independent of the
                # routed all-to-all. vLLM's default aux-stream overlap already
                # hides it, but co-schedules it with the HBM-bound routed
                # monokernel, so the two contend for HBM bandwidth. Here we
                # instead run the shared expert on the aux stream ORDERED AFTER
                # the routed monokernel, so it overlaps the COMBINE collective —
                # which is NVLink/wait-bound with HBM idle. When enabled,
                # mk_owns_shared_expert returns True, so the runner skips its own
                # shared-expert calls and reads the result we store into
                # layer.shared_experts._output. Values are identical to the
                # runner's path (se._layer(x)); only the stream ordering changes.
                import os as _os_ov
                _se = getattr(layer, "shared_experts", None)
                _overlap_shared = (
                    _os_ov.environ.get("VLLM_MOE_EP_SHARED_OVERLAP_COMBINE") == "1"
                    and _se is not None
                    and getattr(_se, "_stream", None) is not None
                )

                def _nccl_combine(partial):
                    return _ep.combine(partial, is_sequence_parallel=False)

                def _combine_with_shared(partial, combine_fn=None):
                    # Combine the per-rank partials back to local tokens,
                    # optionally overlapping the shared expert with the combine.
                    # combine_fn selects the reduce-scatter implementation
                    # (NCCL by default; peer-memory when self-contained EP is on).
                    cfn = combine_fn if combine_fn is not None else _nccl_combine
                    if not _overlap_shared:
                        return cfn(partial)
                    aux = _se._stream
                    cur = torch.cuda.current_stream()
                    # Order the shared expert after ALL main-stream work so far
                    # (dispatch + routed monokernel) so it starts alongside the
                    # combine, not the routed compute.
                    x.record_stream(aux)
                    aux.wait_stream(cur)
                    with torch.cuda.stream(aux):
                        _se._output[_se._output_idx] = _se._layer(x)
                    out = cfn(partial)
                    # Main stream consumes the shared output downstream, so wait.
                    cur.wait_stream(aux)
                    return out

                def _store_shared_inline():
                    # Safety net for diagnostic early-returns that skip combine:
                    # mk_owns_shared_expert is True, so the runner will read
                    # _se._output and would assert if we left it unset.
                    if _overlap_shared:
                        _se._output[_se._output_idx] = _se._layer(x)

                # Per-rank global token layout from DP metadata (CPU-side,
                # already synchronized in coordinate_batch_across_dp — no new
                # collective, no device->host sync). Used by BOTH paths below;
                # identical on every rank, so any decision made from it is
                # DP-consistent (deadlock-safe).
                _sizes = None
                try:
                    from vllm.forward_context import get_forward_context
                    _dpmd = getattr(get_forward_context(), "dp_metadata", None)
                    if _dpmd is not None:
                        _s = _dpmd.get_chunk_sizes_across_dp_rank()
                        if _s is not None:
                            _sizes = (_s.tolist() if hasattr(_s, "tolist")
                                      else list(_s))
                except Exception:
                    _sizes = None

                # ── IN-KERNEL DISPATCH (Stage 2 — all-to-all HIDING). Opt-in
                # via VLLM_MOE_EP_INKERNEL_DISPATCH=1. Instead of all-gathering
                # the activations over NCCL, each rank stages its OWN tokens
                # into a peer-mapped buffer at their GLOBAL positions and the
                # monokernel PEER-READS the remote tokens directly (folding the
                # dispatch transfer into the kernel). Only the tiny router is
                # gathered. Decode-shaped tile only (fits the BS8 kernel and the
                # workspace); prefill uses the NCCL path below.
                #
                # CROSS-RANK ORDERING (graph-safe, no per-layer flag). The
                # monokernel peer-reads the remote tokens, so the peer's write
                # must be visible before we read. Instead of the per-layer
                # storeLL/readLL flag handshake (whose flag value is baked into a
                # CUDA graph and unstable under eager skew), we ORDER via the
                # router all-gatherv we already issue: write our activations into
                # the peer-mapped buffer FIRST, then the all-gatherv acts as a
                # cross-rank barrier (a real NCCL collective that captures and
                # replays correctly), so by the time any rank runs the kernel,
                # every rank has completed its activation write. No baked flag,
                # so this works under CUDA graphs (the regime where the hiding
                # gain actually shows). peer_ll_buffers=None disables the
                # in-kernel handshake (matches the validated M2b peer-read path).
                _inkernel = (
                    getattr(self, "_ep_inkernel_dispatch", False)
                    and getattr(self, "_ep_act_ws", None) is not None
                    and _sizes is not None
                    and sum(_sizes) <= CHUNK
                    and self._moe_ep_size == 2  # single-peer peer_activations
                )
                if _inkernel:
                    from vllm.distributed.parallel_state import get_dp_group
                    ws = self._ep_act_ws
                    rank = _ep.rank_in_group
                    M_g = int(sum(_sizes))
                    start = int(sum(_sizes[:rank]))
                    n_local = int(_sizes[rank])
                    peer = rank ^ 1
                    # 1) Stage this rank's tokens into the peer-mapped buffer at
                    #    their global row positions (BEFORE the barrier).
                    local_view, peer_views = ws.ep_activation_views(M_g)
                    local_view.zero_()
                    local_view[start:start + n_local] = x
                    # 2) Router-only all-gather (tiny: [M_g, 256] bf16). Issued
                    #    AFTER the write, so on every rank the write is stream-
                    #    ordered before this collective; the collective can't
                    #    complete until all ranks have entered it, hence all
                    #    activation writes are done -> safe to peer-read.
                    import os as _os_diag
                    if _os_diag.environ.get("VLLM_MOE_EP_NO_ROUTER_AG") == "1":
                        # DIAGNOSTIC ONLY (wrong output): drop the router
                        # all-gather to measure whether the dispatch collective
                        # is on the wall-clock critical path. Local rows only;
                        # remote router rows are left zero (garbage output), so
                        # use this to read STEP TIME, not coherence. If step
                        # time is unchanged vs the all-gather path, the collective
                        # is overlapped (not wall-critical) -> removing it for
                        # real would not help, and dispatch is exhausted.
                        rl_g = router_logits.new_zeros(
                            (M_g, router_logits.size(1)))
                        rl_g[start:start + n_local] = router_logits
                    else:
                        (rl_g,) = get_dp_group().all_gatherv(
                            [router_logits], dim=0, sizes=_sizes)
                    # 3) Kernel peer-reads the remote tokens (no flag handshake).
                    partial = torch.ops.vllm.moe_monokernel_topk(
                        local_view,
                        rl_g,
                        self._ep_w13_global,
                        self._ep_w13_scale_global,
                        self._ep_w2_global,
                        self._ep_w2_scale_global,
                        self.moe_monokernel_scratchpad,
                        layer.top_k,
                        scoring_func,
                        renormalize,
                        None,  # peer_ll_buffers (handshake disabled; barrier orders)
                        None,  # residual_in
                        None,  # residual_out
                        None,  # rms_gamma
                        0.0,   # rms_eps
                        0,                   # ll_flag (unused)
                        rank,                # tp_rank (= ep_rank)
                        self._moe_ep_size,   # tp_size (= ep_size)
                        self._moe_ep_expert_base,  # expert_base
                        True,                # ep
                        peer_views[peer],    # peer_activations (peer-mapped)
                        start,               # local_token_start
                        n_local,             # n_local_tokens
                    )
                    if _os_diag.environ.get("VLLM_MOE_EP_NO_COMBINE") == "1":
                        # DIAGNOSTIC ONLY (wrong output): skip the reduce-scatter
                        # combine to measure whether the COMBINE collective is on
                        # the wall-clock critical path. Return this rank's local
                        # token slice un-reduced (missing peer contributions ->
                        # garbage), so read STEP TIME, not coherence. If step time
                        # drops vs the combine path, combine IS wall-critical
                        # (lever 2 has headroom); if unchanged, the entire EP
                        # all-to-all is overlapped and has no E2E headroom.
                        _store_shared_inline()
                        return partial[start:start + n_local].contiguous()

                    # ── SELF-CONTAINED COMBINE (opt-in via
                    # VLLM_MOE_EP_INKERNEL_COMBINE=1). Replaces the NCCL
                    # reduce-scatter with a symmetric-memory peer-reduce: each
                    # rank stages its full [M_g, HIDDEN] partial into a distinct
                    # peer-mapped region, a single tiny all-reduce orders the
                    # writes (makes them cross-rank visible), then this rank
                    # peer-reads every peer's partial for ITS OWN token rows and
                    # sums — the reduce-scatter of the activation data now moves
                    # over NVLink peer memory, not a collective kernel. Combined
                    # with the peer-read dispatch above, the entire EP all-to-all
                    # data movement is folded into the monokernel path; only a
                    # 1-element ordering barrier remains (not the all-to-all).
                    def _combine_peer(partial):
                        cl, cp = ws.ep_combine_views(M_g)
                        cl.copy_(partial)
                        # Single ordering barrier: guarantees every rank has
                        # written its partial (and its kernel has retired) before
                        # any peer-read. A distinct region (not the dispatch one)
                        # means no rank overwrites data a peer may still read, so
                        # one barrier suffices.
                        _ep.all_reduce(partial.new_zeros(1))
                        out = partial[start:start + n_local].clone()
                        for p in range(self._moe_ep_size):
                            if p != rank:
                                out = out + cp[p][start:start + n_local]
                        return out

                    if _os_diag.environ.get(
                            "VLLM_MOE_EP_INKERNEL_COMBINE") == "1":
                        return _combine_with_shared(partial, _combine_peer)
                    return _combine_with_shared(partial)

                # ── NCCL DISPATCH (default). All-gather the full token tile
                # across DP (naive dispatch). is_sequence_parallel=False -> DP
                # group (matches the modular kernel's internal dispatch). This
                # is the transfer the in-kernel peer-read dispatch above HIDES.
                x_g, rl_g = _ep.dispatch_router_logits(
                    x, router_logits, is_sequence_parallel=False)

                def _run_chunk(xc, rc):
                    return torch.ops.vllm.moe_monokernel_topk(
                        xc,
                        rc,
                        self._ep_w13_global,
                        self._ep_w13_scale_global,
                        self._ep_w2_global,
                        self._ep_w2_scale_global,
                        self.moe_monokernel_scratchpad,
                        layer.top_k,
                        scoring_func,
                        renormalize,
                        None,  # peer_ll_buffers
                        None,  # residual_in
                        None,  # residual_out
                        None,  # rms_gamma
                        0.0,   # rms_eps
                        0,     # ll_flag
                        _ep.rank_in_group,   # tp_rank (= ep_rank)
                        self._moe_ep_size,   # tp_size (= ep_size)
                        self._moe_ep_expert_base,  # expert_base
                        True,  # ep
                        None,  # peer_activations (in-kernel dispatch: future)
                        0,     # local_token_start
                        0,     # n_local_tokens
                    )

                M_g = x_g.size(0)
                if M_g <= CHUNK:
                    # Decode-shaped tile: single call (CUDA-graph capturable,
                    # identical to the previously-verified decode path).
                    partial = _run_chunk(x_g, rl_g)
                elif (_os_ov.environ.get("VLLM_MOE_EP_PIPELINE_COMBINE") == "1"
                      and _sizes is not None):
                    # ── PIPELINED COMBINE (opt-in, large-token path). Overlap
                    # communication with compute at CHUNK granularity: each
                    # <=8-token chunk's cross-rank reduction runs on a side
                    # stream concurrently with the NEXT chunk's expert compute on
                    # the main stream, instead of one reduce-scatter after all
                    # chunks finish. A chunk is not aligned to the rank-partition
                    # boundaries, so reduce-scatter cannot be applied per chunk;
                    # we instead all-reduce each chunk over the EP group and
                    # slice this rank's own rows at the end (identical result:
                    # sum over ranks then keep own tokens == reduce-scatter). The
                    # chunk count is M_g/CHUNK on every rank, so the collective
                    # sequence is identical and the run is deadlock-safe. At
                    # large token counts the expert compute is large enough to
                    # hide the per-chunk reduction behind it; at decode (single
                    # chunk) there is nothing to overlap, hence the M_g>CHUNK
                    # gate. Only the final chunk's reduction stays exposed.
                    if getattr(self, "_ep_pipeline_stream", None) is None:
                        # Created during eager warmup (before CUDA-graph capture).
                        self._ep_pipeline_stream = torch.cuda.Stream()
                    comm = self._ep_pipeline_stream
                    cur = torch.cuda.current_stream()
                    K_g = x_g.size(1)
                    E_g = rl_g.size(1)
                    rank = _ep.rank_in_group
                    start = int(sum(_sizes[:rank]))
                    n_local = int(_sizes[rank])
                    red_parts = []
                    for s in range(0, M_g, CHUNK):
                        e = min(s + CHUNK, M_g)
                        n = e - s
                        if n == CHUNK:
                            xc = x_g[s:e].contiguous()
                            rc = rl_g[s:e].contiguous()
                        else:
                            xc = x_g.new_zeros((CHUNK, K_g))
                            rc = rl_g.new_zeros((CHUNK, E_g))
                            xc[:n] = x_g[s:e]
                            rc[:n] = rl_g[s:e]
                        pc = _run_chunk(xc, rc)          # main-stream compute
                        # Reduce chunk on the side stream, overlapping the next
                        # chunk's compute. record_stream keeps pc alive across
                        # the stream boundary.
                        pc.record_stream(comm)
                        comm.wait_stream(cur)
                        with torch.cuda.stream(comm):
                            red_parts.append(_ep.all_reduce(pc)[:n])
                    cur.wait_stream(comm)
                    reduced = torch.cat(red_parts, dim=0)  # [M_g, H], summed
                    # Slice this rank's own tokens (== reduce-scatter result).
                    # Shared expert is added by the runner (mk_owns=False).
                    return reduced[start:start + n_local].contiguous()
                else:
                    # Prefill: slice into <=8-token chunks, padding the last
                    # chunk up to the kernel's fixed BS8 tile (pad rows carry
                    # zero activations -> zero output, and are sliced off). The
                    # chunk count is a function of M_g only, so it is identical
                    # on every rank.
                    K_g = x_g.size(1)
                    E_g = rl_g.size(1)
                    out_parts = []
                    for s in range(0, M_g, CHUNK):
                        e = min(s + CHUNK, M_g)
                        n = e - s
                        if n == CHUNK:
                            xc = x_g[s:e].contiguous()
                            rc = rl_g[s:e].contiguous()
                        else:
                            xc = x_g.new_zeros((CHUNK, K_g))
                            rc = rl_g.new_zeros((CHUNK, E_g))
                            xc[:n] = x_g[s:e]
                            rc[:n] = rl_g[s:e]
                        pc = _run_chunk(xc, rc)
                        out_parts.append(pc[:n])
                    partial = torch.cat(out_parts, dim=0)

                # Reduce-scatter the per-rank partials back to local tokens
                # (optionally overlapping the shared expert with the combine).
                return _combine_with_shared(partial)
            M = x.size(0)
            if M <= 8:
                # Scratchpad was moved to the weight device at load time
                # (see process_weights_after_loading) so no host->device
                # copy happens here — that would be illegal during CUDA
                # graph capture.
                top_k = layer.top_k
                scoring_func = getattr(layer, "scoring_func", "softmax")
                renormalize = getattr(layer, "renormalize", True)

                # Fused AR params
                from vllm.distributed.parallel_state import (
                    get_tensor_model_parallel_world_size,
                    get_tensor_model_parallel_rank,
                )
                _tp_size = get_tensor_model_parallel_world_size()
                _tp_rank = get_tensor_model_parallel_rank()

                peer_ll_buffers = None
                ll_flag = 0
                if hasattr(self, "_moe_ll_workspace") and self._moe_ll_workspace is not None:
                    peer_ll_buffers = self._moe_ll_workspace.peer_ll_buffers
                    ll_flag = self._moe_ll_workspace.next_flag()
                    # The kernel's fused AR already all-reduced the output.
                    # Tell the runner to skip the external NCCL all-reduce.
                    self._fused_ar_did_reduce = True
                else:
                    self._fused_ar_did_reduce = False

                # Check if fused residual+RMSNorm is available via side-channel
                from vllm.model_executor.layers.fused_moe.monokernel_fused_norm import (
                    get_fused_norm_inputs,
                    mark_fused,
                )
                residual_in, residual_out, rms_gamma, rms_eps = get_fused_norm_inputs()
                # Mark fusion if all inputs are available
                if residual_in is not None and rms_gamma is not None:
                    mark_fused()

                return torch.ops.vllm.moe_monokernel_topk(
                    x,
                    router_logits,
                    layer.w13_weight,
                    getattr(layer, f"w13_{self.weight_scale_name}"),
                    layer.w2_weight,
                    getattr(layer, f"w2_{self.weight_scale_name}"),
                    self.moe_monokernel_scratchpad,
                    top_k,
                    scoring_func,
                    renormalize,
                    peer_ll_buffers,
                    residual_in,
                    residual_out,
                    rms_gamma,
                    rms_eps if rms_eps else 0.0,
                    ll_flag,
                    _tp_rank,
                    _tp_size,
                )
            # Fall back to the modular kernel for large batches.
            self._fused_ar_did_reduce = False
            return self._apply_modular_fallback(layer, x, router_logits)

        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            x,
            layer.w13_weight,
            layer.w2_weight,
            router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
        )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts_input=shared_experts_input,
        )


# TODO(future PR): remove this class in favor of
# online/fp8.py::Fp8PerTensorOnlineMoEMethod
class Fp8OnlineMoEMethod(Fp8MoEMethod):
    """MoE method for online FP8 quantization.
    Supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    uses_meta_device: bool = True

    def __init__(self, quant_config: Fp8Config, layer: torch.nn.Module):
        super().__init__(quant_config, layer)
        assert not quant_config.is_checkpoint_fp8_serialized
        assert quant_config.activation_scheme == "dynamic"
        assert quant_config.weight_block_size is None

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                device="meta",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                device="meta",  # materialized and processed during loading
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # BIASES (for models like GPT-OSS that have biased MoE)
        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    device="meta",  # materialized and processed during loading
                    dtype=layer.orig_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    hidden_size,
                    device="meta",  # materialized and processed during loading
                    dtype=layer.orig_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

        initialize_online_processing(layer)

    def process_weights_after_loading(self, layer: Module) -> None:
        # TODO(@ksayers): inplace fp8 quant kernel, initialize scales with ones
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        fp8_dtype = current_platform.fp8_dtype()
        w13 = torch.empty_like(layer.w13_weight, dtype=fp8_dtype)
        w2 = torch.empty_like(layer.w2_weight, dtype=fp8_dtype)
        w13_scale = torch.ones(
            layer.num_experts, device=w13.device, dtype=torch.float32
        )
        w2_scale = torch.ones(layer.num_experts, device=w2.device, dtype=torch.float32)
        layer.w13_input_scale = None
        layer.w2_input_scale = None

        for expert in range(layer.local_num_experts):
            w13[expert, :, :], w13_scale[expert] = ops.scaled_fp8_quant(
                layer.w13_weight[expert, :, :]
            )
            w2[expert, :, :], w2_scale[expert] = ops.scaled_fp8_quant(
                layer.w2_weight[expert, :, :]
            )

        # Shuffle weights to runtime format and setup kernel.
        self._setup_kernel(
            layer,
            w13,
            w2,
            w13_scale,
            w2_scale,
            w13_input_scale=layer.w13_input_scale,
            w2_input_scale=layer.w2_input_scale,
        )

        # Prevent duplicate processing (e.g., during weight reload)
        layer._already_called_process_weights_after_loading = True


class Fp8KVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from FP8 checkpoints.
    """

    def __init__(self, quant_config: Fp8Config):
        super().__init__(quant_config)
