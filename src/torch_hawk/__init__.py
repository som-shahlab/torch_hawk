from __future__ import annotations

from . import _torch_hawk
import torch
import math
import einops

from typing import Iterable, Optional, cast

from torch.amp import custom_fwd, custom_bwd

class Conv1dFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        assert x.device == w.device == s.device

        x = x.contiguous()
        w = w.contiguous()
        s = s.contiguous()

        if x.dtype != w.dtype:
           w = w.type(x.dtype)

        assert x.dtype == w.dtype, f'Wrong type {x.dtype} {w.dtype}'

        ctx.save_for_backward(x, w, s)

        n = x.shape[0]
        k = x.shape[1]

        assert x.shape == (n, k)
        assert w.shape == (k, _torch_hawk.conv1d_kernel_size())
        assert s.shape == (n,)

        assert k % 2 == 0
        assert s.dtype == torch.uint8

        o = torch.empty_like(x)

        if x.device.type == 'cuda':
            assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)

            if x.dtype == torch.bfloat16:
                _torch_hawk.conv1d_forward_cuda_bf16(x.detach(), w.detach(), s, o, n, k)
            elif x.dtype == torch.float16:
                _torch_hawk.conv1d_forward_cuda_fp16(x.detach(), w.detach(), s, o, n, k)
            elif x.dtype == torch.float32:
                _torch_hawk.conv1d_forward_cuda_fp32(x.detach(), w.detach(), s, o, n, k)
               
        else:
            assert x.dtype == torch.float32

            _torch_hawk.conv1d_forward_cpu(x.detach(), w.detach(), s, o, n, k)

        return o
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Iterable[Optional[torch.Tensor]]:
        x, w, s = ctx.saved_tensors

        n = x.shape[0]
        k = x.shape[1]

        grad_output = grad_output.contiguous()

        grad_x = torch.empty_like(x)
        grad_w = torch.zeros_like(w, dtype=torch.float32)


        if x.device.type == 'cuda':
            assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        
            if x.dtype == torch.bfloat16:
                _torch_hawk.conv1d_backward_cuda_bf16(x.detach(), w.detach(), s, grad_output, grad_x, grad_w, n, k)
            elif x.dtype == torch.float16:
                _torch_hawk.conv1d_backward_cuda_fp16(x.detach(), w.detach(), s, grad_output, grad_x, grad_w, n, k)
            elif x.dtype == torch.float32:
                _torch_hawk.conv1d_backward_cuda_fp32(x.detach(), w.detach(), s, grad_output, grad_x, grad_w, n, k)    
        else:
            _torch_hawk.conv1d_backward_cpu(x.detach(), w.detach(), s, grad_output, grad_x, grad_w, n, k)

        grad_w = grad_w.type(x.dtype)

        return grad_x, grad_w, None
    

def conv1d(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, Conv1dFunction.apply(x, w, s))
    

class LinearRecurrenceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        assert a.device == x.device
        assert a.dtype == x.dtype
        
        n = a.shape[0]
        k = a.shape[1]

        assert a.shape == (n, k)
        assert x.shape == (n, k)

        a = a.contiguous()
        x = x.contiguous()

        o = torch.empty_like(x)

        if a.device.type == 'cuda':
            assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)

            s = torch.empty(size=_torch_hawk.linear_recurrence_scratch_space(n, k), device=a.device, dtype=torch.float32)

            if x.dtype == torch.bfloat16:
                _torch_hawk.linear_recurrence_forward_cuda_bf16(a.detach(), x.detach(), o, s, n, k)
            elif x.dtype == torch.float16:
                _torch_hawk.linear_recurrence_forward_cuda_fp16(a.detach(), x.detach(), o, s, n, k)
            elif x.dtype == torch.float32:
                _torch_hawk.linear_recurrence_forward_cuda_fp32(a.detach(), x.detach(), o, s, n, k)
        else:
            assert a.dtype == torch.float32

            _torch_hawk.linear_recurrence_forward_cpu(a.detach(), x.detach(), o, n, k)

        ctx.save_for_backward(a, o)

        return o
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Iterable[Optional[torch.Tensor]]:
        a, o = ctx.saved_tensors

        n = a.shape[0]
        k = a.shape[1]

        grad_output = grad_output.contiguous()

        grad_a = torch.empty_like(a)
        grad_x = torch.empty_like(a)

        if a.device.type == 'cuda':
            assert a.dtype in (torch.bfloat16, torch.float16, torch.float32)

            s = torch.empty(size=_torch_hawk.linear_recurrence_scratch_space(n, k), device=a.device, dtype=torch.float32)
            
            if a.dtype == torch.bfloat16:
                _torch_hawk.linear_recurrence_backward_cuda_bf16(a.detach(), o.detach(), s, grad_a, grad_x, grad_output, n, k)
            elif a.dtype == torch.float16:
                _torch_hawk.linear_recurrence_backward_cuda_fp16(a.detach(), o.detach(), s, grad_a, grad_x, grad_output, n, k)
            elif a.dtype == torch.float32:
                _torch_hawk.linear_recurrence_backward_cuda_fp32(a.detach(), o.detach(), s, grad_a, grad_x, grad_output, n, k)

        else:
            _torch_hawk.linear_recurrence_backward_cpu(a.detach(), o.detach(), grad_a, grad_x, grad_output, n, k)


        return grad_a, grad_x
    
def linear_recurrence(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, LinearRecurrenceFunction.apply(a, x))



# The following code is adapted from https://github.com/google-deepmind/recurrentgemma/blob/main/recurrentgemma/torch/layers.py#L216
import math
from typing import Literal, NamedTuple, Optional, overload

import einops
from recurrentgemma import common
from recurrentgemma.torch import array_typing as at
from recurrentgemma.torch import layers
import torch
from torch import nn
import torch_hawk

_MAX_SQRT_GRADIENT = 1000.0

class SqrtBoundDerivative(torch.autograd.Function):
  """Computes a square root with a gradient clipped at `_MAX_SQRT_GRADIENT`."""

  @staticmethod
  def forward(ctx, x: torch.Tensor) -> torch.Tensor:
    """The forward pass, which is a normal `sqrt`."""
    ctx.save_for_backward(x)
    return torch.sqrt(x)

  @staticmethod
  def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
    """The backward pass, which clips the `sqrt` gradient."""
    (x,) = ctx.saved_tensors
    clipped_x_times_4 = torch.clip(4.0 * x, min=1 / (_MAX_SQRT_GRADIENT**2))
    return grad_output / torch.sqrt(clipped_x_times_4)


def gelu(x: torch.Tensor) -> torch.Tensor:
  """Returns the GELU activation function with the same approximation as JAX."""
  return nn.functional.gelu(x, approximate="tanh")

def rnn_param_init(
    tensor: torch.Tensor,
    min_rad: float,
    max_rad: float,
    transform: str = "softplus",
    eps: float = 1e-8,
) -> torch.Tensor:
  """Initializes the `A` parameter of the RG-LRU uniformly on a ring."""
  with torch.no_grad():
    # Proportional to area in a ring.
    # 0.5 * jnp.log(unif * (max_rad**2 - min_rad**2) + min_rad**2 + 1e-8)
    tensor.uniform_(min_rad ** 2 + eps, max_rad ** 2 + eps)
    tensor.log_().mul_(0.5)

    if transform == "softplus":
      # Inverse transform.
      # jnp.log(jnp.exp(-a_real) - 1.0).astype(dtype)
      return tensor.neg_().exp_().sub_(1.0).log_()
    else:
      raise NotImplementedError()


class BlockDiagonalLinear(nn.Module):
  """Block-diagonal linear layer."""

  def __init__(
      self,
      width: int,
      num_blocks: int,
      w_init_variance_scale: float = 1.0,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    """Initializes the BlockDiagonalLinear.

    Args:
      width: The number of dimensions of the input and output.
      num_blocks: The number of diagonal blocks in the layer.
      w_init_variance_scale: A parameters that scales the variance of the
        initialization of the weights.
      device: On what device to initialize parameters. Needed to allow for
        initializing the module without parameter initialzation.
      dtype: What dtype to use for initialziation.
    """
    super().__init__()
    self.width = width
    self.num_blocks = num_blocks
    self.w_init_variance_scale = w_init_variance_scale
    self.block_width = self.width // self.num_blocks

    # Parameters.
    self.w = nn.Parameter(torch.empty(
        [self.num_blocks, self.block_width, self.block_width],
        device=device,
        dtype=dtype
    ))
    self.b = nn.Parameter(torch.empty(
        [self.num_blocks, self.block_width], device=device, dtype=dtype
    ))

    # Initialization.
    self.reset_parameters()

  def reset_parameters(self) -> None:
    """Resets the parameters of the module."""
    self.w_init_(self.w)
    torch.nn.init.zeros_(self.b)

  def w_init_(self, w: torch.Tensor) -> None:
    """Initializes the weight `w` of the layer."""
    std = math.sqrt(self.w_init_variance_scale / self.block_width)
    torch.nn.init.normal_(w, mean=0.0, std=std)

  @at.typed
  def forward(self, x: at.ExpandedActivations) -> at.ExpandedActivations:
    """Calls the BlockDiagonalLinear."""
    # Split x to blocks.
    x = einops.rearrange(x, "... (h i) -> ... h i", h=self.num_blocks)

    # Linear layer over each block + bias.
    y = torch.einsum("... h i, h i j -> ... h j", x, self.w) + self.b

    # Flatten the output.
    return einops.rearrange(y, "... h j -> ... (h j)", h=self.num_blocks)



class RGLRU(nn.Module):
  """A Real-Gated Linear Recurrent Unit (RG-LRU) layer."""

  def __init__(
      self,
      width: int,
      num_heads: int,
      w_init_variance_scale: float = 1.0,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    """Initializes the RG-LRU.

    Args:
      width: The number of dimensions of the input and output.
      num_heads: The number of diagonal blocks in the input and A gate layers.
      w_init_variance_scale: Initialization parameter for the
        BlockDiagonalLinear layers of the gates. See the `BlockDiagonalLinear`
        layer for details.
      device: On what device to initialize parameters. Needed to allow for
        initializing the module without parameter initialzation.
      dtype: What dtype to use for initialziation.
    """
    super().__init__()
    self.width = width
    self.num_heads = num_heads
    self.w_init_variance_scale = w_init_variance_scale

    # Parameters and layers.
    self.a_param = nn.Parameter(torch.empty(
        [self.width], device=device, dtype=dtype
    ))
    self.input_gate = BlockDiagonalLinear(
        width=self.width,
        num_blocks=self.num_heads,
        w_init_variance_scale=w_init_variance_scale,
        device=device,
        dtype=dtype,
    )
    self.a_gate = BlockDiagonalLinear(
        width=self.width,
        num_blocks=self.num_heads,
        w_init_variance_scale=self.w_init_variance_scale,
        device=device,
        dtype=dtype,
    )

    # Initialization
    self.reset_parameters()

  def reset_parameters(self) -> None:
    """Resets the parameters of the module."""
    self.input_gate.reset_parameters()
    self.a_gate.reset_parameters()
    self.a_param_init(self.a_param)

  def a_param_init(self, w: torch.Tensor) -> torch.Tensor:
    """Initializes the `A` parameter of the RG-LRU."""
    return rnn_param_init(w, min_rad=0.9, max_rad=0.999)
  
  @at.typed
  def forward(
      self,
      x: at.ExpandedActivations,
      s: Optional[at.SegmentPos],
  ) -> at.ExpandedActivations:
    """Calls the RG-LRU.

    Args:
      x: Sequence of input activations.
      segment_pos: Position of each token in the sequence.
      cache: The previous hidden state of the RG-LRU.
      return_cache: Whether to compute and return the updated cache.

    Returns:
      Output of the block together with the updated hidden state.

    """
    if s is None:
        reset = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)
        reset[:, 0] = True
    else:
        reset = torch.ones((x.shape[0]), dtype=torch.bool, device=x.device)
        reset[1:] = (s[1:] != s[:-1])

    # Gates for x and a.
    gate_x = torch.sigmoid(self.input_gate(x))
    gate_a = torch.sigmoid(self.a_gate(x))

    # Compute the parameter `A` of the recurrence.
    log_a = -8.0 * gate_a * nn.functional.softplus(self.a_param)
    a = torch.exp(log_a)
    a_square = torch.exp(2 * log_a)

    # Gate the input.
    gated_x = x * gate_x

    # Apply gamma normalization to the input. We need to clip the derivatives of
    # `sqrt` in order to prevent NaNs during training in bfloat16.
    multiplier = SqrtBoundDerivative.apply(1 - a_square)
    multiplier = reset[..., None] + ~reset[..., None] * multiplier
    normalized_x = gated_x * multiplier.type(x.dtype)

    a = a * ~reset[..., None]

    if s is None:
        a = a.reshape((a.shape[0] * a.shape[1], a.shape[2]))
        normalized_x = normalized_x.reshape((normalized_x.shape[0] * normalized_x.shape[1], normalized_x.shape[2]))
    
    y = torch_hawk.linear_recurrence(x=normalized_x, a=a)
    
    y = y.reshape(x.shape)

    return y
  

class Conv1D(nn.Module):
  """A 1D temporal convolution layer."""

  def __init__(
      self,
      width: int,
      temporal_width: int,
      w_init_variance_scale: float = 0.01,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    """Initializes the Conv1D.

    Args:
      width: The number of features for both inputs and outputs.
      temporal_width: The size of the temporal receptive field of the
        convolution. In other words, how much back in time the convolution can
        look to produce an output.
      w_init_variance_scale: A parameter that scales the variance of the
        initialization of the weights.
      device: On what device to initialize parameters. Needed to allow for
        initializing the module without parameter initialzation.
      dtype: What dtype to use for initialziation.
    """
    super().__init__()
    self.width = width
    self.temporal_width = temporal_width
    self.w_init_variance_scale = w_init_variance_scale

    # Parameters.
    self.w = nn.Parameter(torch.empty(
        [self.temporal_width, self.width], device=device, dtype=dtype
    ))
    self.b = nn.Parameter(torch.empty([width], device=device, dtype=dtype))

    # Initialization.
    self.reset_parameters()

  def reset_parameters(self) -> None:
    """Resets the parameters of the module."""
    self.w_init_(self.w)
    torch.nn.init.zeros_(self.b)

  def w_init_(self, w: torch.Tensor) -> None:
    """Initializes the weight matrix `w` of the Conv1D."""
    std = math.sqrt(self.w_init_variance_scale / self.temporal_width)
    torch.nn.init.normal_(w, mean=0.0, std=std)

  @at.typed
  def forward(
      self,
      x: at.ExpandedActivations,
      s: Optional[at.SegmentPos],
  ) -> at.ExpandedActivations:
    """Calls the Conv1D.

    Args:
      x: Sequence of input activations.
      segment_pos: Position of each token in the sequence.
      cache: The cache containing the previous `self.temporal_width-1` inputs
        This is set to `None` in training mode.
      return_cache: Whether to compute and return the updated cache.

    Returns:
      The output of the convolution and the updated state.
    """
    if s is None:
        s = torch.arange(0, x.shape[0], device=x.device, dtype=torch.uint8)
        s = s.unsqueeze(-1)
        s = torch.tile(s, [1, x.shape[1]])
        s = s.reshape(x.shape[0] * x.shape[1])

        x_copy = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        convolution_output = torch_hawk.conv1d(x_copy, self.w.T, s) + self.b
        convolution_output = convolution_output.reshape(x.shape)
    else:
        convolution_output = torch_hawk.conv1d(x, self.w.T, s) + self.b

    return convolution_output

class RecurrentBlock(nn.Module):
  """Griffin and Hawk's recurrent block."""

  def __init__(
      self,
      width: int,
      num_heads: int,
      lru_width: int | None = None,
      conv1d_temporal_width: int = 4,
      final_w_init_variance_scale: float = 1.0,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    """Initializes the recurrent block.

    Args:
      width: The width of the block.
      num_heads: The number of RG-LRU heads/blocks to use.
      lru_width: Internal dimension to be projected into for RG-LRU to operate
        on.
      conv1d_temporal_width: The temporal width of the 1d convolution.
      final_w_init_variance_scale: The scale for the initialization of the last
        layer of the block.
      device: On what device to initialize parameters. Needed to allow for
        initializing the module without parameter initialization.
      dtype: What dtype to use for initialziation.
    """
    super().__init__()
    self.width = width
    self.num_heads = num_heads
    self.lru_width = lru_width or width
    self.conv1d_temporal_width = conv1d_temporal_width
    self.final_w_init_variance_scale = final_w_init_variance_scale

    # Layers.
    self.linear_y = nn.Linear(
        in_features=self.width,
        out_features=self.lru_width,
        device=device,
        dtype=dtype,
    )
    self.linear_x = nn.Linear(
        in_features=self.width,
        out_features=self.lru_width,
        device=device,
        dtype=dtype,
    )
    self.linear_out = nn.Linear(
        in_features=self.lru_width,
        out_features=self.width,
        device=device,
        dtype=dtype,
    )
    self.conv_1d = Conv1D(
        width=self.lru_width,
        temporal_width=self.conv1d_temporal_width,
        device=device,
        dtype=dtype,
    )
    self.rg_lru = RGLRU(
        width=self.lru_width,
        num_heads=self.num_heads,
        device=device,
        dtype=dtype,
    )

    # Initialization.
    self.reset_parameters()

  def reset_parameters(self) -> None:
    """Resets the parameters of the module."""
    self.w_init_(self.linear_x.weight)
    torch.nn.init.zeros_(self.linear_x.bias)
    self.w_init_(self.linear_y.weight)
    torch.nn.init.zeros_(self.linear_y.bias)
    self.out_w_init_(self.linear_out.weight)
    torch.nn.init.zeros_(self.linear_out.bias)
    self.conv_1d.reset_parameters()
    self.rg_lru.reset_parameters()

  def w_init_(self, w: torch.Tensor) -> None:
    """Initializes the weights of the linear x and y layers of the block."""
    torch.nn.init.normal_(w, mean=0.0, std=math.sqrt(1.0 / self.width))

  def out_w_init_(self, w: torch.Tensor) -> None:
    """Initializes the weights of the last layer of the block."""
    std = math.sqrt(self.final_w_init_variance_scale / self.lru_width)
    torch.nn.init.normal_(w, mean=0.0, std=std)

  @at.typed
  def forward(
      self,
      x: at.Activations,
      s: Optional[at.Activations] = None,
  ) -> at.Activations:
    """Calls the recurrent block.

    Args:
      x: Sequence of input activations.
      segment_pos: Position of each token in the sequence.
      cache: Optional cache with the previous state of the RG-LRU and Conv1D.
      return_cache: Whether to compute and return the updated cache.

    Returns:
      Output of the block together with the updated cache. If `cache` is None
      than the returned updated cache is empty initialized and filled in from
      the input sequence.
    """
    # y branch.
    y = self.linear_y(x)
    y = gelu(y)

    # x branch.
    x = self.linear_x(x)
    x = self.conv_1d(
        x=x,
        s=s,
    )
    x = self.rg_lru(
        x=x,
        s=s,
    )

    # Join branches.
    x = x * y
    x = self.linear_out(x)

    return x