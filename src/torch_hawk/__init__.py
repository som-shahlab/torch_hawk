from __future__ import annotations

from . import _torch_hawk
import torch
import math
import einops

from typing import Iterable, Optional, cast

class Conv1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        assert x.device == w.device == s.device
        assert x.dtype == w.dtype

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
            assert x.dtype == torch.float16
            
            _torch_hawk.conv1d_forward_cuda(x.detach(), w.detach(), s, o, n, k)
        else:
            assert x.dtype == torch.float32

            _torch_hawk.conv1d_forward_cpu(x.detach(), w.detach(), s, o, n, k)

        return o
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Iterable[Optional[torch.Tensor]]:
        x, w, s = ctx.saved_tensors

        n = x.shape[0]
        k = x.shape[1]

        grad_x = torch.empty_like(x)
        grad_w = torch.empty_like(w, dtype=torch.float32)


        if x.device.type == 'cuda':
            _torch_hawk.conv1d_backward_cuda(x.detach(), w.detach(), s, grad_output, grad_x, grad_w, n, k)
        else:
            _torch_hawk.conv1d_backward_cpu(x.detach(), w.detach(), s, grad_output, grad_x, grad_w, n, k)


        grad_w = grad_w.type(torch.float16)

        return grad_x, grad_w, None
    

def conv1d(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, Conv1dFunction.apply(x, w, s))


class Conv1d(torch.nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty((k , _torch_hawk.conv1d_kernel_size()), dtype=torch.float16))

        bound = math.sqrt(1 / _torch_hawk.conv1d_kernel_size())

        with torch.no_grad():
            self.w.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        assert s.dtype == torch.uint8

        if x.device.type == 'cuda':
            return conv1d(x.type(torch.float16), self.w.type(torch.float16), s)
        else:
            return conv1d(x, self.w, s)
    

class LinearRecurrenceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        assert a.device == x.device
        assert a.dtype == x.dtype
        
        n = a.shape[0]
        k = a.shape[1]

        assert a.shape == (n, k)
        assert x.shape == (n, k)

        o = torch.empty_like(x)

        if a.device.type == 'cuda':
            assert a.dtype == torch.float16

            s = torch.empty(size=_torch_hawk.linear_recurrence_scratch_space(n, k), device=a.device, dtype=torch.float32)

            _torch_hawk.linear_recurrence_forward_cuda(a.detach(), x.detach(), o, s, n, k)
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

        grad_a = torch.empty_like(a)
        grad_x = torch.empty_like(a)


        if a.device.type == 'cuda':
            s = torch.empty(size=_torch_hawk.linear_recurrence_scratch_space(n, k), device=a.device, dtype=torch.float32)
            _torch_hawk.linear_recurrence_backward_cuda(a.detach(), o.detach(), s, grad_a, grad_x, grad_output, n, k)
        else:
            _torch_hawk.linear_recurrence_backward_cpu(a.detach(), o.detach(), grad_a, grad_x, grad_output, n, k)


        return grad_a, grad_x
    
def linear_recurrence(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, LinearRecurrenceFunction.apply(a, x))



# The following code is adapted from https://github.com/google-deepmind/recurrentgemma/blob/main/recurrentgemma/torch/layers.py#L216
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

class BlockDiagonalLinear(torch.nn.Module):
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
        self.w = torch.nn.Parameter(torch.empty(
            [self.num_blocks, self.block_width, self.block_width],
            device=device,
            dtype=dtype
        ))
        self.b = torch.nn.Parameter(torch.empty(
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

    def forward(self, x):
        """Calls the BlockDiagonalLinear."""
        # Split x to blocks.
        x = einops.rearrange(x, "... (h i) -> ... h i", h=self.num_blocks)

        # Linear layer over each block + bias.
        y = torch.einsum("... h i, h i j -> ... h j", x, self.w) + self.b

        # Flatten the output.
        return einops.rearrange(y, "... h j -> ... (h j)", h=self.num_blocks)


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



class RGLRU(torch.nn.Module):
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
        self.a_param = torch.nn.Parameter(torch.empty(
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


    def forward(
        self,
        x,
        s,
    ):
        """Calls the RG-LRU.

        Args:
        x: Sequence of input activations.
        segment_pos: Position of each token in the sequence.
        cache: The previous hidden state of the RG-LRU.
        return_cache: Whether to compute and return the updated cache.

        Returns:
        Output of the block together with the updated hidden state.
        """

        n, k = x.shape
        assert s.shape == (n,)

        assert s.dtype == torch.uint8

        reset = torch.ones_like(s, dtype=torch.bool)
        reset[1:] = s[1:] != s[:-1]

        # Gates for x and a.
        gate_x = torch.sigmoid(self.input_gate(x))
        gate_a = torch.sigmoid(self.a_gate(x))

        # Compute the parameter `A` of the recurrence.
        log_a = -8.0 * gate_a * torch.nn.functional.softplus(self.a_param)
        a = torch.exp(log_a)
        a_square = torch.exp(2 * log_a)

        # Gate the input.
        gated_x = x * gate_x

        # Apply gamma normalization to the input. We need to clip the derivatives of
        # `sqrt` in order to prevent NaNs during training in bfloat16.
        multiplier = SqrtBoundDerivative.apply(1 - a_square)
        multiplier = reset[..., None] + ~reset[..., None] * multiplier
        normalized_x = gated_x * multiplier.type(x.dtype)

        if a.device.type == 'cuda':
            return linear_recurrence(a=a.type(torch.float16), x=normalized_x.type(torch.float16))
        else:
            return linear_recurrence(a=a.type(torch.float32), x=normalized_x.type(torch.float32))
