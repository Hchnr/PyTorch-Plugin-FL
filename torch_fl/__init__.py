import sys

from accelerator.maca._maca_cudart_shim import ensure_cudart_shim

ensure_cudart_shim()

import torch  # noqa: E402


if sys.platform == "win32":
    from ._utils import _load_dll_libraries

    _load_dll_libraries()
    del _load_dll_libraries


# Apply MACA compatibility patches before importing FlagGems.
# On MetaX (Muxi) hardware, PyTorch's bundled CUDA 12.x runtime is
# ABI-incompatible with MACA's cu-bridge (CUDA 11.6). This patches
# torch.cuda.get_device_properties/get_device_name to use MACA's
# native mcruntime API, allowing FlagGems initialization to succeed.
from accelerator.maca._maca_compat import is_maca_available, patch_torch_cuda_for_maca  # noqa: E402

if is_maca_available():
    patch_torch_cuda_for_maca()


import torch_fl._C  # type: ignore[misc]  # noqa: E402
import torch_fl.flagos  # noqa: E402


torch.utils.rename_privateuse1_backend("flagos")
torch._register_device_module("flagos", torch_fl.flagos)
torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)


# Check whether FlagGems C++ operator registration was compiled in.
# When FLAGGEMS_AVAILABLE was defined at build time, FlagGemsRegistration.cpp
# already registered all operators via TORCH_LIBRARY_IMPL at .so load time,
# so no Python-level registration is needed.
def _cpp_registration_active() -> bool:
    """Return True if the C++ wrapper registered FlagGems ops at load time."""
    try:
        return bool(torch_fl._C._flaggems_cpp_registration)
    except AttributeError:
        return False


_USE_CPP_REGISTRATION = _cpp_registration_active()

# Ops registered in C++ (FlagosMinimal.cpp) — always skip in Python path
_CPP_REGISTERED_OPS = {
    "empty.memory_format",
    "empty_strided",
    "copy_",
    "_to_copy",
    "contiguous",
    "clone",
}

# Ops that use torch_device_fn.device(device) with explicit device parameter.
# These don't work with the flagos device and must use cpu_fallback instead.
# Only relevant when falling back to Python-level registration.
_EXCLUDED_OPS = _CPP_REGISTERED_OPS | {
    # Factory functions that take a device parameter
    "randn",
    "randn_like",
    "rand",
    "rand_like",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "full",
    "full_like",
    "arange",
    "arange.start",
    "arange.start_step",
    "linspace",
    "logspace",
    "eye",
    "eye.m",
    "randperm",
    # Random ops that use device context
    "uniform_",
    "normal.float_Tensor",
    "normal.Tensor_float",
    "normal.Tensor_tensor",
    "exponential_",
    "multinomial",
    # log_softmax - FlagGems Triton kernel exceeds MACA's 4KB/thread private memory
    # limit on large vocab (e.g. Qwen3 151k). Use Python decomposition instead.
    "_log_softmax",
    "_log_softmax_backward_data",
    # pad / constant_pad_nd: FlagGems pad() redispatches to CompositeImplicitAutograd
    # (when grad is enabled) which decomposes into constant_pad_nd, but constant_pad_nd
    # is also registered to PrivateUse1, causing infinite recursion.
    "pad",
    "constant_pad_nd",
}

# When C++ registration is active, these ops are already registered by
# FlagGemsRegistration.cpp and must not be re-registered from Python.
_CPP_FLAGGEMS_OPS = {
    "addmm", "mm", "bmm",
    "sum", "sum.dim_IntList",
    "max", "max.dim", "max.dim_max",
    "rms_norm", "fused_add_rms_norm",
    "cat", "embedding", "embedding_backward",
    "argmax", "nonzero", "topk",
    "div.Tensor", "div_.Tensor", "div.Scalar", "div_.Scalar",
    "div.Tensor_mode", "div_.Tensor_mode", "div.Scalar_mode", "div_.Scalar_mode",
    "floor_divide", "floor_divide_.Tensor", "floor_divide.Scalar", "floor_divide_.Scalar",
    "divide.Tensor", "divide_.Tensor", "divide.Scalar", "divide_.Scalar",
    "divide.Tensor_mode", "divide_.Tensor_mode", "divide.Scalar_mode", "divide_.Scalar_mode",
    "true_divide.Tensor", "true_divide_.Tensor",
    "remainder.Scalar", "remainder_.Scalar", "remainder.Tensor", "remainder_.Tensor",
    "remainder.Scalar_Tensor",
    "sort", "sort.stable",
    "fill.Scalar", "fill.Tensor", "fill_.Scalar", "fill_.Tensor",
    "softmax", "softmax_backward",
    "to_copy", "copy_",
    "zeros", "exponential_",
}


def _patch_cuda_device_context():
    """
    Monkey-patch torch.cuda.device to handle flagos devices.

    FlagGems internally calls torch_device_fn.device(tensor.device), but when
    tensor.device is 'flagos:0', torch.cuda.device() fails because it expects
    a CUDA device. This patch wraps torch.cuda.device.__init__ to extract just
    the device index from flagos/privateuseone devices.
    """
    _original_cuda_device_init = torch.cuda.device.__init__

    def _patched_cuda_device_init(self, device):
        # Handle flagos/privateuseone devices by extracting just the index
        if hasattr(device, "type") and hasattr(device, "index"):
            if device.type in ("privateuseone", "flagos"):
                device = device.index if device.index is not None else 0
        return _original_cuda_device_init(self, device)

    torch.cuda.device.__init__ = _patched_cuda_device_init


# Patch torch.cuda.device before FlagGems is used
_patch_cuda_device_context()


# Global library instance to keep Python-level registrations alive
_flaggems_lib = None
_registered_ops: list = []


def _register_flaggems_operators_python():
    """
    Python-level fallback: register FlagGems operators for PrivateUse1.

    Used only when the C++ wrapper was not compiled in (FLAGGEMS_AVAILABLE
    not defined at build time). Iterates flag_gems._FULL_CONFIG and calls
    torch.library.Library.impl() for each op, which has higher per-call
    CPU overhead than the C++ TORCH_LIBRARY_IMPL path.
    """
    global _flaggems_lib, _registered_ops

    try:
        from flag_gems import _FULL_CONFIG
    except ImportError:
        return 0

    _flaggems_lib = torch.library.Library("aten", "IMPL")
    _registered_ops = []

    for item in _FULL_CONFIG:
        if len(item) < 2:
            continue

        op_name = item[0]
        impl_func = item[1]

        if op_name in _EXCLUDED_OPS:
            continue

        if len(item) > 2:
            condition = item[2]
            if callable(condition) and not condition():
                continue

        try:
            _flaggems_lib.impl(op_name, impl_func, "PrivateUse1")
            _registered_ops.append(op_name)
        except Exception:
            pass

    return len(_registered_ops)


def _register_flaggems_operators_cpp_supplement():
    """
    When C++ registration is active, register any remaining FlagGems ops
    from _FULL_CONFIG that are NOT already covered by the C++ wrapper.

    This catches ops that exist in flag_gems but were not explicitly listed
    in FlagGemsRegistration.cpp (e.g. newly added ops in a newer flag_gems).
    """
    global _flaggems_lib, _registered_ops

    try:
        from flag_gems import _FULL_CONFIG
    except ImportError:
        return 0

    _flaggems_lib = torch.library.Library("aten", "IMPL")
    _registered_ops = list(_CPP_FLAGGEMS_OPS)  # C++ ops count as registered

    skip = _EXCLUDED_OPS | _CPP_FLAGGEMS_OPS

    for item in _FULL_CONFIG:
        if len(item) < 2:
            continue

        op_name = item[0]
        impl_func = item[1]

        if op_name in skip:
            continue

        if len(item) > 2:
            condition = item[2]
            if callable(condition) and not condition():
                continue

        try:
            _flaggems_lib.impl(op_name, impl_func, "PrivateUse1")
            _registered_ops.append(op_name)
        except Exception:
            pass

    return len(_registered_ops)


def _register_composite_ops():
    """
    Register CompositeExplicitAutograd ops that cause cpu_fallback segfault.

    Some PyTorch ops are CompositeExplicitAutograd (not CompositeImplicitAutograd),
    meaning they don't auto-decompose for PrivateUse1. They fall through to
    cpu_fallback which segfaults when handling privateuseone tensors.

    We register these manually by implementing them in terms of ops that are
    already registered (like slice_scatter).
    """
    lib = torch.library.Library("aten", "IMPL")

    # slice_backward: used by autograd for tensor slicing (x[..., :n])
    # Implementation mirrors PyTorch's native slice_backward which calls slice_scatter
    def slice_backward_impl(grad_output, input_sizes, dim, start, end, step):
        input_sizes = [int(s) for s in input_sizes]
        dim = int(dim)
        start = int(start)
        end = int(end)
        step = int(step)
        # Clamp end to input_sizes[dim] (PyTorch passes large values like sys.maxsize)
        if end > input_sizes[dim]:
            end = input_sizes[dim]
        grad_input = torch.zeros(
            input_sizes, dtype=grad_output.dtype, device=grad_output.device
        )
        return torch.slice_scatter(grad_input, grad_output, dim, start, end, step)

    # lib.impl("slice_backward", slice_backward_impl, "PrivateUse1")

    # log_softmax: decompose into softmax + log to avoid FlagGems Triton kernel
    # that exceeds MACA's 4KB/thread private memory on large vocab dimensions.
    # The softmax kernel already has proper tiling for large N.
    def log_softmax_impl(self, dim, half_to_float=False):
        dtype = torch.float32 if half_to_float else self.dtype
        out = torch.softmax(self.to(torch.float32), dim=dim)
        out = torch.log(out)
        return out.to(dtype)

    def log_softmax_backward_impl(grad_output, output, dim, input_dtype):
        exp_output = torch.exp(output)
        grad_input = grad_output - exp_output * grad_output.sum(dim=dim, keepdim=True)
        return grad_input.to(input_dtype)

    lib.impl("_log_softmax", log_softmax_impl, "PrivateUse1")
    lib.impl("_log_softmax_backward_data", log_softmax_backward_impl, "PrivateUse1")

    return lib  # prevent GC


# Hold reference to prevent garbage collection of the library
_composite_ops_lib = None


def get_registered_ops() -> list:
    """Return list of registered FlagGems operators for flagos device."""
    return list(_registered_ops)


def is_flaggems_enabled() -> bool:
    """Check if FlagGems operators are registered for flagos device."""
    return len(_registered_ops) > 0


# Register FlagGems operators on import.
# - C++ path: operators were already registered at .so load time via
#   TORCH_LIBRARY_IMPL in FlagGemsRegistration.cpp. We only supplement
#   any ops not covered by the static list.
# - Python fallback: iterate _FULL_CONFIG and call lib.impl() for each op.
if _USE_CPP_REGISTRATION:
    _register_flaggems_operators_cpp_supplement()
else:
    _register_flaggems_operators_python()

_composite_ops_lib = _register_composite_ops()


# Re-export integration utilities
from torch_fl.integration import (  # noqa: E402
    is_flaggems_available,
    enable_flaggems_for_flagos,
    use_flaggems,
)

__all__ = [
    "flagos",
    "distributed",
    "get_registered_ops",
    "is_flaggems_enabled",
    "is_flaggems_available",
    "enable_flaggems_for_flagos",
    "use_flaggems",
]
