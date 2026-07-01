"""
div.Scalar dispatch tests

div.Scalar is a CompositeImplicitAutograd op — PyTorch decomposes it to
mul.Tensor (multiply by reciprocal) before reaching PrivateUse1 dispatch.
We only verify correctness.

Usage:
    pytest tests/integration/ops/test_div_scalar_dispatch.py -v
"""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


class AtenOpCollector(TorchDispatchMode):
    """Collect ATen ops with call depth info."""

    def __init__(self):
        self.ops = []  # list of (depth, op_name)
        self._depth = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append((self._depth, str(func)))
        self._depth += 1
        try:
            return func(*args, **(kwargs or {}))
        finally:
            self._depth -= 1

    def print_report(self):
        print("\n=== ATen Op Dispatch Trace ===")
        print(f"{'depth':<7} {'label':<12} {'op'}")
        print("-" * 70)
        for idx, (depth, op) in enumerate(self.ops):
            indent = "│ " * depth
            # leaf = no child (next entry isn't deeper)
            is_top = depth == 0
            next_depth = self.ops[idx + 1][0] if idx + 1 < len(self.ops) else 0
            is_leaf = next_depth <= depth
            if is_top and is_leaf:
                label = "[top+leaf]"
            elif is_top:
                label = "[top]"
            elif is_leaf:
                label = "[leaf]"
            else:
                label = "[mid]"
            print(f"{depth:<7} {label:<12} {indent}{op}")
        print("-" * 70)
        print(f"Total ops: {len(self.ops)}, max depth: {max(d for d, _ in self.ops)}")
        print("=== End Trace ===\n")


class TestDivScalarDispatchTrace:
    """Print dispatch trace to visualize top-level vs leaf ops."""

    @pytest.mark.anyplatform
    def test_div_scalar_trace(self):
        torch.manual_seed(0)
        a = torch.randn(4, 4, device=DEVICE)
        collector = AtenOpCollector()
        with collector:
            torch.div(a, 3.0)
        collector.print_report()


class TestDivScalarCorrectness:
    """torch.div(tensor, scalar) correctness on flagos device."""

    @pytest.mark.parametrize("shape", [(128, 256), (1,), (64, 64, 64)])
    @pytest.mark.anyplatform
    def test_div_scalar_shape(self, shape):
        torch.manual_seed(0)
        a = torch.randn(*shape, device=DEVICE)
        out = torch.div(a, 3.0)
        assert out.shape == shape
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_div_scalar_correctness(self):
        torch.manual_seed(1)
        a = torch.randn(32, 32, device=DEVICE)
        out = torch.div(a, 4.0)
        ref = a.cpu() / 4.0
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.cuda
    def test_div_scalar_matches_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(2)
        a_cuda = torch.randn(64, 64, device="cuda:0")
        ref = torch.div(a_cuda, 5.0)
        a = a_cuda.to(DEVICE)
        out = torch.div(a, 5.0)
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.anyplatform
    def test_div_scalar_dtype(self, dtype):
        torch.manual_seed(3)
        a = torch.randn(16, 16, device=DEVICE, dtype=dtype)
        out = torch.div(a, 2.0)
        ref = a.cpu().float() / 2.0
        torch.testing.assert_close(out.cpu().float(), ref, rtol=1e-2, atol=1e-2)
