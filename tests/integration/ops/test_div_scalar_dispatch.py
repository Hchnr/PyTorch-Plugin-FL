"""
div.Scalar dispatch tests

div.Scalar is a CompositeImplicitAutograd op — PyTorch decomposes it to
mul.Tensor (multiply by reciprocal) before reaching PrivateUse1 dispatch.
We only verify correctness.

Usage:
    pytest tests/integration/ops/test_div_scalar_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


def _run_subprocess(
    extra_env: dict, check: bool = True
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch; "
        "a = torch.randn(4,4,device='flagos:0'); "
        "torch.div(a, 3.0)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


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
        torch.testing.assert_close(
            out.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4
        )

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float16, torch.bfloat16]
    )
    @pytest.mark.anyplatform
    def test_div_scalar_dtype(self, dtype):
        torch.manual_seed(3)
        a = torch.randn(16, 16, device=DEVICE, dtype=dtype)
        out = torch.div(a, 2.0)
        ref = a.cpu().float() / 2.0
        torch.testing.assert_close(
            out.cpu().float(), ref, rtol=1e-2, atol=1e-2
        )


class TestDivScalarDispatch:
    """Verify dispatch routing for div.Scalar op."""

    @pytest.mark.flaggems_python
    def test_dispatch_log_flaggems_python(self):
        result = _run_subprocess(
            {
                "FLAGOS_LOG_DISPATCH": "1",
                "FLAGOS_OP_div__Scalar": "flaggems_python",
            },
            check=False,
        )
        assert (
            "[flagos dispatch] div.Scalar -> flagos_python"
            in result.stderr
        )

    @pytest.mark.cuda
    def test_dispatch_log_cuda_override(self):
        result = _run_subprocess(
            {
                "FLAGOS_LOG_DISPATCH": "1",
                "FLAGOS_OP_div__Scalar": "cuda",
            }
        )
        assert result.returncode == 0
        assert (
            "[flagos dispatch] div.Scalar -> cuda" in result.stderr
        )
