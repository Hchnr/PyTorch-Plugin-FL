"""
mm dispatch tests

Verifies that torch.mm and torch.mm.out:
  - produce correct results on flagos device
  - C++ wrapper routes to flaggems (default) or cuda (via env override)
  - dispatch log confirms the actual backend used

Usage:
    pytest tests/integration/ops/test_mm_dispatch.py -v --device flagos
"""

import os
import subprocess
import sys

import pytest
import torch


@pytest.fixture(scope="session")
def device(request):
    dev = request.config.getoption("--device")
    if dev == "flagos":
        import torch_fl  # noqa: F401

        if not torch_fl.flagos.is_available():
            pytest.exit("flagos device is not available.")
    else:
        if not torch.cuda.is_available():
            pytest.exit("CUDA is not available.")
    return f"{dev}:0"


@pytest.fixture(scope="session")
def cuda_ref():
    """Reference mm results computed on CUDA."""
    if not torch.cuda.is_available():
        return None
    torch.manual_seed(42)
    a = torch.randn(128, 256, device="cuda:0", dtype=torch.float32)
    b = torch.randn(256, 64, device="cuda:0", dtype=torch.float32)
    return a, b, torch.mm(a, b)


def _run_mm_subprocess(extra_env: dict, use_out: bool = False) -> str:
    """Run a minimal mm call in a subprocess and return its stderr."""
    env = os.environ.copy()
    env.update(extra_env)
    if use_out:
        code = (
            "import torch_fl, torch; "
            "a = torch.randn(4,4,device='flagos:0'); "
            "b = torch.randn(4,4,device='flagos:0'); "
            "out = torch.empty(4,4,device='flagos:0'); "
            "torch.mm(a, b, out=out)"
        )
    else:
        code = (
            "import torch_fl, torch; "
            "a = torch.randn(4,4,device='flagos:0'); "
            "b = torch.randn(4,4,device='flagos:0'); "
            "torch.mm(a, b)"
        )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    return result.stderr


class TestMmDispatch:
    """torch.mm correctness and cross-device consistency."""

    @pytest.mark.parametrize("M,K,N", [(128, 256, 64), (1, 1, 1), (512, 512, 512)])
    def test_mm_shape(self, device, M, K, N):
        torch.manual_seed(0)
        a = torch.randn(M, K, device=device, dtype=torch.float32)
        b = torch.randn(K, N, device=device, dtype=torch.float32)
        out = torch.mm(a, b)
        assert out.shape == (M, N)
        assert out.device.type == device.split(":")[0]

    def test_mm_out(self, device):
        torch.manual_seed(1)
        a = torch.randn(64, 128, device=device, dtype=torch.float32)
        b = torch.randn(128, 32, device=device, dtype=torch.float32)
        out = torch.empty(64, 32, device=device, dtype=torch.float32)
        ret = torch.mm(a, b, out=out)
        assert ret.data_ptr() == out.data_ptr()
        assert out.shape == (64, 32)

    def test_mm_matches_cuda_ref(self, device, cuda_ref):
        """flagos mm result must match CUDA reference within tolerance."""
        if cuda_ref is None:
            pytest.skip("CUDA not available for reference")
        a_cuda, b_cuda, ref = cuda_ref
        dev_type = device.split(":")[0]
        a = a_cuda.to(device)
        b = b_cuda.to(device)
        out = torch.mm(a, b)
        torch.testing.assert_close(
            out.cpu(), ref.cpu(), rtol=1e-3, atol=1e-3,
            msg=f"mm result on {dev_type} differs from CUDA reference",
        )

    def test_mm_non_contiguous(self, device):
        torch.manual_seed(2)
        a = torch.randn(64, 128, device=device).t()  # non-contiguous
        b = torch.randn(64, 32, device=device)
        out = torch.mm(a, b)
        assert out.shape == (128, 32)

    def test_mm_half(self, device):
        torch.manual_seed(3)
        a = torch.randn(64, 128, device=device, dtype=torch.float16)
        b = torch.randn(128, 32, device=device, dtype=torch.float16)
        out = torch.mm(a, b)
        assert out.dtype == torch.float16
        assert out.shape == (64, 32)


@pytest.fixture(scope="session")
def require_flagos(request):
    if request.config.getoption("--device") != "flagos":
        pytest.skip("dispatch log tests only run on --device flagos")


@pytest.mark.usefixtures("require_flagos")
class TestMmDispatchLog:
    """Verify C++ wrapper routes to the correct backend via FLAGOS_LOG_DISPATCH."""

    def test_dispatch_log_flaggems_default(self):
        """Default config routes mm to flaggems."""
        stderr = _run_mm_subprocess({"FLAGOS_LOG_DISPATCH": "1"})
        assert "[flagos dispatch] mm -> flaggems" in stderr, (
            f"Expected flaggems dispatch log, got:\n{stderr}"
        )

    def test_dispatch_log_cuda_override(self):
        """FLAGOS_OP_mm=cuda overrides to cuda backend."""
        stderr = _run_mm_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_mm": "cuda"}
        )
        assert "[flagos dispatch] mm -> cuda" in stderr, (
            f"Expected cuda dispatch log, got:\n{stderr}"
        )

    def test_dispatch_log_mm_out_flaggems_default(self):
        """Default config routes mm.out to flaggems."""
        stderr = _run_mm_subprocess({"FLAGOS_LOG_DISPATCH": "1"}, use_out=True)
        assert "[flagos dispatch] mm.out -> flaggems" in stderr, (
            f"Expected flaggems dispatch log, got:\n{stderr}"
        )

    def test_dispatch_log_mm_out_cuda_override(self):
        """FLAGOS_OP_mm__out=cuda overrides mm.out to cuda backend."""
        stderr = _run_mm_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_mm__out": "cuda"}, use_out=True
        )
        assert "[flagos dispatch] mm.out -> cuda" in stderr, (
            f"Expected cuda dispatch log for mm.out, got:\n{stderr}"
        )
