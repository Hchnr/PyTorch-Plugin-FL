"""
bmm dispatch tests

Verifies that torch.bmm and torch.bmm.out:
  - produce correct results on flagos device
  - C++ wrapper routes to flaggems (default) or cuda (via env override)
  - dispatch log confirms the actual backend used

Usage:
    pytest tests/integration/ops/test_bmm_dispatch.py -v --device flagos
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
    """Reference bmm results computed on CUDA."""
    if not torch.cuda.is_available():
        return None
    torch.manual_seed(42)
    a = torch.randn(8, 128, 256, device="cuda:0", dtype=torch.float32)
    b = torch.randn(8, 256, 64, device="cuda:0", dtype=torch.float32)
    return a, b, torch.bmm(a, b)


def _run_bmm_subprocess(extra_env: dict, use_out: bool = False) -> str:
    """Run a minimal bmm call in a subprocess and return its stderr."""
    env = os.environ.copy()
    env.update(extra_env)
    if use_out:
        code = (
            "import torch_fl, torch; "
            "a = torch.randn(2,4,4,device='flagos:0'); "
            "b = torch.randn(2,4,4,device='flagos:0'); "
            "out = torch.empty(2,4,4,device='flagos:0'); "
            "torch.bmm(a, b, out=out)"
        )
    else:
        code = (
            "import torch_fl, torch; "
            "a = torch.randn(2,4,4,device='flagos:0'); "
            "b = torch.randn(2,4,4,device='flagos:0'); "
            "torch.bmm(a, b)"
        )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    return result.stderr


class TestBmmDispatch:
    """torch.bmm correctness and cross-device consistency."""

    @pytest.mark.parametrize("B,M,K,N", [(4, 128, 256, 64), (1, 1, 1, 1), (8, 512, 512, 512)])
    def test_bmm_shape(self, device, B, M, K, N):
        torch.manual_seed(0)
        a = torch.randn(B, M, K, device=device, dtype=torch.float32)
        b = torch.randn(B, K, N, device=device, dtype=torch.float32)
        out = torch.bmm(a, b)
        assert out.shape == (B, M, N)
        assert out.device.type == device.split(":")[0]

    def test_bmm_out(self, device):
        torch.manual_seed(1)
        a = torch.randn(4, 64, 128, device=device, dtype=torch.float32)
        b = torch.randn(4, 128, 32, device=device, dtype=torch.float32)
        out = torch.empty(4, 64, 32, device=device, dtype=torch.float32)
        ret = torch.bmm(a, b, out=out)
        assert ret.data_ptr() == out.data_ptr()
        assert out.shape == (4, 64, 32)

    def test_bmm_matches_cuda_ref(self, device, cuda_ref):
        """flagos bmm result must match CUDA reference within tolerance."""
        if cuda_ref is None:
            pytest.skip("CUDA not available for reference")
        a_cuda, b_cuda, ref = cuda_ref
        a = a_cuda.to(device)
        b = b_cuda.to(device)
        out = torch.bmm(a, b)
        torch.testing.assert_close(
            out.cpu(), ref.cpu(), rtol=1e-3, atol=1e-3,
            msg=f"bmm result on {device} differs from CUDA reference",
        )

    def test_bmm_half(self, device):
        torch.manual_seed(3)
        a = torch.randn(4, 64, 128, device=device, dtype=torch.float16)
        b = torch.randn(4, 128, 32, device=device, dtype=torch.float16)
        out = torch.bmm(a, b)
        assert out.dtype == torch.float16
        assert out.shape == (4, 64, 32)


@pytest.fixture(scope="session")
def require_flagos(request):
    if request.config.getoption("--device") != "flagos":
        pytest.skip("dispatch log tests only run on --device flagos")


@pytest.mark.usefixtures("require_flagos")
class TestBmmDispatchLog:
    """Verify C++ wrapper routes to the correct backend via FLAGOS_LOG_DISPATCH."""

    def test_dispatch_log_flaggems_default(self):
        """Default config routes bmm to flaggems."""
        stderr = _run_bmm_subprocess({"FLAGOS_LOG_DISPATCH": "1"})
        assert "[flagos dispatch] bmm -> flaggems" in stderr, (
            f"Expected flaggems dispatch log, got:\n{stderr}"
        )

    def test_dispatch_log_cuda_override(self):
        """FLAGOS_OP_bmm=cuda overrides to cuda backend."""
        stderr = _run_bmm_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_bmm": "cuda"}
        )
        assert "[flagos dispatch] bmm -> cuda" in stderr, (
            f"Expected cuda dispatch log, got:\n{stderr}"
        )

    def test_dispatch_log_bmm_out_flaggems_default(self):
        """Default config routes bmm.out to flaggems."""
        stderr = _run_bmm_subprocess({"FLAGOS_LOG_DISPATCH": "1"}, use_out=True)
        assert "[flagos dispatch] bmm.out -> flaggems" in stderr, (
            f"Expected flaggems dispatch log, got:\n{stderr}"
        )

    def test_dispatch_log_bmm_out_cuda_override(self):
        """FLAGOS_OP_bmm__out=cuda overrides bmm.out to cuda backend."""
        stderr = _run_bmm_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_bmm__out": "cuda"}, use_out=True
        )
        assert "[flagos dispatch] bmm.out -> cuda" in stderr, (
            f"Expected cuda dispatch log for bmm.out, got:\n{stderr}"
        )
