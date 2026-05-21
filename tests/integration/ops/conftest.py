import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "anyplatform: runs on any platform")
    config.addinivalue_line("markers", "cuda: requires CUDA platform")
    config.addinivalue_line("markers", "ascend: requires Ascend platform")
