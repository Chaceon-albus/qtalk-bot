import pytest


@pytest.fixture(autouse=True)
def block_real_network() -> None:
    """覆盖上层的同名夹具：实测本来就要访问真实网络。只在显式 `-m live` 时运行。"""
