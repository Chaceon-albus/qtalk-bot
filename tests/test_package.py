from importlib.metadata import version

import pyqbot

import qtalk_bot


def test_package_is_installed_with_pyqbot() -> None:
    # 骨架阶段的冒烟测试：包装进了环境、pyqbot 能用。有了真正的测试后可以删掉。
    assert version("qtalk-bot")
    assert qtalk_bot.__name__ == "qtalk_bot"
    assert callable(pyqbot.Bot)
