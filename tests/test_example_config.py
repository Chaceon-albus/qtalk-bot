from dataclasses import fields
from pathlib import Path

from qtalk_bot.config import Secrets, Settings, load_secrets, load_settings

ROOT = Path(__file__).parent.parent
EXAMPLE = ROOT / "config.example.toml"
ENV_EXAMPLE = ROOT / ".env.example"


def test_env_example_lists_exactly_the_known_secrets_without_values() -> None:
    # 新增或改名一个密钥时要同步更新 .env.example；示例里不能带值。
    pairs = [
        line.split("=", 1)
        for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    names = [name for name, _ in pairs]
    assert sorted(names) == sorted(field.name.upper() for field in fields(Secrets))
    assert [value for _, value in pairs] == [""] * len(pairs)
    # 这些变量名确实是程序读取的那些：全部填上后能读出来。
    assert load_secrets(dict.fromkeys(names, "x")) == Secrets(
        **{name.lower(): "x" for name in names}
    )


def test_example_config_is_valid_and_spells_out_the_defaults(tmp_path: Path) -> None:
    # 示例要跟着配置项一起更新：键写错或过时会在这里失败，示例里的值与默认值不一致也会。
    empty = tmp_path / "config.toml"
    empty.write_text("", encoding="utf-8")
    defaults = load_settings(empty)

    example = load_settings(EXAMPLE)

    assert isinstance(example, Settings)
    assert example.timezone == defaults.timezone
    assert example.data_dir == ROOT / "data"
    assert example.invites == defaults.invites
