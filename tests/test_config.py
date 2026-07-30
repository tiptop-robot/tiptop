"""Unit tests for TiPToP config loading. Run with: pixi run test-unit"""

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from tiptop import config as tiptop_config
from tiptop.config import default_cfg_path, set_tiptop_cfg_from_file, tiptop_cfg


@pytest.fixture(autouse=True)
def restore_cached_cfg():
    """set_tiptop_cfg_from_file caches into module globals, so keep tests from leaking into each other."""
    cached_cfg, cached_path = tiptop_config._cached_cfg, tiptop_config._cached_cfg_path
    yield
    tiptop_config._cached_cfg, tiptop_config._cached_cfg_path = cached_cfg, cached_path


@pytest.fixture
def packaged_cfg() -> DictConfig:
    return OmegaConf.load(default_cfg_path)


def write_cfg(path: Path, cfg: DictConfig) -> Path:
    path.write_text(OmegaConf.to_yaml(cfg))
    return path


def copy_cfg(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))


def test_packaged_config_loads(packaged_cfg):
    """The shipped config has every key by construction, so it must never raise."""
    assert tiptop_cfg().robot.type == packaged_cfg.robot.type


def test_complete_config_with_overrides_does_not_raise(tmp_path, packaged_cfg):
    """Only *missing* keys are an error; differing values are the normal case."""
    overridden = copy_cfg(packaged_cfg)
    overridden.robot.time_dilation_factor = 0.123
    cfg = set_tiptop_cfg_from_file(write_cfg(tmp_path / "override.yml", overridden))
    assert cfg.robot.time_dilation_factor == 0.123


@pytest.mark.parametrize("section, key", [("perception", "depth_trunc_m"), ("robot", "time_dilation_factor")])
def test_missing_key_raises_on_normal_run(tmp_path, packaged_cfg, section, key):
    stale = copy_cfg(packaged_cfg)
    del stale[section][key]
    with pytest.raises(ValueError, match="missing keys"):
        set_tiptop_cfg_from_file(write_cfg(tmp_path / "stale.yml", stale))


def test_missing_section_raises_on_normal_run(tmp_path, packaged_cfg):
    stale = copy_cfg(packaged_cfg)
    del stale["cameras"]
    with pytest.raises(ValueError, match="missing keys"):
        set_tiptop_cfg_from_file(write_cfg(tmp_path / "stale.yml", stale))


def test_extra_keys_are_kept_and_do_not_raise(tmp_path, packaged_cfg):
    """A recording may carry keys since dropped from the defaults; only *missing* keys are an error."""
    extended = copy_cfg(packaged_cfg)
    extended.robot.retired_option = 42
    cfg = set_tiptop_cfg_from_file(write_cfg(tmp_path / "extended.yml", extended))
    assert cfg.robot.retired_option == 42


def test_fill_missing_recovers_defaults_and_warns(tmp_path, packaged_cfg, caplog):
    """Recorded configs replayed by tiptop-offline predate later keys and cannot be updated after the fact."""
    stale = copy_cfg(packaged_cfg)
    del stale["cameras"]
    del stale.perception.m2t2["apply_bounds"]

    cfg = set_tiptop_cfg_from_file(write_cfg(tmp_path / "old_run.yml", stale), fill_missing=True)

    assert cfg.cameras == packaged_cfg.cameras
    assert cfg.perception.m2t2.apply_bounds == packaged_cfg.perception.m2t2.apply_bounds
    assert any("missing keys" in record.message for record in caplog.records)


def test_fill_missing_keeps_user_values(tmp_path, packaged_cfg):
    """Defaults fill gaps only; anything the recorded config set must survive the merge."""
    stale = copy_cfg(packaged_cfg)
    del stale["cameras"]
    stale.robot.time_dilation_factor = 0.456
    stale.perception.m2t2.url = "http://recorded-host:8123"

    cfg = set_tiptop_cfg_from_file(write_cfg(tmp_path / "old_run.yml", stale), fill_missing=True)

    assert cfg.robot.time_dilation_factor == 0.456
    assert cfg.perception.m2t2.url == "http://recorded-host:8123"


def test_cached_path_is_the_source_file_not_the_defaults(tmp_path, packaged_cfg):
    """recording.py snapshots get_tiptop_cfg_path(), which must stay the config that was actually loaded."""
    stale = copy_cfg(packaged_cfg)
    del stale["cameras"]
    cfg_path = write_cfg(tmp_path / "old_run.yml", stale)

    set_tiptop_cfg_from_file(cfg_path, fill_missing=True)

    assert tiptop_config.get_tiptop_cfg_path() == cfg_path
