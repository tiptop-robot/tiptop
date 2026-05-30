# Changelog

All notable changes to TiPToP are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

### Added

- `tiptop-rerun` CLI: re-runs the pipeline from a saved run directory, reusing the
  recorded observation. Task instruction and planning parameters default to the
  original run's values and can be overridden via flags (#16).
- Pick-only tasks: the pipeline now supports plans that just pick an object, including
  a prompt to safely catch the object on the real robot after a pick (#27).
- Rerun-disabled mode and a config option to run without applying M2T2 bounds (#27).
- `save_dir` is now included in the `tiptop-server` response (#27).
- Integration tests for the offline H5 pipeline (#14).
- `set_tiptop_cfg_from_file()` and `get_tiptop_cfg_path()` in `tiptop.config` for
  loading a config from an explicit path and querying the cached config's source path (#16).

### Changed

- **BREAKING:** `viz-tiptop-run` renamed its `--save-dir` flag to `--run-dir` (#16).
- `tiptop-h5` and the offline rerun logic are consolidated into `tiptop/tiptop_offline.py`
  (entry points `h5_entrypoint` and `rerun_entrypoint`); `tiptop/tiptop_h5.py` is removed.
  The `tiptop-h5` CLI and its flags are unchanged (#16).
- Offline runs save the merged config into the run directory so re-runs are reproducible (#16).
- Logging setup moved to the entrypoint level (#16).
- Updated the Gemini model used for perception (#26).
- Updated cuTAMP to 0.0.4 and added a configurable max number of motion-refinement attempts (#19, #20).
- `tiptop-server` serializes pipeline runs with an asyncio lock so concurrent requests
  no longer interleave (#23).
- Point-cloud erosion falls back to no erosion when it would leave too few points (#27).

### Removed

- **BREAKING:** `tiptop_cfg()` no longer accepts `force_reload` and no longer merges
  CLI overrides from `sys.argv` via `OmegaConf.from_cli`. Use `set_tiptop_cfg_from_file()`
  to load a specific config before the first `tiptop_cfg()` call (#16).
- `tiptop.config.tiptop_config_path` is no longer exported (the default path is internal) (#16).

### Fixed

- Fixed broken intrinsics imports in `calibrate_wrist_cam` (#21).

## [0.1.0]

Initial tagged release.
