# Contributing to TiPToP

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

Follow the [installation guide](installation.md) in the documentation to set up your environment.

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting with a line length of 120. Ruff is installed automatically as part of the pre-commit hooks (via [`ruff-pre-commit`](https://github.com/astral-sh/ruff-pre-commit)) — no separate installation needed. To set up the hooks:

```bash
pixi run install-pre-commit
```

This will run the following on every commit (only on files in `tiptop/`):

- Import sorting (`ruff --select I --fix`)
- Code formatting (`ruff-format`)

## Making Changes

1. Fork the repository
2. Create a branch on your fork
3. Make your changes
4. Ensure pre-commit hooks pass
5. Open a pull request (see below)

## Integration Tests

Integration tests run the full TiPToP H5 pipeline (perception + planning) against 5 test scenes to check that core code paths are working. They require:

- The `tiptop.yml` config to be setup with the `fr3_robotiq` robot type and M2T2 server URL
- A `GOOGLE_API_KEY` environment variable (for Gemini)
- The M2T2 server running (see [simulation setup](simulation.md#setup))

Test assets (~17 MB) are automatically downloaded from Google Drive on the first run and cached locally.

```bash
pixi run test-integration
```

Since the tests exercise the real perception and planning stack, there is some inherent stochasticity. The test scenes are simple enough that they should reliably pass, but an occasional failure doesn't necessarily mean something is broken - re-running usually resolves it. If a test fails consistently, that's worth investigating.

The real validation is running TiPToP on the robot or through the full [simulation pipeline](simulation.md). The integration tests are not a substitute for that.

## Making a Pull Request

When opening a PR, please include:

- A clear description of what you changed and why
- What robot and/or simulation platform you tested on
- Steps to reproduce your results

## Reporting Issues

Please open a [GitHub issue](https://github.com/tiptop-robot/tiptop/issues) using one of the provided templates:

- **Bug Report** — for bugs or unexpected behavior (requires platform, robot/sim, and reproduction steps)
- **Feature Request** — for suggesting new features or improvements
