# TiPToP 🎩🤖

[![arXiv](https://img.shields.io/badge/arXiv-2603.09971-b31b1b.svg)](https://arxiv.org/abs/2603.09971)
[![Documentation](https://readthedocs.org/projects/TiPToP-robot/badge/?version=latest)](https://TiPToP-robot.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A Modular Open-Vocabulary Planning System for Robotic Manipulation**

[Website](https://tiptop-robot.github.io/) | [Documentation](https://tiptop-robot.readthedocs.io/) | [Paper](https://arxiv.org/abs/2603.09971)

<p align="center">
  <img src="docs/_static/logo-light.png" alt="TiPToP Logo" width="200">
</p>

TiPToP solves complex real-world manipulation tasks directly from raw pixels and natural-language commands by combining Task-and-Motion Planning with perception and language models through inference-time search — with zero robot training data.

## Getting Started

See the [documentation](https://tiptop-robot.readthedocs.io/) for installation, setup, and usage instructions.

## Building the Docs

The documentation is hosted at [tiptop-robot.readthedocs.io](https://tiptop-robot.readthedocs.io/) and automatically rebuilds on every push to `main`. To build and serve it locally for previewing changes:

```bash
pixi run docs-install   # Install doc dependencies
pixi run docs-build     # Build HTML docs
pixi run docs-serve     # Serve with live reload
```

## Contributing

See the [Contributing Guide](https://tiptop-robot.readthedocs.io/en/latest/contributing.html) for development setup and guidelines.

## Citation

```bibtex
@article{shen2026tiptop,
    title={{TiPToP}: A Modular Open-Vocabulary Planning System for Robotic Manipulation},
    author={Shen, William and Kumar, Nishanth and Chintalapudi, Sahit and Wang, Jie and Watson, Christopher and Hu, Edward S. and Cao, Jing and Jayaraman, Dinesh and Kaelbling, Leslie Pack and Lozano-P\'{e}rez, Tom\'{a}s},
    journal={arXiv preprint arXiv:2603.09971},
    year={2026}
}
```