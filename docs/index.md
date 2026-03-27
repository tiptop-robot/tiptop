# TiPToP

🌐 [Project Website](https://tiptop-robot.github.io) · 📝 [Paper](https://arxiv.org/abs/2603.09971)

TiPToP is a Task and Motion Planning (TAMP) system that performs complex robot manipulation tasks like sorting, rearranging, and packing from images and natural language instructions. Using a modular architecture that separates perception, planning, and execution, TiPToP works out-of-the-box with zero training, zero demonstrations, and zero object-specific 3D models—yet matches or exceeds vision-language models trained on 350 hours of robot data.

The system combines learned perception models (depth prediction, VLMs, segmentation, grasp detection) with GPU-parallelized TAMP to reason explicitly about physical constraints and object interactions.


These docs make it easy to get TiPToP running on the [DROID hardware setup](https://droid-dataset.github.io/droid/docs/hardware-setup). With a few hours of additional effort, TiPToP can also be adapted to work with new embodiments zero-shot.

---

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} 📦 Installation
:link: installation
:link-type: doc

Install TiPToP and its modules, including perception models and Task and Motion Planners.
:::

:::{grid-item-card} 🚀 Getting Started
:link: getting-started
:link-type: doc

Configure your robot, calibrate cameras, and run your first TiPToP demo!
:::

:::{grid-item-card} 🖥️ Simulation
:link: simulation
:link-type: doc

Run TiPToP in DROID simulation based in IsaacLab.
:::

:::{grid-item-card} 📊 Evaluation Workflow
:link: evaluation
:link-type: doc

Set up scenes, run evaluations, and label results.
:::

:::{grid-item-card} 🤖 New Embodiment
:link: adding-new-embodiment
:link-type: doc

Add support for a new robot arm or camera to TiPToP.
:::

:::{grid-item-card} 📚 Command Reference
:link: command-reference
:link-type: doc

Detailed documentation for TiPToP CLI commands including helper commands.
:::

:::{grid-item-card} 🔧 Troubleshooting
:link: troubleshooting
:link-type: doc

Solutions to common issues with cameras, networking, motion planning, and perception.
:::

:::{grid-item-card} ⚠️ Limitations
:link: limitations
:link-type: doc

Limitations of the current TiPToP system.
:::

:::{grid-item-card} 🤝 Contributing
:link: contributing
:link-type: doc

How to contribute to TiPToP, including development setup and code style.
:::

::::

---

```{toctree}
:maxdepth: 2
:caption: Contents
:hidden:

Home <self>
installation
getting-started
simulation
evaluation
adding-new-embodiment
command-reference
troubleshooting
limitations
contributing
```

## License

TiPToP is released open-source under the MIT License.

However, key dependencies including [cuRobo](https://github.com/NVlabs/curobo), [cuTAMP](https://github.com/NVlabs/cuTAMP), [M2T2](https://github.com/NVlabs/M2T2), and [FoundationStereo](https://github.com/NVlabs/FoundationStereo) are licensed under the **NVIDIA Source Code License**, which permits non-commercial use for **research or evaluation purposes only**.