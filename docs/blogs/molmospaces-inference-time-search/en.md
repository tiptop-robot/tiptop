# Achieving SOTA on the MolmoSpaces benchmark with Inference-Time Search

Author: [Nishanth Kumar](https://nishanthjkumar.com/) · May 08, 2026

Original Blog Link: [here](https://x.com/nishanthkumar23/status/2052766074597265837)

We recently developed [TiPToP](https://tiptop-robot.github.io/): a general-purpose manipulation system that uses foundation models for perception followed by inference-time search for trajectory generation (specifically via Task and Motion Planning). Our system takes in natural language and images and outputs trajectories that attempt to accomplish the language instruction: a comparable I/O spec to end-to-end robot foundation models like Vision-Language-Action Models (VLAs) and World Action Models (WAMs).

Around the time of our release, several new benchmarks emerged that aim to enable comparison across approaches at scale. One of these is [MolmoSpaces](https://molmospaces.allen.ai/leaderboard) — a simulation-based benchmark with a maintained leaderboard covering most state-of-the-art VLAs, WAMs, and related manipulation approaches. We thought it would be interesting to see where TiPToP stands.

TiPToP achieves 46.1% on MolmoSpaces — outperforming every approach not trained on MolmoBot data and nearly doubling the next-best result (MolmoAct2-DROID). This is across 9 tasks with 1000 episodes each, run without modifying the underlying system (the only additional work was an integration layer for the MolmoSpaces API). Notably, TiPToP requires no robot data, was not tuned for this benchmark, generalizes across tasks and embodiments, and is the only inference-time search method on the leaderboard.

```{figure} ../../_static/molmospaces-leaderboard.png
:align: center
:alt: MolmoSpaces benchmark results

Fig. 1: Benchmark results on MolmoSpaces for all 9 tasks as of May 7, 2026 (not including MS-Open and MS-Close).
```

A few other highlights from the results (you can view the leaderboard and explore results [here](https://molmospaces.allen.ai/leaderboard)):

- **#1 on the "Not Using MolmoBot Data" leaderboard for “All Combined” tasks.** TiPToP scores 34.7% combined, ahead of MolmoAct2 (25.0%), π₀.₅-DROID (17.8%), LAP-VLA (10.1%). This is despite our method not running (and thus achieving 0% success) on the MS-Open and MS-Close task variants.
- **#1 on the Place-NextTo task, beating every policy on the leaderboard — including the ones trained on MolmoBot data.** TiPToP achieves 38.0%, ahead of π₀.₅-MolmoBot-FT (28.7%), MolmoBot-f3 (28.4%), and MolmoBot itself (26.4%). We suspect this is due to our approach’s ability to handle complex natural language expressions and spatial relationships.

## Failure analysis

Because TiPToP is a modular system, we can do something with the MolmoSpaces results that is hard to do for a learned policy: trace each failed episode to the specific module that caused it. We traced and labeled failures for each of our 9000 trials. The Sankey diagram below summarizes the result.

```{figure} ../../_static/molmospaces-failure-breakdown.png
:align: center
:alt: Failure breakdown Sankey diagram
```

**Key findings:**

- **There are a substantial number of execution failures.** 25.1% of all trials are episodes where TiPToP found a plan (i.e., the system believed it could solve the task) but open-loop execution failed. Execution failure tends to occur predominantly due to objects slipping out of the gripper during movement, or bad grasps that do not successfully pick the object.
- **The vast majority of planning failures are due to optimization and motion planning.** Optimization timeouts (31.0% of no-plan-found episodes) and cuRobo motion-planning failures (25.8%) together account for ~57% of planning failures. We suspect the bulk of these are due to (1) our approach approximating objects via a convex hull of their segmented pointclouds, and (2) representing objects via oriented bounding boxes during cuRobo motion planning instead of a more accurate collision representation.

We believe there are several clear ways to address these failures and improve the system’s performance. For instance, we discovered our controller in MolmoSpaces does not currently wait for the gripper to fully close before continuing its trajectory, which likely contributes significantly to objects slipping. Planning failures could be reduced by utilizing more accurate representations for objects during both optimization and motion planning, potentially via shape completion.

Importantly, each of these changes can be made to individual modules in our system without affecting the rest of the system. We believe some combination of these improvements will significantly improve TiPToP’s success rate on the overall benchmark.

## Takeaways

We present some general takeaways from these findings below as a series of responses to frequently asked questions (FAQs).

**1. Are you arguing that Task and Motion Planning (TAMP) is better than end-to-end learning?**

No — we feel that our findings point to something more nuanced than a direct ‘better’ or ‘worse’ verdict.

First, the immediate result: a TAMP system built on pretrained foundation models, with no robot training data, outperforms VLAs and WAMs trained on far more data when evaluated on this benchmark. We find this genuinely surprising, and we read it as evidence that modularity and inference-time search can offer powerful leverage in robotics, and deserve more attention and study.

Second, TAMP and end-to-end policies fail differently and succeed differently. TiPToP struggles with dexterity, clutter, and problems requiring skills outside its operator set; end-to-end methods struggle with long-horizon composition, semantic understanding, and precise spatial reasoning. Carefully comparing where each breaks down is, we think, one of the more productive ways to build a real understanding of robotic manipulation as a problem.

Third, these paradigms are not opposed. End-to-end learning offers reactivity and the ability to improve with data; search and symbolic structure offer long-horizon reasoning and the ability to resolve complex, compositional queries. Both seem genuinely useful, and how to bring them together is an open question we find much more interesting than which approach is better or worse than the other.

**2. Doesn't TiPToP fundamentally require tedious hand-engineering?**

Some manual engineering is required — i.e. defining symbolic predicates, operators, and samplers — but we view this as analogous to designing a neural network architecture, and there is [work showing that these structures can also be learned from demonstrations](https://pix2pred.csail.mit.edu/). In practice, TiPToP uses 11 predicates and 5 operators and they do not need to be changed across MolmoSpaces or any of the real-world tasks we have evaluated on. The overall system took 3 PhD students about 2.5 months to build with existing tools, and roughly 1 week of one master's student's time to port to MolmoSpaces. We believe the total effort is comparable to — and in many cases lower than — what is required to train a state-of-the-art VLA or WAM today.

**3. Does your system only apply to pick-and-place problems?**

No. We demonstrate in [our paper](https://arxiv.org/abs/2603.09971) that we can add new skills (namely whiteboard erasing), though additional manual effort is required to define new primitives and symbolic components (in the case of erasing, this took a few hours). We expect a week’s worth of work would enable our system to be extended to handle opening and closing articulated objects like drawers and doors.

**4. Isn't this kind of approach exactly what 'the Bitter Lesson' argues against?**

We don't think so. To quote the Bitter Lesson directly: “breakthrough progress eventually arrives by an opposing approach based on scaling computation by search and learning”. TAMP, and TiPToP in particular, is search-based and scales at inference time. The modular architecture also lets us swap in better foundation models as they appear, so the system inherits gains from large-scale learned models rather than competing with it.

**5. Many of the ideas from TiPToP do not seem new: why was it not possible to build a system like TiPToP much earlier than now?**

While [similar systems](https://arxiv.org/abs/2108.04145) have existed before, at least two things have changed recently. First, foundation models for perception, grounding, and affordance prediction are extremely capable. They can be connected to a planner and deliver meaningful real-world performance, and they continue to improve. Second, planning has gotten significantly faster. Recent improvements in hardware (GPUs and CPUs), and algorithms have enabled extremely fast search and motion planning.

**6. You highlight a number of assumptions and approximations (e.g., convex hull for objects) that your system makes in your paper. Are these limitations fundamental?**

We indeed make a number of explicit assumptions and approximations that we have tried to highlight. These make it clear what sets of problems the system will fail on (i.e., where our assumptions are violated), and we believe this is a strength since it makes the system more interpretable. We think it is interesting and noteworthy that our system performs well on the MolmoSpaces benchmark despite its many limitations and assumptions: not only does this reveal something about the applicability of these assumptions for generalized pick-and-place problems, but it also provides a concrete set of directions for system improvement (namely removing the assumptions). We do not believe these limitations are fundamental: there is existing research on ways to resolve almost every limitation we identify. We believe it is possible to improve our system’s performance very significantly with a few targeted improvements to address specific limitations.

## Acknowledgements

[@ryanlindeborg](https://x.com/ryanlindeborg) Lindeborg led the TiPToP integration with MolmoSpaces and gathered and analyzed the results. [@WillShenSaysHi](https://x.com/WillShenSaysHi) supported the integration, helped run the benchmark experiments, and helped analyze results. [@nishanthkumar23](https://x.com/nishanthkumar23) helped with the integration and helped analyze and present results. [@omarrayyann](https://x.com/omarrayyann) , Maximilian Argus, [@wpumacay7567](https://x.com/wpumacay7567) and [@notmahi](https://x.com/notmahi) provided encouragement and invaluable debugging support enabling TiPToP to be integrated with MolmoSpaces and added our results to the public leaderboard.
