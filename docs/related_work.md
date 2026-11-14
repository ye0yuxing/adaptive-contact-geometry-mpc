# Related work and scope of the project

Survey date: **2026-11-14**. This is a focused literature and public-repository
search, not a systematic review or a guarantee of novelty.

## Conclusion

**The broad idea of differentiable contact geometry feeding a CF model is
already present in the literature.** The original CF paper explicitly
discusses the frozen-geometry limitation and points toward end-to-end smooth
geometry and dynamics. A later paper implements a differentiable collision
pipeline with a full-dynamic CF response and demonstrates downstream MPC.

The search did not locate an exact public counterpart of this repository's
combination of **the two-box keyed-peg task, support-plane/sample-based C-SDF,
quasi-dynamic explicit CF response, and a paired frozen/adaptive horizon
ablation**. That observation supports a modest implementation/evaluation
project; it does not establish a new research contribution or priority.

## Closest sources

| Work | Verified overlap | Distinction relevant to this project |
|---|---|---|
| Jin, [Complementarity-Free Multi-Contact Modeling and Optimization for Dexterous Manipulation](https://arxiv.org/html/2408.07855v3), v3, 2025 | Section V freezes measured-state geometry inside MPC. Section VII-C identifies this limitation for irregular peg-in-hole tasks and discusses differentiable geometry/dynamics, citing ContactSDF. | The motivating limitation and general remedy are already explicit. Our contribution is a task-specific implementation and matched ablation. |
| Yang and Jin, [ContactSDF](https://arxiv.org/html/2408.09612v2), v2, 2025 | C-SDF provides differentiable collision geometry; D-SDF supplies differentiable contact dynamics within MPC. | We use the geometry construction as inspiration, but the response is Jin's dual-cone CF approximation, not ContactSDF's D-SDF response. Adaptive geometry within differentiable MPC itself is established. |
| Wang et al., [ContactGaussian-WM](https://arxiv.org/html/2602.11021v1), 2026 | Sections III-D.1–2 combine differentiable Gaussian/sphere-based collision geometry with full-dynamic CF contact. Section IV-C demonstrates downstream MPC for a LEAP hand in simulation. | This is a direct precedent for the broad A+B composition. Its representation, full dynamics, video-based world-model learning, and evaluation differ from this small quasi-dynamic peg experiment. |
| Borse et al., [ComFree-Sim](https://arxiv.org/html/2603.12185v2), 2026 | GPU-parallel analytical CF contact simulation and sampling-based MPC, including hand control. | It is adjacent CF simulation/control work. Its simulator/MPPI setting is different from the symbolic C-SDF horizon ablation here. |
| Dietz et al., [Robust Rigid Body Assembly via Contact-Implicit Optimal Control with Exact Second-Order Derivatives](https://arxiv.org/html/2601.22849v1), 2026 | Differentiable collision information and contact-implicit optimization for assembly, including different peg shapes. | The method uses optimization-based collision/contact formulations, smoothing and second-order derivatives, rather than this explicit CF response. It is a relevant task-level comparator. |
| Xie et al., [Where to Touch, How to Contact](https://arxiv.org/html/2601.10930v1), 2026 | A hierarchical RL-MPC approach uses local CF control for geometry-aware manipulation. | It addresses geometry through a higher-level policy; it is not evidence of the same C-SDF replacement and paired ablation. |
| Liu et al., [Geometry-Informed Contact Optimization and Model Predictive Control for Efficient Contact-Rich Manipulation](https://doi.org/10.1109/LRA.2026.3711851), RA-L, 2026 | The publisher abstract describes geometry-informed friction metrics and smooth complementarity-free dynamics for MPC. | **Abstract/metadata only:** full text could not be retrieved in this survey. Exact overlap remains unresolved and must not be ruled out. |

The ContactGaussian-WM project page was inspected, but its implementation was
not audited. A paper-level comparison is not a claim that its released code
implements any particular frozen/adaptive switch.

## Search coverage and limits

Queries included:

```text
"complementarity-free" "signed distance" MPC
"ContactSDF" "geometry" "complementarity"
"complementarity-free" "adaptive" geometry
"Complementarity-Free" "C-SDF"
"ContactSDF" "peg-in-hole"
"C-SDF" "CF-MPC"
"contact geometry" "CF-MPC"
"ContactSDF" "CF" MPC
"Geometry-Informed Contact Optimization and Model Predictive Control"
```

Primary papers, author project pages, the public CF and ContactSDF repository
roots/READMEs, and GitHub repository/code search results were checked. GitHub
search included `ContactSDF`, `"complementarity-free" geometry`, and
`ContactSDF ExplicitModel`. Searches are limited by indexing, terminology,
access, and publication timing; they did not audit every fork or all source
files. Unpublished work cannot be excluded. Negative search results are not
proof of absence.

## Project scope

> An implementation study of predicted-state contact geometry in explicit
> complementarity-free MPC, with a controlled peg-in-hole ablation.

This repository credits the geometry and response separately and evaluates
their composition through its own frozen/adaptive switch. Both arms use the
same smooth geometry representation and explicit CF response. Neither arm
represents a reproduction of an upstream project's complete experimental suite.
