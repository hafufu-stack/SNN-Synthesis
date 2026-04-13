# SNN-Synthesis: Quantization Noise as Stochastic Resonance, Multi-Model Beam Ensembles, and ARC-AGI-3 Field Validation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19343952.svg)](https://doi.org/10.5281/zenodo.19343952)

> **Small + Noise > Large + Greedy — Stochastic resonance is architecture-invariant, model-invariant, scale-invariant, quantization-robust, and can compensate for lost parameters**

Successor to [SNN-Genesis](https://github.com/hafufu-stack/snn-genesis) (v1–v20, 111 phases, 127 pages).
SNN-Genesis dissected the black box of LLM reasoning through noise intervention. SNN-Synthesis uses that anatomical map to **build new AI architectures** and proves that stochastic resonance is a **universal, architecture-invariant, model-invariant neural network phenomenon**—then harnesses it for **autonomous self-evolution without any human supervision**.

## 🔬 Research Vision

SNN-Genesis was the **Anatomy & Physiology** phase — discovering the physical laws of reasoning (stochastic resonance, Aha! dimensions, layer localization).

SNN-Synthesis is the **Architecture & Synthesis** phase — building systems that internalize those laws, proving their **universality across architectures (CNN → Transformer), model families (Mistral → Qwen), scales (63K → 7B), precisions (FP16 → 4-bit), and tasks (grid navigation → symbolic reasoning → math → factual QA → ARC-AGI-3 competition)**, and demonstrating that noise + natural selection form a **complete learning paradigm**.

### 🏆 Key Results (v8)

**New in v8 (Phases 61–63 + ARC-AGI-3 Kaggle) — Three New Findings:**

1. **🔥 Quantization Noise IS Stochastic Resonance.**
   4-bit Qwen-1.5B achieves **58% at K=1 without beam search**, surpassing both FP16 (32%) and Mistral-7B baseline (42%). Quantization noise acts as free stochastic resonance. At K=51, all precisions converge to **84%**. A "double noise" non-monotonicity reveals destructive interference. The space-time duality extends to a **space-time-precision triad**. (Phase 61)

   ![Phase 61: Extreme SR-Quantization](figures/phase61_extreme_quantization.png)

2. **🧬 Multi-Model Beam Ensemble: Architectural Diversity is Orthogonal to Noise Diversity.**
   Mixing beams from Mistral-7B (×6) + Qwen-1.5B (×5) achieves **86.7%**, surpassing all single-model ensembles (Mistral ×11: 70%, Qwen ×11: 80%) by +6.7–16.7pp. 5 problems solvable only by one architecture. (Phase 63)

   ![Phase 63: Multi-Model Ensemble](figures/phase63_multi_model_ensemble.png)

3. **🏟️ ARC-AGI-3 Kaggle Field Validation: Thermodynamic Coarse-Graining.**
   Five agents submitted to live competition. The simplest (v5: macro-stats UCB, **score 0.13**) beats all "intelligent" agents (v14 CfC: 0.10, v12 LLM: 0.07, v13 SimHash: 0.02). Root cause: three "death traps" — timeout starvation, pixel-noise sensitivity, action-space explosion. The winning strategy uses **thermodynamic coarse-graining**: macroscopic statistics invariant to microscopic noise.

**v7 Landmark Results (Phases 39–60):**
4. **Stochastic Resonance Quantization**: Qwen-1.5B + NBS (80%) > Mistral-7B baseline (42%) — **space-time duality**. (Phase 59)
5. **The Crossover Law** (Bitter Lesson): Overhead >0.5ms → intelligence loses to random exploration. (Phases 44–46)
6. **Test-Time Compute Scaling Law**: Logarithmic accuracy scaling with K, with **non-monotonic saturation** at K=51. (Phases 60, 62)
7. **SimHash O(1) curiosity** matches RND at ~100× less overhead. (Phase 51)
8. **6 null results** (Phases 53–58) confirm design convergence.

   ![Phase 46: The Crossover Law](figures/phase46_crossover_law.png)

**Established in v1–v6 (Phases 1–38):**
9. **LLM-ExIt achieves Oracle-free self-evolution.** 16% → 100% in 3 iterations. (Phase 32b)
10. **NBS generalizes to math reasoning.** GSM8K 53% → **89.5%** at K=11. (Phase 31b)
11. **NBS is architecture-invariant.** K=11: 78% on 63K CNN, 100% on 7B LLM. (Phase 29)
12. **SNN-ExIt:** Zero knowledge → **99%** on LS20, surpassing Oracle CNN by 21pp. (Phase 20)
13. **Knowledge Multiplexing via Discrete ID Gating.** (Phase 35c)
14. **σ-Diverse NBS eliminates hyperparameter tuning.** (Phase 37a)
15. **NBS is model-invariant.** Qwen2.5-7B matches Mistral-7B. (Phase 38)
16. **21 principal insights, 20 honest null results** across 63 experimental phases.

## 📁 Project Structure

```
snn-synthesis/
├── experiments/          # LLM experiment scripts (Phases 1-7, 3b, 6b, 29-63)
│   ├── phase29_llm_noisy_beam.py        # LLM NBS (v4)
│   ├── phase32b_llm_exit.py             # LLM-ExIt (v5)
│   ├── phase39_curiosity_rnd.py         # RND curiosity (v7)
│   ├── phase44_complexity_budget.py     # Crossover Law (v7)
│   ├── phase51_simhash_curiosity.py     # SimHash O(1) (v7)
│   ├── phase59_sr_quantization.py       # SR-Quantization (v7)
│   ├── phase61_extreme_quantization.py  # Extreme SR-Quant (v8)
│   ├── phase62_spacetime_surface.py     # Space-Time Surface (v8)
│   ├── phase63_multi_model_ensemble.py  # Multi-Model NBS (v8)
│   └── ...
├── arc-agi/              # ARC-AGI-3 experiments + Kaggle agents
│   ├── kaggle_cell2_agent_v15.py   # v15 Thermodynamic Explorer
│   ├── kaggle_cell2_agent_v14.py   # v14 CfC + Spatial Features
│   ├── kaggle_cell2_agent_v13.py   # v13 SimHash+σ-diverse NBS
│   ├── kaggle_cell2_agent_llm.py   # v12 LLM+NBS agent
│   └── ...
├── results/              # Experiment result logs (JSON)
├── figures/              # All experiment figures (PNG)
├── papers/               # LaTeX source (v1–v8, .gitignore'd)
├── LICENSE
└── README.md
```

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/hafufu-stack/snn-synthesis.git
cd snn-synthesis

# Install dependencies (LLM experiments)
pip install torch transformers bitsandbytes peft snntorch matplotlib numpy

# Install dependencies (ARC-AGI-3 experiments)
pip install arcprize
```

## 📄 Papers

- **SNN-Synthesis v8** (latest): [Zenodo (PDF)](https://doi.org/10.5281/zenodo.19343952)
  - **63 experiments** (Phases 1–63), **41 contributions**, **21 principal insights**
  - **Quantization Noise as SR**: 4-bit Qwen-1.5B (58% K=1) > 7B baseline (42%) (Phase 61)
  - **Multi-Model Ensemble**: Mistral+Qwen mix = 86.7% > single-model best 80% (Phase 63)
  - **Kaggle Field Validation**: v5 (0.13) beats v13 (0.02) — thermodynamic coarse-graining
  - **20 honest null results** confirming design convergence
  - v1–v7 findings retained

- **SNN-Synthesis v7**: [Zenodo (PDF)](https://doi.org/10.5281/zenodo.19545095)
  - 60 experiments — SR-Quantization, Crossover Law, TTC Scaling Law

- **SNN-Synthesis v6**: [Zenodo (PDF)](https://doi.org/10.5281/zenodo.19502579)
  - 38 experiments — Knowledge Multiplexing, σ-Diverse NBS, Multi-Model Universality

- **SNN-Synthesis v5**: [Zenodo (PDF)](https://doi.org/10.5281/zenodo.19481773)
  - 33 experiments — LLM-ExIt (16% → 100%), GSM8K NBS (89.5%)

- **SNN-Synthesis v4**: [Zenodo (PDF)](https://doi.org/10.5281/zenodo.19430135)
  - 30 experiments — LLM NBS achieves 100% at K=11

- **SNN-Synthesis v3**: [Zenodo (PDF)](https://doi.org/10.5281/zenodo.19422317)
  - Noisy Beam Search (78% L2), SNN-ExIt (99% LS20)

- **SNN-Synthesis v2**: [Zenodo (PDF)](https://doi.org/10.5281/zenodo.19373028)
- **SNN-Synthesis v1**: [Zenodo (PDF)](https://doi.org/10.5281/zenodo.19343953)

## 📖 Predecessor

- **SNN-Genesis** (v1–v20): [GitHub](https://github.com/hafufu-stack/snn-genesis) | [Zenodo](https://doi.org/10.5281/zenodo.14637029)
  - 111 experiments across 20 versions
  - Key discoveries: Stochastic resonance in LLMs, Aha! steering vectors, layer-specific Prior Override (L16=76.7%), Trajectory Distillation (48%), SNN adaptive control

## 🤖 AI Collaboration

This research is conducted collaboratively between the human author and AI research assistants (Anthropic Claude Opus 4.6 via Google Antigravity). AI contributes to code development, debugging, experimental design, and analysis. All research direction and final interpretation are by the human author.

## 📄 Citation

```bibtex
@misc{funasaki2026snnsynthesis,
  author = {Funasaki, Hiroto},
  title = {SNN-Synthesis v8: Quantization Noise as Stochastic Resonance, Multi-Model Beam Ensembles, and ARC-AGI-3 Field Validation from 63K to 7B Parameters},
  year = {2026},
  doi = {10.5281/zenodo.19343952},
  publisher = {Zenodo},
  url = {https://doi.org/10.5281/zenodo.19343952}
}
```

## 📜 License

MIT License
