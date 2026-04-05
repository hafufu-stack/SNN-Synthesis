# SNN-Synthesis: Architecture-Invariant Stochastic Resonance from 63K CNNs to 7B LLMs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19343952.svg)](https://doi.org/10.5281/zenodo.19343952)

> **100% Solve Rate on Mistral-7B via Noisy Beam Search — Architecture-Invariant across Five Orders of Magnitude**

Successor to [SNN-Genesis](https://github.com/hafufu-stack/snn-genesis) (v1–v20, 111 phases, 127 pages).
SNN-Genesis dissected the black box of LLM reasoning through noise intervention. SNN-Synthesis uses that anatomical map to **build new AI architectures** and proves that stochastic resonance is a **universal, architecture-invariant neural network phenomenon**—then harnesses it for **autonomous self-evolution without any human supervision**.

## 🔬 Research Vision

SNN-Genesis was the **Anatomy & Physiology** phase — discovering the physical laws of reasoning (stochastic resonance, Aha! dimensions, layer localization).

SNN-Synthesis is the **Architecture & Synthesis** phase — building systems that internalize those laws, proving their **universality across architectures (CNN → Transformer) and scales (63K → 7B)**, and demonstrating that noise + natural selection form a **complete learning paradigm**.

### 🏆 Key Results (v4)

1. **Noisy Beam Search is architecture-invariant.** K=11 achieves **78% on 63K CNN** (ARC-AGI) and **100% on Mistral-7B** (Modified Hanoi, p<10⁻¹⁰). The effect is *amplified* at larger scale (+76pp on LLM vs. +66pp on CNN).
2. **Stochastic resonance is scale-invariant.** σ*≈0.05–0.20 across 63K CNNs → 7B Transformers (p<10⁻⁵⁰ at N=1000).
3. **Static noise and fixed K are optimal.** 5 dynamic noise strategies + 3 K schedules all fail to outperform constant parameters.
4. **SNN-ExIt (Expert Iteration):** Oracle-free self-evolution from **zero human knowledge** → **99% clear rate** on LS20 (surpassing Oracle-trained CNN by 21pp).
5. **Two-Condition Theory (validated on all 7 games):** ExIt succeeds iff (a) bootstrap miracle rate > 0 ("activation energy") AND (b) the game's state→action mapping is *generalizably* learnable. 4/7 games fail Condition 1; TR87 fails Condition 2.
6. **Learnability has a threshold effect.** Frame stacking improves TR87's train accuracy from 25%→34.5% but ExIt still fails—generalization, not memorization, determines success.
7. **ExIt is self-healing:** Removing 75% of seed miracles paradoxically improves final performance (57% vs 44%).

## 📁 Project Structure

```
snn-synthesis/
├── experiments/          # LLM experiment scripts (Phases 1-7, 3b, 6b, 29)
│   ├── phase3b_mistral_ppo.py   # Mistral-7B PPO noise optimization
│   └── phase29_llm_noisy_beam.py  # LLM Noisy Beam Search (v4)
├── arc-agi/              # ARC-AGI-3 experiments (Phases 8-28, 30)
│   ├── arc_micro_brain.py       # CNN noise sweep pipeline
│   ├── arc_oracle_dataset.py    # Oracle trajectory extraction
│   ├── agent_ls20_v14.py        # BFS Oracle solver (latest)
│   ├── phase8_extended.py       # Extended resonance analysis
│   ├── phase9_dynamic_sigma.py  # Temporal scheduling
│   ├── phase10_confidence_adaptive.py  # Confidence gating
│   ├── phase*_runner.py         # Multi-phase runners
│   ├── phase20_exit_ls20.py     # SNN-ExIt on LS20
│   ├── phase24_activation_energy.py  # Activation energy curve
│   ├── phase27_frame_stacking_exit.py  # Frame stacking (v4)
│   ├── phase28_intermediate_exit.py    # Intermediate games (v4)
│   ├── phase30_dynamic_k_exit.py       # Dynamic K scheduling (v4)
│   ├── data/                    # Training data & model weights
│   └── results/                 # All results (JSON + PNG)
├── results/              # LLM experiment result logs (JSON)
├── figures/              # All experiment figures (PNG)
├── papers/               # Paper sources (shared via Zenodo)
├── LICENSE
└── README.md
```

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/hafufu-stack/snn-synthesis.git
cd snn-synthesis

# Install dependencies (LLM experiments)
pip install torch transformers bitsandbytes snntorch matplotlib numpy

# Install dependencies (ARC-AGI-3 experiments)
pip install arcprize
```

## 📄 Papers

- **SNN-Synthesis v4** (latest): [Zenodo (PDF)](https://doi.org/10.5281/zenodo.19343952)
  - 30 experiments (Phases 1–30)
  - **LLM Noisy Beam Search**: Mistral-7B achieves **100% solve rate** at K=11 (Phase 29)
  - **Architecture invariance**: Same K scaling law on 63K CNN and 7B Transformer
  - **Extended Two-Condition Map**: All 7 ARC-AGI-3 games classified (Phase 28)
  - **Learnability threshold**: Frame stacking shows continuous but thresholded learnability (Phase 27)
  - **Dynamic K ablation**: Fixed K=11 is optimal (Phase 30)
  - v1–v3 findings retained: SNN-ExIt 99%, scale invariance, static noise optimality, Prior Override

- **SNN-Synthesis v3**: [Zenodo (PDF)](https://doi.org/10.5281/zenodo.19422317)
  - 26 experiments (Phases 1–26) + Phase 6b (MoA)
  - **Noisy Beam Search**: K parallel noisy trajectories, 78% L2 clear rate (from 12%)
  - **SNN-ExIt**: Oracle-free self-evolution, 99% on LS20 from zero knowledge
  - **Two-Condition Theory**: Activation energy + state-action learnability predict ExIt success/failure

- **SNN-Synthesis v2**: [Zenodo (PDF)](https://doi.org/10.5281/zenodo.19373028)
  - 8 experiments (Phases 1–7 + Phase 8: ARC-AGI-3)

- **SNN-Synthesis v1**: [Zenodo (PDF)](https://doi.org/10.5281/zenodo.19343953)
  - 7 experiments, N=100 validated with Fisher exact tests

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
  title = {SNN-Synthesis v4: Noisy Beam Search Scales from 63K CNNs to 7B LLMs---100\% Solve Rate via Architecture-Invariant Stochastic Resonance},
  year = {2026},
  doi = {10.5281/zenodo.19343952},
  publisher = {Zenodo},
  url = {https://doi.org/10.5281/zenodo.19343952}
}
```

## 📜 License

MIT License
