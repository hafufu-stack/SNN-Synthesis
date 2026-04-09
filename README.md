# SNN-Synthesis: Oracle-Free LLM Self-Evolution via Stochastic Resonance

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19343952.svg)](https://doi.org/10.5281/zenodo.19343952)

> **LLM-ExIt: 16% → 100% in 3 iterations — Oracle-free self-evolution via noise + natural selection + self-distillation**

Successor to [SNN-Genesis](https://github.com/hafufu-stack/snn-genesis) (v1–v20, 111 phases, 127 pages).
SNN-Genesis dissected the black box of LLM reasoning through noise intervention. SNN-Synthesis uses that anatomical map to **build new AI architectures** and proves that stochastic resonance is a **universal, architecture-invariant neural network phenomenon**—then harnesses it for **autonomous self-evolution without any human supervision**.

## 🔬 Research Vision

SNN-Genesis was the **Anatomy & Physiology** phase — discovering the physical laws of reasoning (stochastic resonance, Aha! dimensions, layer localization).

SNN-Synthesis is the **Architecture & Synthesis** phase — building systems that internalize those laws, proving their **universality across architectures (CNN → Transformer), scales (63K → 7B), and tasks (grid navigation → symbolic reasoning → math)**, and demonstrating that noise + natural selection form a **complete learning paradigm**.

### 🏆 Key Results (v5)

1. **LLM-ExIt achieves Oracle-free self-evolution.** Combining NBS miracle collection with QLoRA fine-tuning, Mistral-7B goes from **16% → 94% → 98% → 100%** in 3 iterations on Modified Hanoi — no Oracle, no reward shaping, no human demonstrations. (Phase 32b)
2. **NBS generalizes to math reasoning.** On GSM8K, NBS achieves **89.5% accuracy** at K=11 (from 53% baseline, +36.5pp). (Phase 31b)
3. **σ\* is task-dependent.** Optimal noise scales inversely with task complexity: Hanoi σ\*=0.15, ARC-AGI σ\*=0.20, GSM8K σ\*=0.01. Population diversity (K) is universally beneficial regardless of σ\*. (Phase 31/31b)
4. **Noisy Beam Search is architecture-invariant.** K=11 achieves **78% on 63K CNN** (ARC-AGI) and **100% on Mistral-7B** (Modified Hanoi, p<10⁻¹⁰). The effect is *amplified* at larger scale (+76pp on LLM vs. +66pp on CNN). (Phase 29)
5. **SNN-ExIt (Expert Iteration):** Oracle-free self-evolution from **zero human knowledge** → **99% clear rate** on LS20 (surpassing Oracle-trained CNN by 21pp). (Phase 20)
6. **Two-Condition Theory (validated on all 7 games + LLM):** ExIt succeeds iff (a) bootstrap miracle rate > 0 ("activation energy") AND (b) the game's state→action mapping is *generalizably* learnable. (Phases 20–32b)
7. **Static noise and fixed K are optimal.** 5 dynamic noise strategies + 3 K schedules all fail to outperform constant parameters. Confidence-gating *destroys* the effect (p=0.017). (Phases 8ext–11, 30)
8. **ExIt is self-healing:** Removing 75% of seed miracles paradoxically improves final performance (57% vs 44%). (Phase 26a)

## 📁 Project Structure

```
snn-synthesis/
├── experiments/          # LLM experiment scripts (Phases 1-7, 3b, 6b, 29, 31, 31b, 32, 32b)
│   ├── phase29_llm_noisy_beam.py    # LLM Noisy Beam Search (v4)
│   ├── phase31_gsm8k_nbs.py         # GSM8K NBS (v5)
│   ├── phase31b_gsm8k_sigma_opt.py  # GSM8K σ optimization (v5)
│   ├── phase32b_llm_exit.py         # LLM-ExIt (v5)
│   └── ...
├── arc-agi/              # ARC-AGI-3 experiments (Phases 8-28, 30)
│   ├── phase20_exit_ls20.py     # SNN-ExIt on LS20
│   ├── phase24_activation_energy.py  # Activation energy curve
│   └── ...
├── results/              # LLM experiment result logs (JSON)
├── figures/              # All experiment figures (PNG, shared with papers)
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

- **SNN-Synthesis v5** (latest): [Zenodo (PDF)](https://doi.org/10.5281/zenodo.19343952)
  - 33 experiments (Phases 1–32b)
  - **LLM-ExIt**: Oracle-free self-evolution 16% → **100%** in 3 iterations (Phase 32b)
  - **GSM8K NBS**: Math reasoning 53% → **89.5%** at K=11, σ\*=0.01 (Phase 31b)
  - **Task-dependent σ\***: Hanoi 0.15, ARC-AGI 0.20, GSM8K 0.01
  - v1–v4 findings retained: architecture invariance, SNN-ExIt 99%, Two-Condition Theory, scale invariance

- **SNN-Synthesis v4**: [Zenodo (PDF)](https://doi.org/10.5281/zenodo.19430135)
  - 30 experiments (Phases 1–30)
  - **LLM Noisy Beam Search**: Mistral-7B achieves **100% solve rate** at K=11 (Phase 29)
  - **Architecture invariance**: Same K scaling law on 63K CNN and 7B Transformer

- **SNN-Synthesis v3**: [Zenodo (PDF)](https://doi.org/10.5281/zenodo.19422317)
  - 26 experiments (Phases 1–26) + Phase 6b (MoA)
  - **Noisy Beam Search**: K parallel noisy trajectories, 78% L2 clear rate (from 12%)
  - **SNN-ExIt**: Oracle-free self-evolution, 99% on LS20 from zero knowledge

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
  title = {SNN-Synthesis v5: Oracle-Free LLM Self-Evolution---From 16\% to 100\% via Stochastic Resonance ExIt},
  year = {2026},
  doi = {10.5281/zenodo.19343952},
  publisher = {Zenodo},
  url = {https://doi.org/10.5281/zenodo.19343952}
}
```

## 📜 License

MIT License
