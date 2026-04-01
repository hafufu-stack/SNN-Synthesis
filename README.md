# SNN-Synthesis: Scale-Invariant Stochastic Resonance in Neural Networks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19343952.svg)](https://doi.org/10.5281/zenodo.19343952)

> **From External Noise Intervention to Native Self-Evolving Architectures — Now Spanning 63K to 7B Parameters**

Successor to [SNN-Genesis](https://github.com/hafufu-stack/snn-genesis) (v1–v20, 111 phases, 127 pages).
SNN-Genesis dissected the black box of LLM reasoning through noise intervention. SNN-Synthesis uses that anatomical map to **build new AI architectures** and proves that stochastic resonance is a **universal neural network phenomenon**.

## 🔬 Research Vision

SNN-Genesis was the **Anatomy & Physiology** phase — discovering the physical laws of reasoning (stochastic resonance, Aha! dimensions, layer localization).

SNN-Synthesis is the **Architecture & Synthesis** phase — building systems that internalize those laws, and proving their **universality across five orders of magnitude** in model scale.

### 🏆 Key Result (v2)

**Stochastic resonance is scale-invariant.** The optimal noise σ*=0.2 is identical in:
- **63K-parameter CNNs** (Micro-Brain, ARC-AGI-3 grid navigation)
- **244K-parameter CNNs** (Micro-Brain, ARC-AGI-3 grid navigation)
- **7B-parameter Transformers** (Mistral/Qwen, Modified Hanoi reasoning)

This establishes that the optimal noise level is determined by **task structure, not model capacity**.

## 📁 Project Structure

```
snn-synthesis/
├── experiments/          # LLM experiment scripts (Phases 1-7)
├── arc-agi/              # ARC-AGI-3 experiments (Phase 8)
│   ├── arc_micro_brain.py      # CNN noise sweep pipeline
│   ├── arc_oracle_dataset.py   # Oracle trajectory extraction
│   ├── agent_ls20_v14.py       # BFS Oracle solver
│   ├── data/                   # Training data & model weights
│   └── results/                # Final results (JSON + PNG)
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

- **SNN-Synthesis v2** (latest): [Zenodo (PDF)](https://doi.org/10.5281/zenodo.19343952)
  - 8 experiments (Phases 1–7 from v1 + Phase 8: ARC-AGI-3)
  - **Phase 8 highlight**: CNN agent (63K/244K params) shows inverse-U stochastic resonance at σ=0.2 on ARC-AGI-3, matching 7B LLM findings → scale invariance across 5 orders of magnitude
  - v1 findings retained: L18 trajectory injection 13%→47% (p<10⁻⁶), catastrophic dual-layer interference (0%), strict task specificity of Aha! vectors

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
  title = {SNN-Synthesis v2: Scale-Invariant Stochastic Resonance---From Billion-Parameter LLMs to Micro-Brain CNNs},
  year = {2026},
  doi = {10.5281/zenodo.19343952},
  publisher = {Zenodo},
  url = {https://doi.org/10.5281/zenodo.19343952}
}
```

## 📜 License

MIT License
