# SNN-Synthesis: Internalizing Aha! Trajectories and Cognitive Pacemakers in LLMs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19343952.svg)](https://doi.org/10.5281/zenodo.19343952)

> **From External Noise Intervention to Native Self-Evolving Architectures**

Successor to [SNN-Genesis](https://github.com/hafufu-stack/snn-genesis) (v1–v20, 111 phases, 127 pages).
SNN-Genesis dissected the black box of LLM reasoning through noise intervention. SNN-Synthesis uses that anatomical map to **build new AI architectures**.

## 🔬 Research Vision

SNN-Genesis was the **Anatomy & Physiology** phase — discovering the physical laws of reasoning (stochastic resonance, Aha! dimensions, layer localization).

SNN-Synthesis is the **Architecture & Synthesis** phase — building systems that internalize those laws.

## 📁 Project Structure

```
snn-synthesis/
├── experiments/          # Experiment scripts
├── results/              # Experiment result logs (JSON)
├── figures/              # Experiment figures (PNG)
├── papers/               # Paper sources
├── LICENSE
└── README.md
```

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/hafufu-stack/snn-synthesis.git
cd snn-synthesis

# Install dependencies
pip install torch transformers bitsandbytes snntorch matplotlib numpy
```

## 📄 Paper

- **SNN-Synthesis v1** (latest): [Zenodo (PDF)](https://doi.org/10.5281/zenodo.19343952)
  - 7 experiments, N=100 validated with Fisher exact tests
  - Key findings: L18 trajectory injection 13%→47% (p<10⁻⁶), catastrophic dual-layer interference (0%), strict task specificity of Aha! vectors

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
  title = {SNN-Synthesis v1: Task Specificity, Model Scaling, and Temporal Dynamics of Latent Trajectory Distillation in LLM Reasoning},
  year = {2026},
  doi = {10.5281/zenodo.19343952},
  publisher = {Zenodo},
  url = {https://doi.org/10.5281/zenodo.19343952}
}
```

## 📜 License

MIT License
