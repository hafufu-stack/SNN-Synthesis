# SNN-Synthesis: Oracle-Free Self-Evolution via Stochastic Resonance

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19343952.svg)](https://doi.org/10.5281/zenodo.19343952)

> **From Noise-Enhanced Inference to Autonomous Self-Evolving Agents — Spanning 63K to 7B Parameters**

Successor to [SNN-Genesis](https://github.com/hafufu-stack/snn-genesis) (v1–v20, 111 phases, 127 pages).
SNN-Genesis dissected the black box of LLM reasoning through noise intervention. SNN-Synthesis uses that anatomical map to **build new AI architectures** and proves that stochastic resonance is a **universal neural network phenomenon**—then harnesses it for **autonomous self-evolution without any human supervision**.

## 🔬 Research Vision

SNN-Genesis was the **Anatomy & Physiology** phase — discovering the physical laws of reasoning (stochastic resonance, Aha! dimensions, layer localization).

SNN-Synthesis is the **Architecture & Synthesis** phase — building systems that internalize those laws, proving their **universality across five orders of magnitude**, and demonstrating that noise + natural selection form a **complete learning paradigm**.

### 🏆 Key Results (v3)

1. **Stochastic resonance is scale-invariant.** σ*≈0.05–0.20 across 63K CNNs → 7B Transformers (p<10⁻⁵⁰ at N=1000).
2. **Static noise is optimal.** 5 dynamic strategies (temporal scheduling, confidence gating, bandit, PPO-SNN, MoA 3-expert routing) all fail to outperform constant σ.
3. **Noisy Beam Search:** K parallel noisy trajectories → 78% L2 clear rate (from 12%) with strict monotonic K scaling.
4. **SNN-ExIt (Expert Iteration):** Oracle-free self-evolution from **zero human knowledge** → **99% clear rate** on LS20 (surpassing Oracle-trained CNN by 21pp).
5. **Two-Condition Theory:** ExIt succeeds iff (a) bootstrap miracle rate > 0 ("activation energy") AND (b) the game's state→action mapping is learnable. TR87 fails at 72% miracle rate because its mapping is unlearnable.
6. **ExIt is self-healing:** Removing 75% of seed miracles paradoxically improves final performance (57% vs 44%).

## 📁 Project Structure

```
snn-synthesis/
├── experiments/          # LLM experiment scripts (Phases 1-7, 3b, 6b)
│   └── phase3b_mistral_ppo.py   # Mistral-7B PPO noise optimization
├── arc-agi/              # ARC-AGI-3 experiments (Phases 8-26)
│   ├── arc_micro_brain.py       # CNN noise sweep pipeline
│   ├── arc_oracle_dataset.py    # Oracle trajectory extraction
│   ├── agent_ls20_v14.py        # BFS Oracle solver (latest)
│   ├── phase8_extended.py       # Extended resonance analysis
│   ├── phase9_dynamic_sigma.py  # Temporal scheduling
│   ├── phase10_confidence_adaptive.py  # Confidence gating
│   ├── phase*_runner.py         # Multi-phase runners
│   ├── phase16c_parallel.py     # Multi-game validation
│   ├── phase18_scaling_law.py   # K scaling law
│   ├── phase20_exit_ls20.py     # SNN-ExIt on LS20
│   ├── phase21_exit_tr87.py     # SNN-ExIt on TR87
│   ├── phase23_visual_exit.py   # Visual ExIt (pixel input)
│   ├── phase24_activation_energy.py  # Activation energy curve
│   ├── phase25_enriched_exit.py      # Enriched state ExIt
│   ├── phase26_ablations.py          # Final ablation studies
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

- **SNN-Synthesis v3** (latest): [Zenodo (PDF)](https://doi.org/10.5281/zenodo.19343952)
  - 26 experiments (Phases 1–26) + Phase 6b (MoA)
  - **Noisy Beam Search**: K parallel noisy trajectories, 78% L2 clear rate (from 12%)
  - **SNN-ExIt**: Oracle-free self-evolution, 99% on LS20 from zero knowledge
  - **Two-Condition Theory**: Activation energy + state-action learnability predict ExIt success/failure
  - **Ablations**: ExIt robust to 75% miracle reduction; K=10 suffices for easy games
  - v1–v2 findings retained: scale invariance (63K–7B), static noise optimality, Prior Override

- **SNN-Synthesis v2**: [Zenodo (PDF)](https://doi.org/10.5281/zenodo.19373028)
  - 8 experiments (Phases 1–7 + Phase 8: ARC-AGI-3)
  - Scale-invariant stochastic resonance at σ=0.2 across 63K–244K CNNs and 7B LLMs

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
  title = {SNN-Synthesis v3: Noisy Beam Search and Oracle-Free Self-Evolution---Scaling Stochastic Resonance from 63K-Parameter CNNs to 7B LLMs},
  year = {2026},
  doi = {10.5281/zenodo.19343952},
  publisher = {Zenodo},
  url = {https://doi.org/10.5281/zenodo.19343952}
}
```

## 📜 License

MIT License
