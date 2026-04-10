# SNN-Synthesis: Knowledge Multiplexing, σ-Diverse Beam Search, and Universal Stochastic Resonance

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19343952.svg)](https://doi.org/10.5281/zenodo.19343952)

> **Knowledge Multiplexing + Hyperparameter-Free Exploration + Multi-Model Universality — Stochastic resonance is architecture-invariant, model-invariant, and scale-invariant**

Successor to [SNN-Genesis](https://github.com/hafufu-stack/snn-genesis) (v1–v20, 111 phases, 127 pages).
SNN-Genesis dissected the black box of LLM reasoning through noise intervention. SNN-Synthesis uses that anatomical map to **build new AI architectures** and proves that stochastic resonance is a **universal, architecture-invariant, model-invariant neural network phenomenon**—then harnesses it for **autonomous self-evolution without any human supervision**.

## 🔬 Research Vision

SNN-Genesis was the **Anatomy & Physiology** phase — discovering the physical laws of reasoning (stochastic resonance, Aha! dimensions, layer localization).

SNN-Synthesis is the **Architecture & Synthesis** phase — building systems that internalize those laws, proving their **universality across architectures (CNN → Transformer), model families (Mistral → Qwen), scales (63K → 7B), and tasks (grid navigation → symbolic reasoning → math → factual QA)**, and demonstrating that noise + natural selection form a **complete learning paradigm**.

### 🏆 Key Results (v6)

**New in v6 (Phases 33–38):**
1. **Knowledge Multiplexing via Discrete ID Gating.** A single 115K CNN stores distinct knowledge for multiple games without interference. 4 alternative approaches (noise modulation, SNN chaos, continuous waves, pink noise) all fail — only categorical gating succeeds, mirroring biological neurotransmitter-based mode switching. (Phases 35–36)
2. **σ-Diverse NBS eliminates hyperparameter tuning.** Assigning different σ to each beam matches best individually-tuned fixed σ across all difficulties — no per-task σ\* calibration needed. (Phase 37a)
3. **Gating acts as positive regularization.** At 115K parameters, gated models *surpass* ungated (0.706 > 0.625). (Phase 38a)
4. **NBS is model-invariant.** Qwen2.5-7B achieves identical 100% at K=11 to Mistral-7B. (Phase 38)
5. **σ\* map extended to 4 tasks.** GSM8K (0.01), TruthfulQA (0.2), Hanoi (0.15), ARC-AGI (0.2). (Phase 34)
6. **GSM8K LLM-ExIt.** 56.5% → 58.0% over 3 iterations — modest but confirms ExIt on open-ended math. (Phase 33)

**Established in v3–v5:**
7. **LLM-ExIt achieves Oracle-free self-evolution.** 16% → 94% → 98% → 100% in 3 iterations on Modified Hanoi. (Phase 32b)
8. **NBS generalizes to math reasoning.** GSM8K 53% → **89.5%** at K=11. (Phase 31b)
9. **Noisy Beam Search is architecture-invariant.** K=11: 78% on 63K CNN, 100% on 7B LLM. (Phase 29)
10. **SNN-ExIt:** Zero knowledge → **99%** on LS20, surpassing Oracle CNN by 21pp. (Phase 20)
11. **Two-Condition Theory:** ExIt succeeds iff miracle rate > 0 AND mapping is generalizably learnable. (Phases 20–32b)
12. **Static noise and fixed K are optimal.** All dynamic strategies fail. (Phases 8ext–11, 30)
13. **ExIt is self-healing.** Fewer seed miracles paradoxically improve performance. (Phase 26a)

## 📁 Project Structure

```
snn-synthesis/
├── experiments/          # LLM experiment scripts (Phases 1-7, 3b, 6b, 29-38)
│   ├── phase29_llm_noisy_beam.py        # LLM NBS (v4)
│   ├── phase31b_gsm8k_sigma_opt.py      # GSM8K σ optimization (v5)
│   ├── phase32b_llm_exit.py             # LLM-ExIt (v5)
│   ├── phase33_gsm8k_exit.py            # GSM8K LLM-ExIt (v6)
│   ├── phase34_sigma_prediction.py      # σ* prediction (v6)
│   ├── phase35c_temporal_distillation.py # ID gating (v6)
│   ├── phase37a_sigma_diverse_nbs.py    # σ-diverse NBS (v6)
│   ├── phase38_multi_model_nbs.py       # Multi-model NBS (v6)
│   ├── phase38a_capacity_scaling.py     # Capacity scaling (v6)
│   └── ...
├── arc-agi/              # ARC-AGI-3 experiments (Phases 8-28, 30) + Kaggle agents
│   ├── phase20_exit_ls20.py         # SNN-ExIt on LS20
│   ├── kaggle_cell2_agent.py        # v8 Curiosity agent
│   ├── kaggle_cell2_agent_llm.py    # v10 LLM+NBS agent
│   └── ...
├── results/              # Experiment result logs (JSON)
├── figures/              # All experiment figures (PNG)
├── papers/               # LaTeX source (v1–v6)
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

- **SNN-Synthesis v6** (latest): [Zenodo (PDF)](https://doi.org/10.5281/zenodo.19343952)
  - 38 experiments (Phases 1–38)
  - **Knowledge Multiplexing**: Discrete ID gating only method that works (5 tested, Phase 35c)
  - **σ-Diverse NBS**: Hyperparameter-free exploration (Phase 37a)
  - **Capacity Scaling**: Gating as positive regularization at 115K (Phase 38a)
  - **Multi-Model**: Qwen2.5-7B matches Mistral-7B (Phase 38)
  - v1–v5 findings retained

- **SNN-Synthesis v5**: [Zenodo (PDF)](https://doi.org/10.5281/zenodo.19481773)
  - 33 experiments (Phases 1–32b)
  - **LLM-ExIt**: 16% → **100%** in 3 iterations (Phase 32b)
  - **GSM8K NBS**: 53% → **89.5%** at K=11, σ\*=0.01 (Phase 31b)

- **SNN-Synthesis v4**: [Zenodo (PDF)](https://doi.org/10.5281/zenodo.19430135)
  - 30 experiments (Phases 1–30)
  - **LLM NBS**: Mistral-7B achieves **100%** at K=11 (Phase 29)

- **SNN-Synthesis v3**: [Zenodo (PDF)](https://doi.org/10.5281/zenodo.19422317)
  - 26 experiments + Phase 6b (MoA)
  - **Noisy Beam Search**: 78% L2 clear rate (from 12%)
  - **SNN-ExIt**: 99% on LS20 from zero knowledge

- **SNN-Synthesis v2**: [Zenodo (PDF)](https://doi.org/10.5281/zenodo.19373028)
  - 8 experiments (Phases 1–7 + Phase 8)

- **SNN-Synthesis v1**: [Zenodo (PDF)](https://doi.org/10.5281/zenodo.19343953)
  - 7 experiments, N=100 validated

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
  title = {SNN-Synthesis v6: Knowledge Multiplexing, $\sigma$-Diverse Beam Search, and Universal Stochastic Resonance from 63K to 7B Parameters},
  year = {2026},
  doi = {10.5281/zenodo.19343952},
  publisher = {Zenodo},
  url = {https://doi.org/10.5281/zenodo.19343952}
}
```

## 📜 License

MIT License
