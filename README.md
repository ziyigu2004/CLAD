# Cluster-Level Attention-Guided Parallel Decoding for Masked Diffusion Language Models

[**arXiv:2605.29607**](https://arxiv.org/abs/2605.29607)

![overview](assets/overview.png)

## 🚀 Introduction

We propose **CLAD**, a training-free parallel decoding method for masked diffusion language models.

CLAD speeds up MDLM inference by moving the decoding unit from individual tokens to reliable spans. At each denoising step, it groups neighboring high-confidence masked positions into cluster-level candidates, then uses attention information from the same forward pass to avoid committing strongly dependent clusters together.


## 🔧 Quick Start

```bash
git clone https://github.com/ziyigu2004/CLAD.git
cd CLAD

conda create -n CLAD python=3.10
conda activate CLAD

pip install -r requirements.txt
