# ML Paper Implementations from Scratch

Implementing foundational deep learning papers using only NumPy (at first) to deeply understand the underlying mathematics. 

I'm also planning on reimplementing some papers using PyTorch to build competency in industry-standard frameworks.

## Papers Implemented

| Paper | Year | Notebook | Key Concepts |
|-------|------|----------|--------------|
| [Attention Is All You Need](papers/attention_is_all_you_need/) | 2017 | [📓 Notebook](papers/attention_is_all_you_need/transformer_numpy.ipynb) | Self-attention, Multi-head attention, Positional encoding |
| More coming soon... | | | |

## Setup

```bash
# Clone and install
git clone https://github.com/TeebooGH/ml-paper-implementations.git
cd ml-paper-implementations
uv sync

# Run notebooks, if you're not running this project on an IDE that supports Jupyter Notebooks.
uv run jupyter lab
```

## Why From Scratch? 

Implementing without frameworks forces understanding of:
- Forward and backward passes
- Gradient computation
- Numerical stability considerations

## File Outline

```
ml-paper-implementations/
├── pyproject.toml
├── README.md                           # Overview + links to each paper
├── uv.lock
│
├── papers/
│   ├── activation_functions             
│   │   ├── activation_functions.ipynb  # Generally used activation functions and when to use a certain one
│   │
│   ├── attention_is_all_you_need/
│   │   ├── README.md                   # Paper-specific explanation
│   │   ├── transformer_numpy.ipynb
│   │   └── transformer_numpy/
│   │       ├── __init__. py
│   │       ├── attention.py
│   │       ├── layers.py
│   │       └── model.py
│   │
│   ├── resnet/                         # (Example) Future paper
│   │   ├── README.md
│   │   └── ... 
│   │
│   └── gan/                            # (Example) Future paper
│       └── ... 
│
└── shared/                             # Optional: common utilities
    ├── __init__. py
    └── viz.py                          # Shared visualization helpers
```