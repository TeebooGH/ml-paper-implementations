# Implémentations de papers ML from scratch

Implémentation de papers fondamentaux en deep learning en utilisant uniquement NumPy (dans un premier temps), afin de comprendre en profondeur les mathématiques sous-jacentes.

Je prévois également de réimplémenter certains papers avec PyTorch pour acquérir des compétences sur les frameworks standards de l'industrie.

## Papers implémentés — 1 en cours

| Paper | Année | Implémentation | Concepts clés |
|-------|-------|----------------|---------------|
| [Attention Is All You Need](papers/attention_is_all_you_need/transformer_torch/) | 2017 | [🔥 PyTorch](papers/attention_is_all_you_need/transformer_torch/) · [📓 Notebook NumPy](papers/attention_is_all_you_need/transformer_numpy.ipynb) | Self-attention, Attention multi-têtes, Encodage positionnel |
| D'autres à venir... | | | |

## Guide de navigation — à l'attention de M. Hossam Afifi

> **Note :** Le dépôt contient deux implémentations distinctes du Transformer. Seule l'implémentation PyTorch constitue le rendu du projet.

| Dossier | Contenu | Pertinent ? |
|---------|---------|-------------|
| `papers/attention_is_all_you_need/transformer_torch/` | **Projet principal** — implémentation PyTorch complète from scratch | ✅ Oui |
| `papers/attention_is_all_you_need/transformer_numpy.ipynb` | Prototype NumPy initial pour explorer les mathématiques | Exploratoire uniquement, **à ne pas regarder** |
| `papers/activation_functions/` | Notebook sur les fonctions d'activation | Sans rapport |

### Par où commencer

1. Lire [`papers/attention_is_all_you_need/transformer_torch/README.md`](papers/attention_is_all_you_need/transformer_torch/README.md) pour une description complète du projet, de l'architecture et des instructions d'exécution.
2. Le modèle est découpé en six fichiers dans `transformer_torch/model/` — chacun correspond à un composant spécifique du paper Transformer.
3. [`transformer_torch/transformer_output.py`](papers/attention_is_all_you_need/transformer_torch/transformer_output.py) assemble l'ensemble et effectue un passage avant complet avec vérification des formes des tenseurs.

---

## Installation

```bash
# Cloner et installer
git clone https://github.com/TeebooGH/ml-paper-implementations.git
cd ml-paper-implementations
uv sync

# Lancer les notebooks, si vous n'utilisez pas un IDE supportant Jupyter Notebooks.
uv run jupyter lab
```

## Pourquoi from scratch ?

Implémenter sans framework oblige à comprendre :
- Les passes avant et arrière
- Le calcul des gradients
- Les considérations de stabilité numérique

## Structure des fichiers

```
ml-paper-implementations/
├── pyproject.toml
├── README.md                           # Vue d'ensemble + liens vers chaque paper
├── uv.lock
│
├── papers/
│   ├── activation_functions             
│   │   ├── activation_functions.ipynb  # Fonctions d'activation courantes et quand les utiliser
│   │
│   ├── attention_is_all_you_need/
│   │   ├── transformer_numpy.ipynb     # Prototype NumPy
│   │   ├── transformer_numpy/          # Implémentation NumPy
│   │   │
│   │   └── transformer_torch/          # ← Projet principal (PyTorch, from scratch)
│   │       ├── README.md               # README du projet (en français)
│   │       ├── transformer_output.py   # Test du passage avant complet
│   │       └── model/
│   │           ├── embeddings.py
│   │           ├── positioning.py
│   │           ├── attention.py
│   │           ├── feed_forward.py
│   │           ├── normalization.py
│   │           └── transformer.py
│   │
│   ├── resnet/                         # (Exemple) Paper futur
│   │   ├── README.md
│   │   └── ... 
│   │
│   └── gan/                            # (Exemple) Paper futur
│       └── ... 
│
└── shared/                             # Utilitaires communs (optionnel)
    ├── __init__.py
    └── viz.py                          # Helpers de visualisation partagés
```