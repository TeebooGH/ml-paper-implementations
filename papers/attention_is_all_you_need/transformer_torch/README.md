# Transformer — « Attention is All You Need » (from scratch)

**NET 4201 — Projet Réseaux de Neurones**  
Télécom SudParis — Hossam Afifi  
Auteurs : **Thibaud Ou** & **Karel Steeve Tchami**

---

## Motivation

Le machine learning moderne est de plus en plus façonné par les grands modèles de langage (LLMs) — de la traduction automatique au raisonnement, en passant par la génération de code. Au cœur de la quasi-totalité de ces LLMs se trouve l'**architecture Transformer**, introduite en 2017 dans l'article [*Attention is All You Need*](https://arxiv.org/abs/1706.03762) de Vaswani et al.

Plutôt que d'utiliser une bibliothèque clé en main, nous avons choisi d'implémenter le Transformer **from scratch en PyTorch**, en travaillant directement à partir de l'article original. L'objectif était double :

- **Comprendre en profondeur les concepts fondamentaux du ML et du NLP** : embeddings, encodage positionnel, attention produit scalaire normalisé, attention multi-têtes, normalisation par couche, connexions résiduelles, et la structure encodeur-décodeur.
- **Développer des compétences de recherche concrètes** — lire un article scientifique, traduire une notation mathématique en code fonctionnel, et valider rigoureusement une implémentation.

Pour Thibaud, ce projet s'inscrit également dans un objectif à plus long terme : poursuivre en recherche en machine learning. Implémenter un Transformer from scratch est l'un des exercices les plus formateurs pour vraiment comprendre le fonctionnement des systèmes NLP modernes.

---

## Architecture

L'implémentation suit le **Transformer de base** décrit dans l'article (Section 3, Tableau 3) :

| Hyperparamètre | Valeur |
|----------------|--------|
| `d_model`      | 512    |
| `n_heads`      | 8      |
| `n_layers`     | 6      |
| `d_ff`         | 2048   |
| `dropout`      | 0.1    |

### Structure des fichiers

```
model/
├── embeddings.py       # Embeddings appris, mis à l'échelle par sqrt(d_model)
├── positioning.py      # Encodage positionnel sinusoïdal fixe
├── attention.py        # ScaledDotProductAttention, MultiHeadAttention,
│                       # SelfAttention, MaskedSelfAttention
├── feed_forward.py     # Réseau feed-forward position par position (FFN)
├── normalization.py    # LayerNormalization + ResidualConnection (Add & Norm)
└── transformer.py      # EncoderLayer, DecoderLayer, Encoder, Decoder, Transformer

transformer_output.py   # Test du passage avant complet avec vérification des formes
visualizations/
└── viz_positional_encoding.py  # Heatmap de la matrice PE
```

---

## Exécution

### Prérequis

Assurez-vous d'avoir Python 3.11+ et les dépendances installées :

```sh
pip install torch matplotlib
```

Ou, en utilisant l'environnement virtuel du projet :

```sh
source .venv/bin/activate
```

### Lancer le test du passage avant complet

Depuis le répertoire `attention_is_all_you_need/` :

```sh
python -m transformer_torch.transformer_output
```

Sortie attendue :
```
--- Testing Full Transformer Model ---
Transformer created with 100,970,632 parameters.

Source shape:    torch.Size([2, 10])
Target shape:    torch.Size([2, 12])
src_mask shape:  torch.Size([2, 1, 1, 10])
tgt_mask shape:  torch.Size([2, 1, 1, 12])

Logits shape: torch.Size([2, 12, 37000])

Full Transformer test passed (shape check).
```

### Visualiser l'encodage positionnel

```sh
python -m transformer_torch.visualizations.viz_positional_encoding
```

Affiche la matrice PE sinusoïdale sous forme de heatmap (position × dimension).

---

## Référence

> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).  
> **Attention is All You Need.**  
> *Advances in Neural Information Processing Systems*, 30.  
> https://arxiv.org/abs/1706.03762
