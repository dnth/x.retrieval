[colab_badge]: https://img.shields.io/badge/Open%20In-Colab-blue?style=for-the-badge&logo=google-colab
[kaggle_badge]: https://img.shields.io/badge/Open%20In-Kaggle-blue?style=for-the-badge&logo=kaggle

<div align="center">
    <img src="https://raw.githubusercontent.com/dnth/x.retrieval/main/assets/logo.png" alt="x.retrieval" width="500"/>
    <br />
    <br />
    <a href="https://dnth.github.io/x.retrieval" target="_blank" rel="noopener noreferrer"><strong>Explore the docs »</strong></a>
    <br />
    <a href="#-quickstart" target="_blank" rel="noopener noreferrer">Quickstart</a>
    ·
    <a href="https://github.com/dnth/x.retrieval/issues/new?assignees=&labels=Feature+Request&projects=&template=feature_request.md" target="_blank" rel="noopener noreferrer">Feature Request</a>
    ·
    <a href="https://github.com/dnth/x.retrieval/issues/new?assignees=&labels=bug&projects=&template=bug_report.md" target="_blank" rel="noopener noreferrer">Report Bug</a>
    ·
    <a href="https://github.com/dnth/x.retrieval/discussions" target="_blank" rel="noopener noreferrer">Discussions</a>
    ·
    <a href="https://dicksonneoh.com/" target="_blank" rel="noopener noreferrer">About</a>
</div>
Evaluate your multimodal retrieval system with any models and datasets.


Specific inputs:

- A dataset
- A model
- A mode (e.g. `image-to-image`)

Get evaluation metrics:

- A retrieval results dataframe
- A retrieval metrics dataframe

## 🌟 Key Features

- ✅ Supports a wide range of models and datasets.
- ✅ Installation in one line.
- ✅ Run benchmarks with one function call.

## 🚀 Quickstart

[![Open In Colab][colab_badge]](https://colab.research.google.com/github/dnth/x.retrieval/blob/main/nbs/quickstart.ipynb)
[![Open In Kaggle][kaggle_badge]](https://kaggle.com/kernels/welcome?src=https://github.com/dnth/x.retrieval/blob/main/nbs/quickstart.ipynb)

```python
import xretrieval

metrics, results_df = xretrieval.run_benchmark(
    dataset="coco-val-2017",
    model_id="transformers/Salesforce/blip2-itm-vit-g",
    mode="text-to-text",
)

metrics
```


```bash
{
    'MRR': 0.2953,
    'NormalizedDCG': 0.3469,
    'Precision': 0.2226,
    'Recall': 0.4864,
    'HitRate': 0.4864,
    'MAP': 0.2728
}

```

## 📦 Installation

```bash
pip install xretrieval
```

## 🛠️ Usage

List datasets:

```python
xretrieval.list_datasets()
```

List models:

```python
xretrieval.list_models()
```

## 🧰 Supported Models and Datasets

Models:

```bash
                         Available Models                         
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Model ID                                         ┃ Model Input ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ transformers/Salesforce/blip2-itm-vit-g          │ text-image  │
│ transformers/Salesforce/blip2-itm-vit-g-text     │ text        │
│ transformers/Salesforce/blip2-itm-vit-g-image    │ image       │
│ sentence-transformers/paraphrase-MiniLM-L3-v2    │ text        │
│ sentence-transformers/paraphrase-albert-small-v2 │ text        │
│ sentence-transformers/multi-qa-distilbert-cos-v1 │ text        │
│ sentence-transformers/all-MiniLM-L12-v2          │ text        │
│ sentence-transformers/all-distilroberta-v1       │ text        │
│ sentence-transformers/multi-qa-mpnet-base-dot-v1 │ text        │
│ sentence-transformers/all-mpnet-base-v2          │ text        │
│ sentence-transformers/multi-qa-MiniLM-L6-cos-v1  │ text        │
│ sentence-transformers/all-MiniLM-L6-v2           │ text        │
│ timm/resnet18.a1_in1k                            │ image       │
└──────────────────────────────────────────────────┴─────────────┘
```

Datasets:

```bash
                    Available Datasets                     
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Dataset Name  ┃ Description                             ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ coco-val-2017 │ The COCO Validation Set with 5k images. │
└───────────────┴─────────────────────────────────────────┘
```
