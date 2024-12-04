[colab_badge]: https://img.shields.io/badge/Open%20In-Colab-blue?style=for-the-badge&logo=google-colab
[kaggle_badge]: https://img.shields.io/badge/Open%20In-Kaggle-blue?style=for-the-badge&logo=kaggle

[python_badge]: https://img.shields.io/badge/Python-3.10+-brightgreen?style=for-the-badge&logo=python&logoColor=white
[pypi_badge]: https://img.shields.io/pypi/v/xretrieval.svg?style=for-the-badge&logo=pypi&logoColor=white&label=PyPI&color=blue
[downloads_badge]: https://img.shields.io/pepy/dt/xretrieval.svg?style=for-the-badge&logo=pypi&logoColor=white&label=Downloads&color=purple
[license_badge]: https://img.shields.io/badge/License-Apache%202.0-green.svg?style=for-the-badge&logo=apache&logoColor=white

[![Python][python_badge]](https://pypi.org/project/xretrieval/)
[![PyPI version][pypi_badge]](https://pypi.org/project/xretrieval/)
[![Downloads][downloads_badge]](https://pypi.org/project/xretrieval/)
![License][license_badge]

<div align="center">
    <img src="https://raw.githubusercontent.com/dnth/x.retrieval/main/assets/logo.png" alt="x.retrieval" width="600"/>
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
    <br />
    <br />
</div>
Evaluate your multimodal retrieval system in 3 lines of code.


## 🌟 Key Features

- ✅ Load datasets and models with one line of code.
- ✅ Built in support for Sentence Transformers, TIMM, BM25, and Transformers models.
- ✅ Run benchmarks and get retrieval metrics like MRR, NormalizedDCG, Precision, Recall, HitRate, and MAP.
- ✅ Visualize retrieval results to understand how your model is performing.
- ✅ Combine retrieval results from multiple models using Reciprocal Rank Fusion (RRF).

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

```

```bash

 Retrieval Metrics @ k=10 
┏━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric        ┃ Score  ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ MRR           │ 0.2358 │
│ NormalizedDCG │ 0.2854 │
│ Precision     │ 0.1660 │
│ Recall        │ 0.4248 │
│ HitRate       │ 0.4248 │
│ MAP           │ 0.2095 │
└───────────────┴────────┘

```

## 📦 Installation
From PyPI:
```bash
pip install xretrieval
```

From source:

```bash
pip install git+https://github.com/dnth/x.retrieval
```

## 🛠️ Usage

List datasets:

```python
xretrieval.list_datasets()
```

```bash
                                     Available Datasets                                      
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Dataset Name                 ┃ Description                                                ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ coco-val-2017                │ The COCO Validation Set with 5k images.                    │
│ coco-val-2017-blip2-captions │ The COCO Validation Set with 5k images and BLIP2 captions. │
│ coco-val-2017-vlrm-captions  │ The COCO Validation Set with 5k images and VLRM captions.  │
└──────────────────────────────┴────────────────────────────────────────────────────────────┘
```

List models:

```python
xretrieval.list_models()
```

```bash
                         Available Models                         
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Model ID                                         ┃ Model Input ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ transformers/Salesforce/blip2-itm-vit-g          │ text-image  │
│ transformers/Salesforce/blip2-itm-vit-g-text     │ text        │
│ transformers/Salesforce/blip2-itm-vit-g-image    │ image       │
│ xhluca/bm25s                                     │ text        │
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


Run benchmarks:

```python
results, results_df = xretrieval.run_benchmark_bm25("coco-val-2017-blip2-captions")
```

Visualize retrieval results:

```python
xretrieval.visualize_retrieval(results_df)
```

![alt text](assets/viz1.png)
![alt text](assets/viz2.png)

Run hybrid search with Reciprocal Rank Fusion (RRF):

```python
results_df = xretrieval.run_rrf([results_df, results_df], "coco-val-2017")
```

See [RRF notebook](nbs/rrf.ipynb) for more details.