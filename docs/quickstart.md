
[colab_badge]: https://img.shields.io/badge/Open%20In-Colab-blue?style=for-the-badge&logo=google-colab
[kaggle_badge]: https://img.shields.io/badge/Open%20In-Kaggle-blue?style=for-the-badge&logo=kaggle

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
    Retrieval Metrics     
┏━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric        ┃ Score  ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ MRR           │ 0.3032 │
│ NormalizedDCG │ 0.3497 │
│ Precision     │ 0.2274 │
│ Recall        │ 0.4898 │
│ HitRate       │ 0.4898 │
│ MAP           │ 0.2753 │
└───────────────┴────────┘

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

```bash
                    Available Datasets                     
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Dataset Name  ┃ Description                             ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ coco-val-2017 │ The COCO Validation Set with 5k images. │
└───────────────┴─────────────────────────────────────────┘
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


Visualize retrieval results:

```python
xretrieval.visualize_retrieval(results_df)
```

![alt text](assets/viz1.png)
![alt text](assets/viz2.png)