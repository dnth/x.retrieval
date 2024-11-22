List datasets:

```python
import xretrieval
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

![Visualization 1](https://raw.githubusercontent.com/dnth/x.retrieval/main/assets/viz1.png)
![Visualization 2](https://raw.githubusercontent.com/dnth/x.retrieval/main/assets/viz2.png)
