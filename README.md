
# Automatic Retrieval Benchmark

This project wants to make it easier to create automated text-image retrieval benchmarks.

User should be able to compute retrieval metrics on their own datasets and models.

- Input - Dataset, Model
- Output - Metrics (mAP, Precision@K, Recall@K, etc.)

## Dataset

- COCO
- Flickr30k
- MSCOCO
- ECCV Captions
- Conceptual Captions
- SBU Captions
- ...

Download COCO validation dataset images:

```bash
mkdir -p data/coco/
cd data/coco/
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
rm val2017.zip
```

Download COCO validation annotations:

```bash
cd data/coco/
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip
```

You should end up with the following folder structure:

```
data/coco/
├── annotations
│   └── captions_val2017.json
└── val2017
    └── 000000000000.jpg
    └── ...
```

## Model

- CLIP
- BLIP
- ...


## Metrics
Use TorchMetrics.

- mAP
- Precision@K
- Recall@K
- ...

## Usage

Define model, dataset and metrics.

```python

model = Model("clip-vit-b-32")
dataset = Dataset("coco")

embed = ["image", "text", "image_text"]
metrics = ["mAP@1", "Precision@1", "Recall@1"]

results = run_benchmark(model, dataset, embed=embed, metrics=metrics)

```

```
{
    'embed': 'image',
    'metrics': {
        'mAP@1': 0.5,
        'Precision@1': 0.5,
        'Recall@1': 0.5
    },
    'embed': 'text',
    'metrics': {
        'mAP@1': 0.5,
        'Precision@1': 0.5,
        'Recall@1': 0.5
    },
    'embed': 'image_text',
    'metrics': {
        'mAP@1': 0.5,
        'Precision@1': 0.5,
        'Recall@1': 0.5
    }
}
```


## References

- [TorchMetrics](https://torchmetrics.readthedocs.io/en/latest/)
- [VectorHub](https://github.com/superlinked/VectorHub/blob/main/research/vision-research/readme.md)
