
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


## References

- [TorchMetrics](https://torchmetrics.readthedocs.io/en/latest/)
- [VectorHub](https://github.com/superlinked/VectorHub/blob/main/research/vision-research/readme.md)
