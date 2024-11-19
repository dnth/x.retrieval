__version__ = "0.0.1"
__author__ = "Dickson Neoh"
__email__ = "dickson.neoh@gmail.com"

from .core import (
    list_datasets,
    list_models,
    load_dataset,
    load_model,
    run_benchmark,
    visualize_retrieval,
)
from .datasets import COCODataset
from .datasets_registry import DatasetRegistry
from .models import SentenceTransformerModel
from .models_registry import ModelRegistry

__all__ = [
    "list_datasets",
    "load_dataset",
    "list_models",
    "load_model",
    "run_benchmark",
    "visualize_retrieval",
]
