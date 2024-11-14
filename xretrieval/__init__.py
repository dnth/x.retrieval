__version__ = "0.0.1"
__author__ = "Dickson Neoh"
__email__ = "dickson.neoh@gmail.com"

from .core import list_datasets, load_dataset
from .datasets import COCODataset
from .registry import DatasetRegistry

__all__ = ["list_datasets", "load_dataset"]
