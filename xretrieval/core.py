from .registry import DatasetRegistry


def list_datasets(search: str = ""):
    # Convert wildcard pattern to simple regex-like matching
    search = search.replace("*", "").lower()
    return [ds for ds in DatasetRegistry.list() if search in ds.lower()]


def list_models():
    pass


def load_dataset(name: str):
    dataset_class = DatasetRegistry.get(name)
    return dataset_class.get_dataset()
