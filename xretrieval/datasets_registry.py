from dataclasses import dataclass
from typing import Any, Callable, Type

# Type aliases for clarity
DatasetType = Type[Any]  # Type of the dataset class
DecoratorFunction = Callable[[DatasetType], DatasetType]


@dataclass
class DatasetInfo:
    """Stores information about a registered dataset."""

    name: str
    dataset_class: DatasetType
    description: str = ""


class DatasetRegistry:
    """Registry for managing datasets."""

    _datasets: dict[str, DatasetInfo] = {}

    @classmethod
    def register(cls, name: str, description: str = "") -> DecoratorFunction:
        """Decorator to register a dataset.

        Args:
            name: Name for the dataset
            description: Optional description of the dataset

        Returns:
            Decorator function that registers the dataset class
        """

        def decorator(dataset_class: DatasetType) -> DatasetType:
            if name in cls._datasets:
                raise ValueError(f"Dataset '{name}' is already registered")

            cls._datasets[name] = DatasetInfo(
                name=name, dataset_class=dataset_class, description=description
            )
            return dataset_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Any:
        """Retrieve a dataset instance by name."""
        if name not in cls._datasets:
            raise KeyError(f"Dataset '{name}' not found in registry")
        return cls._datasets[name].dataset_class()

    @classmethod
    def list(cls) -> dict[str, DatasetInfo]:
        """List all registered datasets with their descriptions."""
        return {name: info.description for name, info in cls._datasets.items()}
