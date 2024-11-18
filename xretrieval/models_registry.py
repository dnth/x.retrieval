from dataclasses import dataclass
from typing import Any, Callable, Type

# Type aliases for clarity
ModelType = Type[Any]  # Type of the model class
DecoratorFunction = Callable[[ModelType], ModelType]


@dataclass
class ModelInfo:
    """Stores information about a registered model."""

    name: str
    model_class: ModelType
    model_id: str  # HuggingFace model ID
    description: str = ""


class ModelRegistry:
    """Registry for managing Sentence Transformer models."""

    _models: dict[str, ModelInfo] = {}

    @classmethod
    def register(cls, model_id: str, description: str = "") -> DecoratorFunction:
        """Decorator to register a model.

        Args:
            model_id: HuggingFace model ID for the Sentence Transformer
            description: Optional description of the model

        Returns:
            Decorator function that registers the model class
        """

        def decorator(model_class: ModelType) -> ModelType:
            if model_id in cls._models:
                raise ValueError(f"Model '{model_id}' is already registered")

            cls._models[model_id] = ModelInfo(
                name=model_id,
                model_class=model_class,
                model_id=model_id,
                description=description,
            )
            return model_class

        return decorator

    @classmethod
    def get(cls, name: str) -> ModelType:
        """Retrieve a model class by name."""
        if name not in cls._models:
            raise KeyError(f"Model '{name}' not found in registry")

        model_info = cls._models[name]
        return model_info.model_class

    @classmethod
    def list(cls) -> dict[str, ModelInfo]:
        """List all registered models."""
        return cls._models.copy()
