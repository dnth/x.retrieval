from dataclasses import dataclass
from typing import Any, Callable, Literal, Type

# Type aliases for clarity
ModelType = Type[Any]  # Type of the model class
DecoratorFunction = Callable[[ModelType], ModelType]


@dataclass
class ModelInfo:
    """Stores information about a registered model."""

    model_id: str
    model_class: ModelType
    model_input: Literal["text", "image", "multi-modal"]


class ModelRegistry:
    """Registry for managing Sentence Transformer models."""

    _models: dict[str, ModelInfo] = {}

    @classmethod
    def register(
        cls, model_id: str, model_input: Literal["text", "image", "text-image"]
    ) -> DecoratorFunction:
        """Decorator to register a model.

        Args:
            model_id: HuggingFace model ID for the Sentence Transformer

        Returns:
            Decorator function that registers the model class
        """

        def decorator(model_class: ModelType) -> ModelType:
            if model_id in cls._models:
                raise ValueError(f"Model '{model_id}' is already registered")

            cls._models[model_id] = ModelInfo(
                model_id=model_id,
                model_class=model_class,
                model_input=model_input,
            )
            return model_class

        return decorator

    @classmethod
    def get(cls, model_id: str) -> ModelType:
        """Retrieve a model class by name."""
        if model_id not in cls._models:
            raise KeyError(f"Model '{model_id}' not found in registry")

        model_info = cls._models[model_id]
        return model_info.model_class

    @classmethod
    def get_model_info(cls, model_id: str) -> ModelInfo:
        """Retrieve a model info by name."""
        if model_id not in cls._models:
            raise KeyError(f"Model '{model_id}' not found in registry")

        return cls._models[model_id]

    @classmethod
    def list(cls) -> dict[str, str]:
        """List all registered models with their types.

        Returns:
            A dictionary mapping model IDs to their types (text/image/multi-modal)
        """
        return {model_id: info.model_input for model_id, info in cls._models.items()}
