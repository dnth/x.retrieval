from abc import ABC, abstractmethod


class MultiModalModel(ABC):
    """
    A model that takes both text and image as input.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def encode_image(self, image_path: str):
        pass

    @abstractmethod
    def encode_text(self, text: str):
        pass


class TextModel(ABC):
    """
    A model that only takes text as input.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def encode_text(self, text: str):
        pass


class ImageModel(ABC):
    """
    A model that only takes image as input.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def encode_image(self, image_path: str):
        pass
