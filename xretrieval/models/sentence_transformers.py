from sentence_transformers import SentenceTransformer

from xretrieval.models.base import TextModel
from xretrieval.models_registry import ModelRegistry


@ModelRegistry.register(
    "sentence-transformers/all-MiniLM-L6-v2",
)
class SentenceTransformerModel(TextModel):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = self.load_model()

    def load_model(self):
        return SentenceTransformer(self.model_id)

    def encode_text(self, text: list[str]):
        return self.model.encode(text)
