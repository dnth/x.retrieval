from sentence_transformers import SentenceTransformer

from xretrieval.models.base import TextModel
from xretrieval.models_registry import ModelRegistry


@ModelRegistry.register("sentence-transformers/all-MiniLM-L6-v2")
@ModelRegistry.register("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
@ModelRegistry.register("sentence-transformers/all-mpnet-base-v2")
@ModelRegistry.register("sentence-transformers/multi-qa-mpnet-base-dot-v1")
@ModelRegistry.register("sentence-transformers/all-distilroberta-v1")
@ModelRegistry.register("sentence-transformers/all-MiniLM-L12-v2")
@ModelRegistry.register("sentence-transformers/multi-qa-distilbert-cos-v1")
@ModelRegistry.register("sentence-transformers/paraphrase-albert-small-v2")
@ModelRegistry.register("sentence-transformers/paraphrase-MiniLM-L3-v2")
class SentenceTransformerModel(TextModel):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = self.load_model()

    def load_model(self):
        return SentenceTransformer(self.model_id)

    def encode_text(self, text: list[str]):
        return self.model.encode(text)
