from sentence_transformers import SentenceTransformer

from xretrieval.models_registry import ModelRegistry


@ModelRegistry.register(
    "sentence-transformers/all-MiniLM-L6-v2",
)
class SentenceTransformerModel:
    def __init__(self, model_id: str):
        self.model_id = model_id

    def load_model(self):
        return SentenceTransformer(self.model_id)
