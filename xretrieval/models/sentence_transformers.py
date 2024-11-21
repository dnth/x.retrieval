import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from xretrieval.models.base import TextModel
from xretrieval.models_registry import ModelRegistry


@ModelRegistry.register("sentence-transformers/all-MiniLM-L6-v2", model_input="text")
@ModelRegistry.register(
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1", model_input="text"
)
@ModelRegistry.register("sentence-transformers/all-mpnet-base-v2", model_input="text")
@ModelRegistry.register(
    "sentence-transformers/multi-qa-mpnet-base-dot-v1", model_input="text"
)
@ModelRegistry.register(
    "sentence-transformers/all-distilroberta-v1", model_input="text"
)
@ModelRegistry.register("sentence-transformers/all-MiniLM-L12-v2", model_input="text")
@ModelRegistry.register(
    "sentence-transformers/multi-qa-distilbert-cos-v1", model_input="text"
)
@ModelRegistry.register(
    "sentence-transformers/paraphrase-albert-small-v2", model_input="text"
)
@ModelRegistry.register(
    "sentence-transformers/paraphrase-MiniLM-L3-v2", model_input="text"
)
class SentenceTransformerModel(TextModel):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = self.load_model()

    def load_model(self):
        return SentenceTransformer(self.model_id)

    def encode_text(self, text: list[str], batch_size: int = 32):
        all_features = []
        for i in tqdm(range(0, len(text), batch_size), desc="Encoding text"):
            batch_text = text[i : i + batch_size]
            features = self.model.encode(batch_text)
            all_features.append(features)
        return np.concatenate(all_features, axis=0)
