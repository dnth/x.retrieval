import bm25s
import numpy as np
import Stemmer

from xretrieval.models_registry import ModelRegistry


@ModelRegistry.register("xhluca/bm25s", model_input="text")
class BM25sModel:
    def __init__(self, model_id: str = "xhluca/bm25s"):
        self.model_id = model_id
        self.model = self.load_model()

        self.corpus_tokens = None
        self.stemmer = Stemmer.Stemmer("english")

    def load_model(self):
        return bm25s.BM25()

    def tokenize_text(self, text: list[str]):
        corpus_tokens = bm25s.tokenize(text, stopwords="en", stemmer=self.stemmer)
        self.model.index(corpus_tokens)
        self.corpus_tokens = corpus_tokens

    def retrieve(self, queries: list[str], top_k: int) -> np.ndarray:
        queries_tokens = bm25s.tokenize(queries, stopwords="en", stemmer=self.stemmer)
        results = self.model.retrieve(
            queries_tokens, k=top_k + 1
        )  # +1 for self-matches

        retrieved_ids = []
        # Filter self matches for each query
        for idx, docs in enumerate(results.documents):
            filtered_docs = [doc for doc in docs if doc != idx][:top_k]
            retrieved_ids.append(filtered_docs)
        retrieved_ids = np.array(retrieved_ids)

        return retrieved_ids
