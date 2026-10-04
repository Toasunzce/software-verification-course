import numpy as np

from app.ml.embedder import Embedder


class ClassifierService:

    def __init__(self, embedder: Embedder, categories: list[str]):
        self._embedder = embedder
        self._categories = categories
        self._category_vectors = None

    def _get_category_vectors(self) -> np.ndarray:
        if self._category_vectors is None:
            self._category_vectors = np.array(self._embedder.embed(self._categories))
        return self._category_vectors

    def classify_vector(self, vector) -> tuple[str, float]:
        v = np.array(vector)
        cats = self._get_category_vectors()
        sims = cats @ v / (np.linalg.norm(cats, axis=1) * np.linalg.norm(v) + 1e-9)
        best = int(np.argmax(sims))
        return self._categories[best], float(sims[best])

    def classify_text(self, text: str) -> tuple[str, float]:
        return self.classify_vector(self._embedder.embed([text])[0])