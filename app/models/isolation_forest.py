import joblib
import numpy as np

class IsolationForestWrapper:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def decision_function(self, X):
        # Normalize to 0–1 for ensemble scoring
        raw_scores = self.model.decision_function(X)
        min_score = np.min(raw_scores)
        max_score = np.max(raw_scores)
        if max_score - min_score == 0:
            return np.zeros_like(raw_scores)
        normalized = (raw_scores - min_score) / (max_score - min_score)
        return normalized.tolist()

    def predict(self, X):
        return self.model.predict(X).tolist()
