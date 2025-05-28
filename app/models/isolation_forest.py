class MockIsolationForest:
    def predict(self, X):
        return [0] * len(X)

    def decision_function(self, X):
        return [0.0] * len(X)  # Neutral score

def load_isolation_forest():
    return MockIsolationForest()
