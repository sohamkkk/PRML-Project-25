import numpy as np

class SimpleNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_probs = {}
        self.feature_stats = {}
        for c in self.classes:
            X_c = X[y == c]
            self.class_probs[c] = len(X_c) / len(X)
            self.feature_stats[c] = {
                "mean": np.mean(X_c, axis=0),
                "var": np.var(X_c, axis=0) + 1e-6
            }

    def calculate_prob(self, class_val, x):
        mean = self.feature_stats[class_val]["mean"]
        var = self.feature_stats[class_val]["var"]
        prob = np.exp(- (x - mean)**2 / (2 * var)) / np.sqrt(2 * np.pi * var)
        return np.prod(prob) * self.class_probs[class_val]

    def predict(self, X):
        preds = []
        for x in X:
            class_probs = {c: self.calculate_prob(c, x) for c in self.classes}
            best_class = max(class_probs, key=class_probs.get)
            preds.append(best_class)
        return np.array(preds)
