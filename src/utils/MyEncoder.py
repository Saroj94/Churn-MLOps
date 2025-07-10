from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

# Custom label encoder wrapper
class MyLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.encoders[col] = le
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for col in X.columns:
            le = self.encoders[col]
            X_encoded[col] = le.transform(X[col])
        return X_encoded

    def inverse_transform(self, X):
        X_decoded = X.copy()
        for col in X.columns:
            le = self.encoders[col]
            X_decoded[col] = le.inverse_transform(X[col])
        return X_decoded