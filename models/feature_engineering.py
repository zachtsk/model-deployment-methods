from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler


class PositionalSelector(BaseEstimator, TransformerMixin):
    def __init__(self, positions: List[int]):
        self.positions = positions

    def fit(self, x: pd.DataFrame, y=None):
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        return np.array(x)[:, self.positions]


class SimpleOneHotEncoder(BaseEstimator, TransformerMixin):
    def fit(self, x: pd.DataFrame, y=None):
        self.values = []
        for c in range(x.shape[1]):
            Y = x[:, c]
            values = {v: i for i, v in enumerate(np.unique(Y))}
            self.values.append(values)
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        x = np.array(x)
        matrices = []
        for c in range(x.shape[1]):
            y = x[:, c]
            matrix = np.zeros(shape=(len(y), len(self.values[c])), dtype=np.int8)
            for i, val in enumerate(y):
                if val in self.values[c]:
                    matrix[i][self.values[c][val]] = 1
            matrices.append(matrix)
        res = np.concatenate(matrices, axis=1)
        return res


class FeatureEngineer:
    def fit(self, df: pd.DataFrame):
        # For categorical values, use a one-hot encoder
        # For numerical values, use a simple imputer & scaler
        category_cols = ['ocean_proximity']
        numeric_cols = [
            'housing_median_age',
            'total_rooms',
            'total_bedrooms',
            'population',
            'households',
            'median_income'
        ]
        numeric_idxs = [idx for idx, name in enumerate(df.columns) if name in numeric_cols]
        category_idxs = [idx for idx, name in enumerate(df.columns) if name in category_cols]
        p1 = make_pipeline(PositionalSelector(numeric_idxs), SimpleImputer(), StandardScaler())
        p2 = make_pipeline(PositionalSelector(category_idxs), SimpleOneHotEncoder())
        self.features = FeatureUnion([
            ('numericals', p1),
            ('categoricals', p2),
        ])
        self.features.fit(df)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.features.transform(df)
