import os
from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split
from models.feature_engineering import FeatureEngineer
import tensorflow as tf


def train_nn(df: pd.DataFrame,
             feature_cols: List[str],
             target_col: str,
             batch_size: int = 32,
             epochs: int = 100):
    xtrain, xtest, ytrain, ytest = train_test_split(df.loc[:, feature_cols],
                                                    df.loc[:, target_col],
                                                    test_size=0.20,
                                                    random_state=42)

    # Fit scalers only on training data to avoid data leakage
    feature_transformer = FeatureEngineer()
    feature_transformer.fit(xtrain)
    xtrain = feature_transformer.transform(xtrain)

    # Fit model on transformed inputs
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(12, input_dim=xtrain.shape[1], activation='relu'),
        tf.keras.layers.Dense(12, input_dim=xtrain.shape[1], activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss=tf.losses.MeanAbsolutePercentageError(), optimizer='adam')
    model.fit(xtrain, ytrain, batch_size=batch_size, epochs=epochs)
    model_card = dict(
        model=model,
        feature_transformer=feature_transformer,
        feature_cols=feature_cols,
        target=target_col
    )
    return model_card
