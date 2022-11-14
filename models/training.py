from typing import List
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from models.feature_engineering import FeatureEngineer


def train_gbt(df: pd.DataFrame, feature_cols: List[str], target_col: str):
    xtrain, xtest, ytrain, ytest = train_test_split(df.loc[:, feature_cols],
                                                    df.loc[:, [target_col]],
                                                    test_size=0.20,
                                                    random_state=42)

    # Fit scalers only on training data to avoid data leakage
    feature_transformer = FeatureEngineer()
    feature_transformer.fit(xtrain)
    xtrain = feature_transformer.transform(xtrain)

    # Fit model on transformed inputs
    model = GradientBoostingRegressor()
    model.fit(xtrain, ytrain)

    model_card = dict(
        model=model,
        feature_transformer=feature_transformer,
        feature_cols=feature_cols,
        target=target_col
    )
    return model_card
