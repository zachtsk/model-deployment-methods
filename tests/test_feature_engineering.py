import numpy as np
import pandas as pd

from models.feature_engineering import FeatureEngineer


def test_feature_engineer(dataset):
    df = pd.DataFrame(dataset)
    fe = FeatureEngineer()
    fe.fit(df)
    df_mod = fe.transform(df)
    scaled_to_unit_variance = [
        [0.23378595, -0.80920686, -0.77425914, -0.79747631, -0.7611782, 0.73087484, 1.],
        [-1.32478705, 1.40903769, 1.41201595, 1.41018447, 1.41279829, 0.68306936, 1.],
        [1.0910011, -0.59983083, -0.63775681, -0.61270816, -0.65162009, -1.4139442, 1.]
    ]
    assert all(np.isclose(df_mod, scaled_to_unit_variance).tolist())
