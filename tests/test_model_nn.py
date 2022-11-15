import pandas as pd
from models.model_nn import train_nn


def test_train_nn(dataset):
    df = pd.DataFrame(dataset)
    x_cols = [
        'housing_median_age',
        'total_rooms',
        'total_bedrooms',
        'population',
        'households',
        'ocean_proximity'
    ]
    y_col = 'median_income'
    model_card = train_nn(df, x_cols, y_col, epochs=2)
    assert any(model_card)
