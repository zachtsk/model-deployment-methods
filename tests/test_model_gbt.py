import pandas as pd

from models.model_gbt import train_gbt


def test_train_gbt(dataset):
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
    model_card = train_gbt(df, x_cols, y_col)
    assert any(model_card)