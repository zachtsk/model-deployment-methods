from pytest import fixture


@fixture(scope="module")
def dataset():
    """From the California Housing Data Dataset"""
    return [{'longitude': -122.23,
             'latitude': 37.88,
             'housing_median_age': 41.0,
             'total_rooms': 880.0,
             'total_bedrooms': 129.0,
             'population': 322.0,
             'households': 126.0,
             'median_income': 8.3252,
             'median_house_value': 452600.0,
             'ocean_proximity': 'NEAR BAY'},
            {'longitude': -122.22,
             'latitude': 37.86,
             'housing_median_age': 21.0,
             'total_rooms': 7099.0,
             'total_bedrooms': 1106.0,
             'population': 2401.0,
             'households': 1138.0,
             'median_income': 8.3014,
             'median_house_value': 358500.0,
             'ocean_proximity': 'NEAR BAY'},
            {'longitude': -122.24,
             'latitude': 37.85,
             'housing_median_age': 52.0,
             'total_rooms': 1467.0,
             'total_bedrooms': 190.0,
             'population': 496.0,
             'households': 177.0,
             'median_income': 7.2574,
             'median_house_value': 352100.0,
             'ocean_proximity': 'NEAR BAY'}]
