import os
import pathlib
import pickle
from io import BytesIO

import joblib
import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage

from models.model_gbt import train_gbt

load_dotenv('local.env')


def train_and_save_gbt():
    """Create/Save transformer pipeline object and model object"""
    storage_client = storage.Client()
    filepath = pathlib.Path(__file__).parent / "../data/housing.csv"
    df = pd.read_csv(filepath)

    feat_cols = [
        'housing_median_age',
        'total_rooms',
        'total_bedrooms',
        'population',
        'households',
        'median_income',
        'ocean_proximity'
    ]
    target_col = 'median_house_value'
    model_card = train_gbt(df, feat_cols, target_col)
    bucket = storage_client.get_bucket(os.environ['CLOUD_BUCKET_NAME'])

    # Save Transformer
    transformer_bytes = BytesIO()
    joblib.dump(model_card['feature_transformer'], transformer_bytes)
    transformer_blob = bucket.blob(os.environ['FEATURE_PIPELINE'])
    transformer_blob.upload_from_file(transformer_bytes, rewind=True)

    # Save Model
    model_bytes = BytesIO()
    joblib.dump(model_card['model'], model_bytes)
    model_blob = bucket.blob(os.environ['GBT_MODEL'])
    model_blob.upload_from_file(model_bytes, rewind=True)


def check_gbt_model():
    """Test object retrieval from GCS"""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(os.environ['CLOUD_BUCKET_NAME'])

    model_blob = bucket.blob(os.environ['GBT_MODEL'])
    with model_blob.open(mode="rb") as file:
        model = joblib.load(file)

    transformer_blob = bucket.blob(os.environ['FEATURE_PIPELINE'])
    with transformer_blob.open(mode='rb') as file:
        transformer = joblib.load(file)

    df = pd.DataFrame([{'longitude': -122.24,
                        'latitude': 37.85,
                        'housing_median_age': 52.0,
                        'total_rooms': 1467.0,
                        'total_bedrooms': 190.0,
                        'population': 496.0,
                        'households': 177.0,
                        'median_income': 7.2574,
                        'median_house_value': 352100.0,
                        'ocean_proximity': 'NEAR BAY'}])
    df_ = transformer.transform(df)
    ytest = df['median_house_value'][0]
    yhat = model.predict(df_)[0]
    print(f"\nPredicted {yhat:,.0f}\n   Actual {ytest:,.0f}\n\t({(1-(abs(ytest - yhat) / ytest)) * 100:.2f}% MAPE)")


def main():
    """TODO: Move this to CD Process"""
    print(f"*" * 10, 'Train Model')
    train_and_save_gbt()

    print(f"*" * 10, 'Load and Test Model from Cloud Storage')
    check_gbt_model()

    print(f"*" * 10, 'Build Docker Image')
    os.system("docker-compose build")

    print(f"*" * 10, 'Retag Docker Image')
    project_id = os.environ['PROJECT_ID']
    cmd = f"docker tag zachtsk/model-deployment-example:1.0 gcr.io/{project_id}/flask:1.2.3"
    os.system(cmd)

    print(f"*" * 10, 'Push to cloud repo')
    cmd = f"docker push gcr.io/{project_id}/flask:1.2.3"
    os.system(cmd)


if __name__ == "__main__":
    main()
