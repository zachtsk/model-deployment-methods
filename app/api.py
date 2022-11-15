import json
import os
import joblib
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, request
from google.cloud import storage

app = Flask(__name__)

runtime = os.environ.get('RUNTIME', 'dev')
if runtime != 'prod':
    load_dotenv('local.env')

storage_client = storage.Client()
bucket = storage_client.get_bucket(os.environ['CLOUD_BUCKET_NAME'])

### Load Feature Engineering Pipeline
transformer_blob = bucket.blob(os.environ['FEATURE_PIPELINE'])
with transformer_blob.open(mode="rb") as file:
    transformer = joblib.load(file)

### Load GBT Model
model_blob = bucket.blob(os.environ['GBT_MODEL'])
with model_blob.open(mode="rb") as file:
    model_gbt = joblib.load(file)


@app.post("/gbt")
def gbt():
    # Load as Pandas dataframe, transform, predict
    df = pd.DataFrame(json.loads(request.data))
    df_ = transformer.transform(df)
    yhats = model_gbt.predict(df_)
    return str(yhats)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
