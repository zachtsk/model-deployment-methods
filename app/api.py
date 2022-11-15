import json
import os
import pathlib

import joblib
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, request
from google.cloud import storage
from keras.models import load_model
from keras import backend as K

# Speed Up NN loading
K.clear_session()

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

### Download NN files
blobs = bucket.list_blobs(prefix=os.environ['NN_MODEL'])
current_path = pathlib.Path(__file__).parent
for blob in blobs:
    if blob.name.endswith("/"):
        continue
    file_split = blob.name.split("/")
    directory = current_path / "/".join(file_split[0:-1])
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(current_path / blob.name)
model_nn = load_model(current_path / os.environ['NN_MODEL'])


@app.post("/gbt")
def gbt():
    # Load as Pandas dataframe, transform, predict
    df = pd.DataFrame(json.loads(request.data))
    df_ = transformer.transform(df)
    yhats = model_gbt.predict(df_)
    return str(yhats)


@app.post("/nn")
def nn():
    # Load as Pandas dataframe, transform, predict
    df = pd.DataFrame(json.loads(request.data))
    df_ = transformer.transform(df)
    yhats = model_nn.predict(df_).flatten().tolist()
    return str(yhats)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
