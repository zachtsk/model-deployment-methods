import os
import pathlib

from dotenv import load_dotenv

load_dotenv('local.env')


def create_service_yaml():
    contents = f"""
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: flask-ml
  generation: 1
  annotations:
    run.googleapis.com/client-name: gcloud
    run.googleapis.com/ingress: all
    run.googleapis.com/ingress-status: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: '0'
        autoscaling.knative.dev/maxScale: '1'
    spec:
      containers:
        - image: gcr.io/{os.environ['PROJECT_ID']}/flask:1.2.3
          env:
            - name: CLOUD_BUCKET_NAME
              value: {os.environ['CLOUD_BUCKET_NAME']}
            - name: FEATURE_PIPELINE
              value: {os.environ['FEATURE_PIPELINE']}
            - name: GBT_MODEL
              value: {os.environ['GBT_MODEL']}
            - name: NN_MODEL
              value: {os.environ['NN_MODEL']}
          ports:
            - name: http1
              containerPort: 8080
          resources:
            limits:
              memory: 1024Mi
              cpu: 1000m
  traffic:
    - percent: 100
      latestRevision: true 
    """
    save_path = pathlib.Path(__file__) / '../../service.yaml'
    with open(save_path, 'w') as file:
        file.write(contents)


if __name__ == "__main__":
    create_service_yaml()
