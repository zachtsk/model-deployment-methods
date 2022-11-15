REGION = us-central1
SERVICE = flask-ml

test:
	python -m pytest tests

flask:
	python -m app.api

init:
	python -m scripts.init

service_yaml:
	python -m scripts.deployment

replace_service: service_yaml
	gcloud run services replace service.yaml --region=$(REGION)

deploy: replace_service
	gcloud run services set-iam-policy $(SERVICE) policy.yaml --region=$(REGION) --quiet

remove:
	gcloud run services delete $(SERVICE) --region=$(REGION) --quiet