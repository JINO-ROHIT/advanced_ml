DOCKER_REPOSITORY := jinorohit/ml_systems


DOCKERFILE := Dockerfile
IMAGE_VERSION := 0.1

MODEL_DB := model_in_image
MODEL_LOAD_PATTERN_PORT := 8000


.PHONY: build
build:
	docker build \
		-t $(DOCKER_REPOSITORY):$(MODEL_DB)_$(IMAGE_VERSION) \
		-f $(DOCKERFILE) \
		.

.PHONY: push
push:
	docker push $(DOCKER_REPOSITORY):$(MODEL_DB)_$(IMAGE_VERSION)

.PHONY: deploy
deploy:
	kubectl apply -f manifests/namespace.yaml
	kubectl apply -f manifests/deployment.yaml

.PHONY: delete
delete:
	kubectl delete ns model-in-image
