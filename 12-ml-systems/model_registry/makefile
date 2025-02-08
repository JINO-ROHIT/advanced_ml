DOCKER_REPOSITORY := jinorohit/ml_systems


DOCKERFILE := Dockerfile
DOCKER_COMPOSE := docker-compose.yaml
IMAGE_VERSION := 0.1

MODEL_DB := model_registry
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

.PHONY: up
up:
	docker-compose \
		-f ./$(DOCKER_COMPOSE) \
		up -d

.PHONY: down
down:
	docker-compose \
		-f ./$(DOCKER_COMPOSE) \
		down