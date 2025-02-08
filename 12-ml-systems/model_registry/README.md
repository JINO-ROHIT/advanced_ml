# Model Registry using PostgreSQL for ML Experiments

This project provides a model registry for machine learning experiments using FastAPI and PostgreSQL. It allows you to manage projects, models, and experiments efficiently.

## Features

- **Projects**: Create and manage projects.
- **Models**: Add and retrieve models associated with projects.
- **Experiments**: Track experiments, including parameters, datasets, evaluations, and artifact file paths.

## Endpoints

### Projects
- `GET /projects/all`: Retrieve all projects.
- `GET /projects/id/{project_id}`: Retrieve a project by its ID.
- `GET /projects/name/{project_name}`: Retrieve a project by its name.
- `POST /projects`: Add a new project.

### Models
- `GET /models/all`: Retrieve all models.
- `GET /models/id/{model_id}`: Retrieve a model by its ID.
- `GET /models/project-id/{project_id}`: Retrieve models by project ID.
- `GET /models/name/{model_name}`: Retrieve models by name.
- `POST /models`: Add a new model.

### Experiments
- `GET /experiments/all`: Retrieve all experiments.
- `GET /experiments/id/{experiment_id}`: Retrieve an experiment by its ID.
- `GET /experiments/model-version-id/{model_version_id}`: Retrieve experiments by model version ID.
- `GET /experiments/model-id/{model_id}`: Retrieve experiments by model ID.
- `GET /experiments/project-id/{project_id}`: Retrieve experiments by project ID.
- `POST /experiments`: Add a new experiment.
- `POST /experiments/evaluations/{experiment_id}`: Update experiment evaluations.
- `POST /experiments/artifact-file-paths/{experiment_id}`: Update experiment artifact file paths.

## Setup

### Environment Variables

1. Create a `.env` file with the following content from the `.env.example`

2. Build the containers:

```bash
make build
```

3. Start the containers:

```bash
make up
```

3. [Optionally] push the containers to your registry:

```bash
make push
```

4. Shut down the containers

```bash
make down
```