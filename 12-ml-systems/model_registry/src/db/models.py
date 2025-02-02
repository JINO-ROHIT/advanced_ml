from sqlalchemy import Column, DateTime, ForeignKey, String, Text
from sqlalchemy.sql.functions import current_timestamp
from sqlalchemy.types import JSON
from src.db.database import Base


class Project(Base):
    __tablename__ = "projects"

    project_id = Column(
        String(255),
        primary_key=True,
        comment="project id",
    )
    project_name = Column(
        String(255),
        nullable=False,
        unique=True,
        comment="project name",
    )
    description = Column(
        Text,
        nullable=True,
        comment="description",
    )
    created_datetime = Column(
        DateTime(timezone=True),
        server_default=current_timestamp(),
        nullable=False,
    )


class Model(Base):
    __tablename__ = "models"

    model_id = Column(
        String(255),
        primary_key=True,
        comment="model id",
    )
    project_id = Column(
        String(255),
        ForeignKey("projects.project_id"),
        nullable=False,
        comment="project id",
    )
    model_name = Column(
        String(255),
        nullable=False,
        comment="model name",
    )
    description = Column(
        Text,
        nullable=True,
        comment="description",
    )
    created_datetime = Column(
        DateTime(timezone=True),
        server_default=current_timestamp(),
        nullable=False,
    )


class Experiment(Base):
    __tablename__ = "experiments"

    experiment_id = Column(
        String(255),
        primary_key=True,
        comment="expt id",
    )
    model_id = Column(
        String(255),
        ForeignKey("models.model_id"),
        nullable=False,
        comment="model id",
    )
    model_version_id = Column(
        String(255),
        nullable=False,
        comment="version id",
    )
    parameters = Column(
        JSON,
        nullable=True,
        comment="params",
    )
    training_dataset = Column(
        Text,
        nullable=True,
        comment="train ds",
    )
    validation_dataset = Column(
        Text,
        nullable=True,
        comment="val ds",
    )
    test_dataset = Column(
        Text,
        nullable=True,
        comment="test ds",
    )
    evaluations = Column(
        JSON,
        nullable=True,
        comment="evals",
    )
    artifact_file_paths = Column(
        JSON,
        nullable=True,
        comment="artifacts",
    )
    created_datetime = Column(
        DateTime(timezone=True),
        server_default=current_timestamp(),
        nullable=False,
    )