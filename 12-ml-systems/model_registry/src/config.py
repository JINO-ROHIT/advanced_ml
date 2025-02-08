import os

POSTGRES_USER = "user"
POSTGRES_PASSWORD = "password"
POSTGRES_DB = "model_db"
POSTGRES_SERVER = "postgres"
POSTGRES_PORT = "5432"

class DBConfigurations:
    postgres_username = POSTGRES_USER
    postgres_password = POSTGRES_PASSWORD
    postgres_port = 5432
    postgres_db = POSTGRES_DB
    postgres_server = POSTGRES_SERVER
    sql_alchemy_database_url = (
        f"postgresql://{postgres_username}:{postgres_password}@{postgres_server}:{postgres_port}/{postgres_db}"
    )


class APIConfigurations:
    title = os.getenv("API_TITLE", "Model_DB_Service")
    description = os.getenv("API_DESCRIPTION", "machine learning systems")
    version = os.getenv("API_VERSION", "0.1")