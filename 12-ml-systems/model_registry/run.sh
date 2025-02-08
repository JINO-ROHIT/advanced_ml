set -eu

APP_NAME=${APP_NAME:-"src.api.main:app"}
uvicorn ${APP_NAME}