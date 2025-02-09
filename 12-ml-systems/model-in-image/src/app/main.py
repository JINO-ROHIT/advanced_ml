from fastapi import FastAPI

from src.app.routers import routers

app = FastAPI(
    title = "Model in Image Serving Pattern"
)

app.include_router(routers.router)