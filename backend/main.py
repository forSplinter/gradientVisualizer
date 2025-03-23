from fastapi import FastAPI
from api.endpoints import router

app = FastAPI(title="GradientVisu")
app.include_router(router, prefix="/api")


@app.get("/")
def home():
    return {"message": "Welcome to the mother fucker gradient visualizer api"}
