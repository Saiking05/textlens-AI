from fastapi import FastAPI
from routes.reddit import router

app = FastAPI()

app.include_router(router)

@app.get("/")
def home():
    return {
        "message": "Welcome to Reddit Trend Hunter 🚀"
    }


