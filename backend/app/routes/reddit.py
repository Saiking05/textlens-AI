from fastapi import APIRouter

router = APIRouter()

@router.get("/reddit")
def reddit_home():
    return {
        "message": "Reddit Route Working!"
    }