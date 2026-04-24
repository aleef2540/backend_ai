from fastapi import FastAPI
from app.core.cors import setup_cors

from app.modules.ai_custom.router import router as ai_custom_router
from app.modules.ai_self_learning.router import router as ai_self_learning_router
from app.modules.ai_coach.router import router as ai_coach_router

app = FastAPI(title="Entraining Chat API")


setup_cors(app)

app.include_router(ai_custom_router)
app.include_router(ai_self_learning_router)
app.include_router(ai_coach_router)