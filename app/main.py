from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.cors import setup_cors

from app.modules.ai_custom.router import router as ai_custom_router
from app.modules.ai_self_learning.router import router as ai_self_learning_router
from app.modules.ai_coach.router import router as ai_coach_router
from app.modules.ai_sale.router import router as ai_sale_router
from app.modules.ai_assis.router import router as ai_assis


app = FastAPI(title="Entraining Chat API")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()

    print("[422 VALIDATION ERROR]", exc.errors(), flush=True)
    print("[422 BODY]", body.decode("utf-8", errors="ignore"), flush=True)

    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": body.decode("utf-8", errors="ignore"),
        },
    )


setup_cors(app)

app.include_router(ai_custom_router)
app.include_router(ai_self_learning_router)
app.include_router(ai_coach_router)
app.include_router(ai_sale_router)
app.include_router(ai_assis)