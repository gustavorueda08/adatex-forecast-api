from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import health, forecasts

app = FastAPI(
    title="Adatex Demand Forecast API",
    description=(
        "Advanced demand forecasting powered by Prophet. "
        "Provides product-level forecasts with confidence intervals, "
        "safety stock calculations, and ABC-XYZ product classification."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://www.adatex.com.co",
        "https://adatex.com.co",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(forecasts.router, prefix="/forecasts", tags=["forecasts"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port, reload=True)
