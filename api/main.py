#api/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import predict, explain, stability

app = FastAPI(
    title="YieldLens API",
    description="Stability-aware sensor reduction for semiconductor yield monitoring",
    version="1.0.0",
)

# Allow the React frontend (running on a different port during dev) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(stability.router)


@app.get("/")
def root():
    return {"status": "YieldLens API running", "docs": "/docs"}