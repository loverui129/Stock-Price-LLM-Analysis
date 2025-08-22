from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from .routers import analyze



app = FastAPI(title="Stock Price Prediction API", version="1.0.0", description="API for predicting stock prices")
# register routers
app.include_router(analyze.router)

@app.get("/health")
def health():
    return {"ok": True}