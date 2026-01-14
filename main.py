from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rul_predictor import RULPredictor

# =========================
# APP SETUP
# =========================
app = FastAPI(title="RUL Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LOAD PREDICTOR
# =========================
predictor = RULPredictor("rul_model.pkl")

# =========================
# REQUEST SCHEMA
# =========================
class RULRequest(BaseModel):
    data: dict

# =========================
# ROUTES
# =========================
@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "RUL Prediction API"
    }

@app.post("/predict")
def predict_rul(request: RULRequest):
    rul = predictor.predict(request.data)

    return {
        "predicted_rul_hours": round(rul, 2)
    }
