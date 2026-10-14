from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from predict import NextWordPredictor

app = FastAPI(title="Lyric Predictor API")

predictor = NextWordPredictor("models/lstm.pt")


class PredictRequest(BaseModel):
    seed: str = ""
    n_words: int = Field(default=30, ge=1, le=100)


@app.post("/api/predict")
def api_predict(req: PredictRequest):
    seed = req.seed.strip()
    if not seed:
        return JSONResponse(status_code=400, content={"error": "Please enter a seed phrase."})

    text = predictor.predict(seed, n_words=req.n_words)
    return {"text": text}