from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import numpy as np
from pathlib import Path

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent

app.mount("/static", StaticFiles(directory=BASE_DIR / "templates"), name="static")

@app.get("/")
async def root():
    return FileResponse(BASE_DIR / "templates" / "index.html")

class AHPRequest(BaseModel):
    decision: str
    criteria: list
    alternatives: list
    criteria_comparisons: List[float]
    alt_data: List[List[float]]

@app.post("/api/calculate")
async def calculate(req: AHPRequest):
    return {"message": "Backend working ✅", "decision": req.decision}
