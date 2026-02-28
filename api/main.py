from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import numpy as np
from pathlib import Path

app = FastAPI(title="AHP Decision Companion")

BASE_DIR = Path(__file__).resolve().parent.parent

# Serve static frontend
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

@app.get("/")
async def root():
    return FileResponse(BASE_DIR / "static" / "index.html")

# ---------------------------
# AHP Constants
# ---------------------------
RI = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
}

# ---------------------------
# Core Logic
# ---------------------------
def calculate_weights(matrix: np.ndarray):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_index = int(np.argmax(eigenvalues.real))
    lambda_max = float(eigenvalues.real[max_index])
    weights = eigenvectors[:, max_index].real
    weights = weights / np.sum(weights)
    return weights.tolist(), lambda_max


def consistency_ratio(n: int, lambda_max: float) -> float:
    if n < 3:
        return 0.0
    CI = (lambda_max - n) / (n - 1)
    return float(CI / RI[n])


def build_matrix(n: int, comparisons: List[float]) -> np.ndarray:
    matrix = np.ones((n, n))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            v = comparisons[idx]
            matrix[i][j] = v
            matrix[j][i] = 1.0 / v
            idx += 1
    return matrix


def normalize_objective(values: List[float], criterion_type: str) -> List[float]:
    arr = np.array(values, dtype=float)
    if criterion_type == "benefit":
        normalized = arr / np.sum(arr)
    else:
        inverted = 1.0 / arr
        normalized = inverted / np.sum(inverted)
    return normalized.tolist()

# ---------------------------
# Models
# ---------------------------
class Criterion(BaseModel):
    name: str
    type: str
    mode: str


class AHPRequest(BaseModel):
    decision: str
    criteria: List[Criterion]
    alternatives: List[str]
    criteria_comparisons: List[float]
    alt_data: List[List[float]]

# ---------------------------
# API Endpoints
# ---------------------------
@app.post("/api/validate-criteria")
async def validate_criteria(payload: dict):
    n = payload["n"]
    comparisons = payload["comparisons"]
    matrix = build_matrix(n, comparisons)
    weights, lmax = calculate_weights(matrix)
    cr = consistency_ratio(n, lmax)
    return {
        "weights": weights,
        "lambda_max": lmax,
        "consistency_ratio": cr,
        "consistent": cr <= 0.1
    }


@app.post("/api/calculate")
async def calculate(req: AHPRequest):

    n_criteria = len(req.criteria)
    n_alt = len(req.alternatives)

    crit_matrix = build_matrix(n_criteria, req.criteria_comparisons)
    crit_weights, crit_lmax = calculate_weights(crit_matrix)
    crit_cr = consistency_ratio(n_criteria, crit_lmax)

    final_scores = np.zeros(n_alt)
    alt_weights_list = []
    detailed_scores = {}
    alt_crs = []

    for i, criterion in enumerate(req.criteria):

        if criterion.mode == "objective":
            alt_weights = normalize_objective(req.alt_data[i], criterion.type)
            alt_crs.append(None)

        else:
            raw_matrix = build_matrix(n_alt, req.alt_data[i])

            if criterion.type == "cost":
                for r in range(n_alt):
                    for c in range(r + 1, n_alt):
                        raw_matrix[r][c] = 1.0 / raw_matrix[r][c]
                        raw_matrix[c][r] = raw_matrix[r][c]

            alt_weights, alt_lmax = calculate_weights(raw_matrix)
            alt_cr = consistency_ratio(n_alt, alt_lmax)
            alt_crs.append(alt_cr)

        alt_weights_list.append(alt_weights)
        contribution = [crit_weights[i] * w for w in alt_weights]
        detailed_scores[criterion.name] = contribution
        final_scores += np.array(contribution)

    ranking = np.argsort(final_scores)[::-1].tolist()

    return {
        "decision": req.decision,
        "criteria_weights": crit_weights,
        "criteria_cr": crit_cr,
        "criteria_consistent": crit_cr <= 0.1,
        "alt_weights_list": alt_weights_list,
        "alt_crs": alt_crs,
        "final_scores": final_scores.tolist(),
        "ranking": ranking,
        "best": req.alternatives[ranking[0]],
        "detailed_scores": detailed_scores
    }
