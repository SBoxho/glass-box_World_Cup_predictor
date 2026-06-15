"""Phase-2 API stub — intentionally minimal.

Demonstrates the decoupled architecture: this FastAPI app imports the *same* ``core`` functions
the Streamlit app uses, so a future frontend (e.g. React on Vercel) could consume a JSON API
instead of the Streamlit UI. It is not deployed and not the 5-day deliverable — it exists to
show the upgrade path.

Run locally with:  uvicorn api.main:app --reload
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402

from core import config, ingest, model, simulate  # noqa: E402
from core.features import build_inference_state  # noqa: E402

app = FastAPI(
    title="Glass-Box World Cup Predictor API",
    description="Calibrated match predictions + tournament simulation over the same core engine.",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def _predictor() -> model.Predictor:
    matches = (
        pd.read_parquet(config.MATCHES_PATH)
        if config.MATCHES_PATH.exists()
        else ingest.get_clean_matches()
    )
    return model.Predictor(model.load_model(), build_inference_state(matches))


@app.get("/predict")
def predict(home: str, away: str, neutral: bool = True):
    """Calibrated {H, D, A} probabilities for a single match."""
    pred = _predictor()
    home, away = config.normalize_team(home), config.normalize_team(away)
    if home not in pred.state.ratings and away not in pred.state.ratings:
        raise HTTPException(status_code=404, detail="Unknown team(s).")
    probs = pred.predict(home, away, neutral=neutral)
    return {"home": home, "away": away, "neutral": neutral, "probabilities": probs}


@app.post("/simulate")
def run_simulation(n_sims: int = 5000, seed: int = config.SEED):
    """Run the tournament Monte Carlo and return per-team stage probabilities."""
    sim = simulate.TournamentSimulator(_predictor(), ingest.load_wc2026(), seed=seed)
    result = sim.run(n_sims=n_sims, seed=seed)
    return {
        "n_sims": result.n_sims,
        "most_likely_final": result.most_likely_final,
        "table": result.table.to_dict(orient="records"),
    }
