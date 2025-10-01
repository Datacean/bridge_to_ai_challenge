
"""Simple FastAPI app that uses a local Hugging Face Transformers
pipeline for sentiment analysis on product reviews.

The model will be downloaded automatically the first time the pipeline is
initialized (internet required once). After that the model runs locally.

Usage:
1. Install dependencies from `requirements.txt` (pip install -r requirements.txt).
2. Run: python product_analytics_intelligence.py
3. POST JSON to /analyze: {"reviews": ["I love this product!", "Terrible."]}

The endpoint returns a JSON object with predicted label and score for each review.
"""

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from transformers import pipeline, Pipeline


logger = logging.getLogger(__name__)


class ReviewRequest(BaseModel):
	reviews: List[str]
	# Optional: allow changing the model name; default is a small sentiment model
	model: Optional[str] = "distilbert-base-uncased-finetuned-sst-2-english"


app = FastAPI(title="Product Review Sentiment API")

# Global pipeline (initialized on first request or at startup)
sentiment_pipe: Optional[Pipeline] = None
loaded_model: Optional[str] = None


def get_pipeline(model_name: str) -> Pipeline:
	"""Return a cached pipeline for the given model name, loading it if needed."""
	global sentiment_pipe, loaded_model
	if sentiment_pipe is None or loaded_model != model_name:
		logger.info("Loading model %s", model_name)
		try:
			sentiment_pipe = pipeline("sentiment-analysis", model=model_name)
			loaded_model = model_name
		except Exception as exc:
			logger.exception("Failed to load model %s", model_name)
			raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}")
	return sentiment_pipe


@app.post("/analyze", summary="Analyze sentiment for product reviews")
def analyze(req: ReviewRequest):
	if not req.reviews:
		raise HTTPException(status_code=400, detail="No reviews provided")

	pipe = get_pipeline(req.model)

	try:
		preds = pipe(req.reviews)
	except Exception as exc:
		logger.exception("Inference failed")
		raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

	results = []
	for review, pred in zip(req.reviews, preds):
		# pred is like {"label": "POSITIVE", "score": 0.999}
		results.append({"review": review, "label": pred.get("label"), "confidence": pred.get("score")})

	return {"model": req.model, "results": results}


if __name__ == "__main__":
	# Run with: python product_analytics_intelligence.py
	import uvicorn

	uvicorn.run(app, host="0.0.0.0", port=8000)
