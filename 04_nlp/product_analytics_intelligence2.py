
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
	# Optional: allow changing the sentiment model name; default is a small sentiment model
	model: Optional[str] = "distilbert-base-uncased-finetuned-sst-2-english"

	# Urgency detection (zero-shot by default). You can change the model name or set
	# include_urgency to False to skip urgency classification.
	include_urgency: Optional[bool] = True
	urgency_model: Optional[str] = "facebook/bart-large-mnli"


app = FastAPI(title="Product Review Sentiment API")

# Global pipeline (initialized on first request or at startup)
sentiment_pipe: Optional[Pipeline] = None
loaded_model: Optional[str] = None

# Zero-shot pipeline for urgency / topic-like classification
zero_shot_pipe: Optional[Pipeline] = None
loaded_zero_shot_model: Optional[str] = None

# Default candidate labels for urgency detection
URGENCY_LABELS = ["urgent", "standard"]


def get_zero_shot_pipeline(model_name: str) -> Pipeline:
	"""Return a cached zero-shot pipeline for the given model name, loading it if needed."""
	global zero_shot_pipe, loaded_zero_shot_model
	if zero_shot_pipe is None or loaded_zero_shot_model != model_name:
		logger.info("Loading zero-shot model %s", model_name)
		try:
			zero_shot_pipe = pipeline("zero-shot-classification", model=model_name)
			loaded_zero_shot_model = model_name
		except Exception as exc:
			logger.exception("Failed to load zero-shot model %s", model_name)
			raise HTTPException(status_code=500, detail=f"Failed to load zero-shot model: {exc}")
	return zero_shot_pipe


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

	# Sentiment
	pipe = get_pipeline(req.model)

	try:
		preds = pipe(req.reviews)
	except Exception as exc:
		logger.exception("Inference failed")
		raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

	# Optionally run urgency detection using a zero-shot classifier
	urgency_preds = None
	if req.include_urgency:
		zero_pipe = get_zero_shot_pipeline(req.urgency_model)
		try:
			# The zero-shot pipeline accepts a list of sequences and a candidate_labels arg
			urgency_preds = zero_pipe(req.reviews, candidate_labels=URGENCY_LABELS)
		except Exception as exc:
			logger.exception("Urgency inference failed")
			raise HTTPException(status_code=500, detail=f"Urgency inference error: {exc}")

	results = []
	for idx, (review, pred) in enumerate(zip(req.reviews, preds)):
		# pred is like {"label": "POSITIVE", "score": 0.999}
		item = {"review": review, "label": pred.get("label"), "confidence": pred.get("score")}

		if urgency_preds:
			# urgency_preds item is like {"sequence":..., "labels": [...], "scores": [...]}
			up = urgency_preds[idx]
			# take the top label and its score
			urgency_label = up.get("labels")[0] if up.get("labels") else None
			urgency_score = up.get("scores")[0] if up.get("scores") else None
			item["urgency"] = {"label": urgency_label, "confidence": urgency_score}

		results.append(item)

	return {"sentiment_model": req.model, "urgency_model": (req.urgency_model if req.include_urgency else None), "results": results}


if __name__ == "__main__":
	# Run with: python product_analytics_intelligence.py
	import uvicorn

	uvicorn.run(app, host="0.0.0.0", port=8000)
