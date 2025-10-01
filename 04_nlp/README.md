# Product review sentiment + urgency API

This folder contains a small FastAPI service that analyzes product reviews for sentiment and (optionally) whether the review needs urgent attention.

Files
- `product_analytics_intelligence.py` — original/simple FastAPI example that demonstrates sentiment-only analysis (single-model pipeline).
- `product_analytics_intelligence2.py` — FastAPI app. POST JSON to `/analyze` to get sentiment and urgency information for reviews.
- `requirements.txt` — Python dependencies used by the app.

What it does
- Uses a Hugging Face `sentiment-analysis` pipeline (default `distilbert-base-uncased-finetuned-sst-2-english`) to predict sentiment labels and confidence scores for each review.
- Optionally runs a zero-shot `zero-shot-classification` pipeline (default model `facebook/bart-large-mnli`) to classify reviews as `urgent` or `standard`.

Quick start
1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the API:

```bash
python product_analytics_intelligence2.py
```

The app listens on http://0.0.0.0:8000 by default.

API usage
POST /analyze
Request JSON shape:

```json
{
  "reviews": ["I need a refund ASAP, the item arrived broken.", "Lovely product, very satisfied."],
  "model": "distilbert-base-uncased-finetuned-sst-2-english",     // optional
  "include_urgency": true,                                        // optional (default true)
  "urgency_model": "facebook/bart-large-mnli"                    // optional
}
```

Response example:

```json
{
  "sentiment_model": "distilbert-base-uncased-finetuned-sst-2-english",
  "urgency_model": "facebook/bart-large-mnli",
  "results": [
    {
      "review": "I need a refund ASAP, the item arrived broken.",
      "label": "NEGATIVE",
      "confidence": 0.999,
      "urgency": {"label": "urgent", "confidence": 0.94}
    },
    {
      "review": "Lovely product, very satisfied.",
      "label": "POSITIVE",
      "confidence": 0.995,
      "urgency": {"label": "standard", "confidence": 0.89}
    }
  ]
}
```

Notes & tips
- The first time models are used they will be downloaded from Hugging Face (internet required). The `facebook/bart-large-mnli` model is large; expect a noticeable download and memory usage.
- If you only care about sentiment, set `include_urgency` to `false` to skip loading the zero-shot model.
- For production, consider fine-tuning a small supervised classifier for urgency (faster and more accurate for a fixed domain) or using a distilled MNLI model to reduce memory and latency.
