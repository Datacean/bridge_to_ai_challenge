"""Simple image generator using OpenAI's text-to-image model.

Usage:
  - Fill `05_cv/.env` (or environment) with OPENAI_API_KEY.
  - Run: python content_generator.py --prompt "red running shoe, modern style" --count 2

This script intentionally imports the OpenAI client only when needed so
syntax checks or static analysis won't attempt a network call.
"""
from __future__ import annotations

from openai import OpenAI
import argparse
import base64
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from dotenv import load_dotenv


def slugify(text: str, maxlen: int = 40) -> str:
    """Create a short, filesystem-safe slug from a prompt."""
    text = text.lower()
    # keep letters, numbers, hyphens
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:maxlen].rstrip("-") or "image"


def save_base64_png(b64: str, out_path: Path) -> None:
    data = base64.b64decode(b64)
    out_path.write_bytes(data)


def generate_images(
    prompt: str,
    api_key: str,
    model: str = "dall-e-2",
    size: str = "1024x1024",
    count: int = 1,
    output_dir: str | Path = "outputs",
) -> List[Path]:
    """Call OpenAI Images API and save generated PNGs.

    Returns list of saved file paths.
    """

    client = OpenAI(api_key=api_key)

    # Build the call. The official Images API accepts model, prompt, size, and n (count).
    response = client.images.generate(
        model=model, prompt=prompt, size=size, n=count, response_format="b64_json"
    )

    # Response shape: response.data -> list with items having .b64_json
    items = getattr(response, "data", []) or []

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slug = slugify(prompt)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    saved_paths: List[Path] = []
    for i, item in enumerate(items, start=1):
        # item may be dict-like or object with attribute
        b64 = None
        if isinstance(item, dict):
            b64 = item.get("b64_json")
        else:
            b64 = getattr(item, "b64_json", None)

        if not b64:
            # skip if no image data
            continue

        filename = f"{timestamp}-{slug}-{i}.png"
        out_path = out_dir / filename
        save_base64_png(b64, out_path)
        saved_paths.append(out_path)

    return saved_paths


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate images from text prompts using OpenAI")
    p.add_argument("--prompt", "-p", required=True, help="Text prompt for the image model")
    p.add_argument("--count", "-n", type=int, default=1, help="Number of images to generate (1-10)")
    p.add_argument("--size", "-s", default="1024x1024", help="Image size, e.g. 512x512 or 1024x1024")
    p.add_argument("--model", "-m", default=os.getenv("OPENAI_IMAGE_MODEL", "dall-e-2"), help="Image model to use (change if your org requires verification)")
    p.add_argument("--out", "-o", default="./05_cv/outputs", help="Output directory to save images")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    load_dotenv()  # loads .env if present
    args = parse_args(argv)

    api_key = os.getenv("OPENAI_API_KEY")

    if args.count < 1 or args.count > 10:
        print("--count must be between 1 and 10")
        return 2

    try:
        saved = generate_images(
            prompt=args.prompt,
            api_key=api_key,
            model=args.model,
            size=args.size,
            count=args.count,
            output_dir=args.out,
        )
    except Exception as e:
        print("Error while generating images:", e)
        return 1

    if not saved:
        print("No images returned by the API.")
        return 1

    for p in saved:
        print("Saved:", p)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
