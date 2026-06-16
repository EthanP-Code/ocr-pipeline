"""
azure_client.py

Handles all communication with Azure Document Intelligence.

This module is intentionally "dumb", its only job is to send an image to
Azure and return the raw analysis result. Turning that result into clean,
database-ready fields is the job of parser.py (separation of concerns).

To avoid burning API calls while iterating on the parser/evaluator, raw
Azure responses can be cached to disk as JSON. analyze_document() will
read from the cache (if present) instead of calling Azure, unless
force_refresh=True is passed.

Setup required (.env file in project root):
    AZURE_DOCINTEL_ENDPOINT=https://<your-resource-name>.cognitiveservices.azure.com/
    AZURE_DOCINTEL_KEY=<your-api-key>
"""

import json
import os
from pathlib import Path

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentAnalysisFeature
from dotenv import load_dotenv

# Load variables from .env into the environment
load_dotenv()

ENDPOINT = os.environ.get("AZURE_DOCINTEL_ENDPOINT")
KEY = os.environ.get("AZURE_DOCINTEL_KEY")

# Default model. "prebuilt-layout" returns text, lines, words, tables, and
    # bounding boxes/polygons without needing any custom training.
# Swap to a custom model ID (e.g. "specimen-label-extractor") once trained.
DEFAULT_MODEL_ID = "prebuilt-layout"

# Where cached raw Azure responses are stored, as JSON (one file per image,
# keyed by the image's stem + the model id used).
CACHE_DIR = Path("data/processed/azure_cache")

def get_client() -> DocumentIntelligenceClient:
    """
    Create and return an authenticated DocumentIntelligenceClient.

    Raises:
        ValueError: if endpoint/key are missing from the environment.
    """
    if not ENDPOINT or not KEY:
        raise ValueError(
            "Missing Azure credentials. Make sure AZURE_DOCINTEL_ENDPOINT "
            "and AZURE_DOCINTEL_KEY are set in your .env file."
        )
    return DocumentIntelligenceClient(
        endpoint=ENDPOINT, credential=AzureKeyCredential(KEY)
    )


def _cache_path(image_path: str, model_id: str) -> Path:
    """
    Build the cache file path for a given image + model combination.

    Cache files are named "<image_stem>__<model_id>.json" so that results
    from different models (e.g. prebuilt-layout vs. a custom model) for the
    same image don't collide.
    """
    stem = Path(image_path).stem
    safe_model_id = model_id.replace("/", "_")
    return CACHE_DIR / f"{stem}__{safe_model_id}.json"


def _load_cached_result(image_path: str, model_id: str) -> dict | None:
    """Return the cached raw Azure response dict for this image/model, or
    None if no cache entry exists."""
    cache_file = _cache_path(image_path, model_id)
    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: failed to read cache file {cache_file} ({e}). Re-fetching from Azure.")
        return None


def _save_cached_result(image_path: str, model_id: str, raw_dict: dict) -> None:
    """Write the raw Azure response dict to the cache as JSON."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(image_path, model_id)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(raw_dict, f, indent=2, ensure_ascii=False)


def analyze_document(
    image_path: str,
    model_id: str = DEFAULT_MODEL_ID,
    force_refresh: bool = False,
) -> dict:
    """
    Get the raw Azure Document Intelligence analysis for a single document
    image, as a plain JSON-serializable dict.

    By default this checks data/processed/azure_cache/ first and returns
    the cached result if one exists, avoiding a paid API call. Pass
    force_refresh=True to bypass the cache and re-query Azure (the new
    result is written back to the cache, overwriting any old entry).

    Args:
        image_path: Path to the image file (png, jpg, pdf, etc.)
        model_id: Which Azure model to use. Defaults to "prebuilt-layout".
        force_refresh: If True, ignore any cached result and call Azure.

    Returns:
        dict: the analysis result as a plain dict (same shape as
        AnalyzeResult.as_dict()), containing pages, lines, words, and
        (for layout/custom models) tables and key-value pairs.

    Raises:
        FileNotFoundError: if image_path does not exist.
        HttpResponseError: if the Azure API call fails.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not force_refresh:
        cached = _load_cached_result(image_path, model_id)
        if cached is not None:
            return cached

    client = get_client()

    with open(path, "rb") as f:
        try:
            # EXPLICIT FIX: Added the features array to force Key-Value extraction
            poller = client.begin_analyze_document(
                model_id,
                body=f,
                features=[DocumentAnalysisFeature.KEY_VALUE_PAIRS]
            )
            result = poller.result()
        except HttpResponseError as e:
            print(f"Azure Document Intelligence request failed for {image_path}: {e.message}")
            raise

    raw_dict = result.as_dict() if hasattr(result, "as_dict") else dict(result)
    _save_cached_result(image_path, model_id, raw_dict)
    return raw_dict


if __name__ == "__main__":
    # Quick manual test: run this file directly to sanity-check your
    # Azure credentials (or your cache) and see the output for one image.

    test_image = r"data/raw/training_data/images/{image_id}.png"  # <-- DEBUGGING: hardcode your path here
    print(f"Analyzing: {test_image}")

    raw_dict = analyze_document(test_image)

    pages = raw_dict.get("pages", [])
    total_lines = sum(len(page.get("lines", [])) for page in pages)
    print(f"\nExtracted {total_lines} lines across {len(pages)} page(s):\n")

    for page in pages:
        for line in page.get("lines", []):
            print(f"  [page {page.get('pageNumber')}] {line.get('content')}")

    print(f"\nCached at: {_cache_path(test_image, DEFAULT_MODEL_ID)}")