"""
pipeline.py

The main orchestrator script for the OCR Document Intelligence pipeline.
It selects a seeded random sample of images, logs them for auditability,
processes them through Azure, parses the output to FUNSD format, and 
evaluates the final accuracy.
"""

import os
import json
import random
from pathlib import Path

# Import our custom microservices
import src.azure_client as azure_client
from src.parser import AzureToFUNSDParser
from src.evaluator import OCREvaluator

# DB specific imports
from datetime import datetime
from src.database import init_db, insert_run, insert_document, insert_elements, DB_PATH

# ==========================================
# Configuration & Paths
# ==========================================
IMAGE_DIR = Path("data/raw/training_data/images")
ANNOTATION_DIR = Path("data/raw/training_data/annotations")
OUTPUT_DIR = Path("data/processed/predictions")
AUDIT_FILE = Path("data/processed/audit_log.json")

# Pipeline Parameters
SAMPLE_SIZE = 20
RANDOM_SEED = 42  # Change this to grab a different batch of images

# If True, bypass the Azure response cache and re-call the API for every
# image, overwriting any cached results. Leave False during normal
# development to avoid unnecessary API calls.
FORCE_REFRESH = False

def ensure_directories():
    """Creates necessary output directories if they don't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)

def select_images(image_dir, num_images, seed):
    """Randomly selects a subset of images using a seed for reproducibility."""
    all_images = list(image_dir.glob("*.png"))
    
    if len(all_images) < num_images:
        print(f"Warning: Only found {len(all_images)} images. Processing all of them.")
        num_images = len(all_images)
        
    random.seed(seed)
    selected_images = random.sample(all_images, num_images)
    return selected_images

def run_pipeline():
    ensure_directories()
    
    # Initialize database for this run
    con = init_db()
    run_id = insert_run(con, RANDOM_SEED, datetime.now().isoformat())
    
    # ---------------------------------------------------------
    # 1. Image Selection & Audit Logging
    # ---------------------------------------------------------
    print(f"--- Pipeline Initialization ---")
    print(f"Seed: {RANDOM_SEED} | Target Sample Size: {SAMPLE_SIZE}")
    
    selected_images = select_images(IMAGE_DIR, SAMPLE_SIZE, RANDOM_SEED)
    
    # Create the audit log JSON
    audit_data = {
        "seed": RANDOM_SEED,
        "sample_size": len(selected_images),
        "selected_images": [img.name for img in selected_images],
        "image_paths": [str(img) for img in selected_images]
    }
    
    with open(AUDIT_FILE, 'w') as f:
        json.dump(audit_data, f, indent=4)
    print(f"Audit log saved to: {AUDIT_FILE}")
    
    # Initialize our microservices
    parser = AzureToFUNSDParser()
    evaluator = OCREvaluator()
    
    # ---------------------------------------------------------
    # 2. Data Processing Loop
    # ---------------------------------------------------------
    print("\n--- Starting Data Processing ---")
    for idx, img_path in enumerate(selected_images, 1):
        print(f"[{idx}/{len(selected_images)}] Processing {img_path.name}...")
        
        try:
            # Step A: Send to Azure (or pull from cache if already analyzed)
            raw_azure_dict = azure_client.analyze_document(
                str(img_path), force_refresh=FORCE_REFRESH
            )

            # Step B: Parse the raw output to FUNSD standard
            parser.parse_azure_result(raw_azure_dict)
            
            # Step C: Save prediction to disk
            output_file = OUTPUT_DIR / f"{img_path.stem}.json"
            parser.export_to_json(output_file)

            # Step D: Store results in SQLite using evaluator's own scoring logic
            annotation_path = ANNOTATION_DIR / f"{img_path.stem}.json"
            metrics = evaluator.evaluate_single_document(annotation_path, output_file)

            with open(output_file, encoding='utf-8') as f:
                pred_data = json.load(f)

            doc_id = insert_document(con, run_id, img_path.name, metrics["cer"], metrics["wer"])
            insert_elements(con, doc_id, pred_data)
            
        except Exception as e:
            import traceback
            print(f"  -> Error processing {img_path.name}: {e}")
            traceback.print_exc()

    # ---------------------------------------------------------
    # 3. Final Pipeline Evaluation
    # ---------------------------------------------------------
    print("\n--- Starting Pipeline Evaluation ---")
    
    # Because we cleared the predictions folder (or are overwriting), 
    # we can run the batch evaluator to grade the files we just created.
    evaluator.evaluate_batch(
        truth_dir=ANNOTATION_DIR,
        pred_dir=OUTPUT_DIR
    )

    con.close()
    print(f"\nResults saved to: {DB_PATH}")

if __name__ == "__main__":
    # Optional: Clear out old predictions before running to ensure a clean evaluation
    if OUTPUT_DIR.exists():
        for file in OUTPUT_DIR.glob("*.json"):
            file.unlink()
            
    run_pipeline()