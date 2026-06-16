"""
evaluator.py

Calculates the accuracy of the OCR pipeline by comparing the parsed Azure 
output against the FUNSD ground-truth JSON annotations.
Uses industry-standard Character Error Rate (CER) and Word Error Rate (WER).
"""

import json
import os
from pathlib import Path
import jiwer  
import re

class OCREvaluator:
    def __init__(self):
        # We use a y-tolerance to group words on the same horizontal line.
        # This prevents slight bounding-box pixel variations from breaking the reading order.
        self.y_tolerance = 15 

    def _sort_reading_order(self, elements):
        """
        Sorts bounding boxes top-to-bottom, then left-to-right.
        Uses a binning heuristic on the Y-axis to group elements on the same line.
        """
        # elements box format: [left, top, right, bottom]
        return sorted(elements, key=lambda e: (e['box'][1] // self.y_tolerance, e['box'][0]))

    def _normalize_text(self, text):
        """Strips punctuation, lowercases, and removes extra spaces for fair evaluation."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip() # Collapse extra spaces
        return text

    def _extract_full_text(self, funsd_json):
        """
        Extracts and concatenates all text from a FUNSD formatted dictionary 
        into a single string, following standard human reading order.
        """
        if 'form' not in funsd_json:
            return ""
        
        sorted_elements = self._sort_reading_order(funsd_json['form'])
        
        # Extract text, ignore empty strings, and join with spaces
        extracted_words = [el.get('text', '').strip() for el in sorted_elements if el.get('text')]
        return " ".join(extracted_words)

    def evaluate_single_document(self, truth_path, pred_path):
        """
        Compares a single ground-truth JSON file against a predicted JSON file.
        Returns the CER and WER.
        """
        try:
            with open(truth_path, 'r', encoding='utf-8') as f:
                truth_data = json.load(f)
            with open(pred_path, 'r', encoding='utf-8') as f:
                pred_data = json.load(f)
        except Exception as e:
            print(f"Error loading files: {e}")
            return None

        truth_text = self._extract_full_text(truth_data)
        pred_text = self._extract_full_text(pred_data)

        # Edge case: Empty document
        if not truth_text and not pred_text:
            return {"cer": 0.0, "wer": 0.0}
        if not truth_text or not pred_text:
            return {"cer": 1.0, "wer": 1.0}

        # Apply normalization here!
        truth_text = self._normalize_text(truth_text)
        pred_text = self._normalize_text(pred_text)

        # Compute metrics using jiwer
        # Lower is better (0.0 = perfect match, 1.0 = completely wrong)
        cer = jiwer.cer(truth_text, pred_text)
        wer = jiwer.wer(truth_text, pred_text)

        return {
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "truth_sample": truth_text[:50] + "...", # Preview for debugging
            "pred_sample": pred_text[:50] + "..."
        }

    def evaluate_batch(self, truth_dir, pred_dir):
        """
        Iterates through a directory of predicted JSONs, finds their matching 
        ground-truth files, and calculates average metrics for the pipeline.
        """
        pred_files = list(Path(pred_dir).glob("*.json"))
        
        if not pred_files:
            print(f"No JSON files found in {pred_dir}")
            return
            
        total_cer, total_wer, count = 0, 0, 0

        print(f"Evaluating {len(pred_files)} documents...")
        print("-" * 50)

        for pred_file in pred_files:
            # Assume ground truth file has the exact same name
            truth_file = Path(truth_dir) / pred_file.name
            
            if not truth_file.exists():
                print(f"Warning: Ground truth not found for {pred_file.name}. Skipping.")
                continue
                
            metrics = self.evaluate_single_document(truth_file, pred_file)
            if metrics:
                print(f"File: {pred_file.name} | CER: {metrics['cer']:.2%} | WER: {metrics['wer']:.2%}")
                total_cer += metrics['cer']
                total_wer += metrics['wer']
                count += 1

        print("-" * 50)
        if count > 0:
            avg_cer = total_cer / count
            avg_wer = total_wer / count
            print(f"PIPELINE AVERAGE CER: {avg_cer:.2%}")
            print(f"PIPELINE AVERAGE WER: {avg_wer:.2%}")
            return {"avg_cer": avg_cer, "avg_wer": avg_wer}
        else:
            print("No valid files evaluated.")