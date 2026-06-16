"""
test_evaluator.py

Unit tests for OCREvaluator (src/evaluator.py).

These tests cover the reading-order sort heuristic, text normalization,
full-text extraction from FUNSD-formatted dicts, and the CER/WER scoring
(including edge cases like empty documents).

Run with:
    pytest tests/test_evaluator.py -v
"""

import json

import pytest

from src.evaluator import OCREvaluator


@pytest.fixture
def evaluator():
    return OCREvaluator()


# ---------------------------------------------------------------------------
# _sort_reading_order
# ---------------------------------------------------------------------------

class TestSortReadingOrder:
    def test_sorts_top_to_bottom(self, evaluator):
        """Elements on clearly different lines should be ordered by their
        top (y) coordinate, ignoring their order in the input list."""
        elements = [
            {"text": "second line", "box": [0, 100, 50, 120]},
            {"text": "first line", "box": [0, 0, 50, 20]},
        ]
        ordered = evaluator._sort_reading_order(elements)
        assert [el["text"] for el in ordered] == ["first line", "second line"]

    def test_sorts_left_to_right_within_same_line(self, evaluator):
        """Elements whose y-coordinates fall in the same bin (within
        y_tolerance) should be ordered left-to-right by x."""
        elements = [
            {"text": "right", "box": [100, 0, 150, 20]},
            {"text": "left", "box": [0, 0, 50, 20]},
        ]
        ordered = evaluator._sort_reading_order(elements)
        assert [el["text"] for el in ordered] == ["left", "right"]

    def test_small_y_jitter_does_not_break_reading_order(self, evaluator):
        """Two elements meant to be on the same line, but with slightly
        different top values (within y_tolerance=15), should still sort
        left-to-right rather than by their tiny y difference."""
        elements = [
            {"text": "right", "box": [100, 5, 150, 25]},
            {"text": "left", "box": [0, 0, 50, 20]},
        ]
        ordered = evaluator._sort_reading_order(elements)
        assert [el["text"] for el in ordered] == ["left", "right"]

    def test_large_y_difference_creates_new_line(self, evaluator):
        """A y difference greater than y_tolerance should place the element
        on a later line, regardless of x position."""
        elements = [
            {"text": "second line, far left", "box": [0, 200, 50, 220]},
            {"text": "first line, far right", "box": [500, 0, 550, 20]},
        ]
        ordered = evaluator._sort_reading_order(elements)
        assert [el["text"] for el in ordered] == ["first line, far right", "second line, far left"]


# ---------------------------------------------------------------------------
# _normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_lowercases_text(self, evaluator):
        assert evaluator._normalize_text("HELLO World") == "hello world"

    def test_removes_punctuation(self, evaluator):
        assert evaluator._normalize_text("3-Hydroxy-3-methylbutanoic acid.") == "3hydroxy3methylbutanoic acid"

    def test_collapses_extra_whitespace(self, evaluator):
        assert evaluator._normalize_text("hello    world\t\n") == "hello world"

    def test_strips_leading_and_trailing_whitespace(self, evaluator):
        assert evaluator._normalize_text("  hello  ") == "hello"

    def test_empty_string_stays_empty(self, evaluator):
        assert evaluator._normalize_text("") == ""


# ---------------------------------------------------------------------------
# _extract_full_text
# ---------------------------------------------------------------------------

class TestExtractFullText:
    def test_missing_form_key_returns_empty_string(self, evaluator):
        assert evaluator._extract_full_text({}) == ""

    def test_extracts_and_joins_text_in_reading_order(self, evaluator):
        funsd = {
            "form": [
                {"text": "World", "box": [100, 0, 150, 20]},
                {"text": "Hello", "box": [0, 0, 50, 20]},
            ]
        }
        assert evaluator._extract_full_text(funsd) == "Hello World"

    def test_skips_elements_with_empty_text(self, evaluator):
        funsd = {
            "form": [
                {"text": "Hello", "box": [0, 0, 50, 20]},
                {"text": "", "box": [60, 0, 70, 20]},
                {"text": "World", "box": [100, 0, 150, 20]},
            ]
        }
        assert evaluator._extract_full_text(funsd) == "Hello World"

    def test_strips_whitespace_from_each_element(self, evaluator):
        funsd = {
            "form": [
                {"text": "  Hello  ", "box": [0, 0, 50, 20]},
                {"text": "World", "box": [100, 0, 150, 20]},
            ]
        }
        assert evaluator._extract_full_text(funsd) == "Hello World"


# ---------------------------------------------------------------------------
# evaluate_single_document
# ---------------------------------------------------------------------------

class TestEvaluateSingleDocument:
    def _write_funsd(self, tmp_path, filename, form_elements):
        path = tmp_path / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"form": form_elements}, f)
        return path

    def test_perfect_match_gives_zero_error(self, evaluator, tmp_path):
        elements = [{"text": "Hello World", "box": [0, 0, 100, 20]}]
        truth_path = self._write_funsd(tmp_path, "truth.json", elements)
        pred_path = self._write_funsd(tmp_path, "pred.json", elements)

        metrics = evaluator.evaluate_single_document(truth_path, pred_path)

        assert metrics["cer"] == 0.0
        assert metrics["wer"] == 0.0

    def test_both_empty_documents_returns_zero_error(self, evaluator, tmp_path):
        truth_path = self._write_funsd(tmp_path, "truth.json", [])
        pred_path = self._write_funsd(tmp_path, "pred.json", [])

        metrics = evaluator.evaluate_single_document(truth_path, pred_path)

        assert metrics == {"cer": 0.0, "wer": 0.0}

    def test_empty_prediction_with_nonempty_truth_returns_max_error(self, evaluator, tmp_path):
        truth_path = self._write_funsd(
            tmp_path, "truth.json", [{"text": "Hello World", "box": [0, 0, 100, 20]}]
        )
        pred_path = self._write_funsd(tmp_path, "pred.json", [])

        metrics = evaluator.evaluate_single_document(truth_path, pred_path)

        assert metrics == {"cer": 1.0, "wer": 1.0}

    def test_empty_truth_with_nonempty_prediction_returns_max_error(self, evaluator, tmp_path):
        truth_path = self._write_funsd(tmp_path, "truth.json", [])
        pred_path = self._write_funsd(
            tmp_path, "pred.json", [{"text": "Hello World", "box": [0, 0, 100, 20]}]
        )

        metrics = evaluator.evaluate_single_document(truth_path, pred_path)

        assert metrics == {"cer": 1.0, "wer": 1.0}

    def test_partial_mismatch_gives_nonzero_but_bounded_error(self, evaluator, tmp_path):
        truth_path = self._write_funsd(
            tmp_path, "truth.json", [{"text": "Hello World", "box": [0, 0, 100, 20]}]
        )
        pred_path = self._write_funsd(
            tmp_path, "pred.json", [{"text": "Hello Wrold", "box": [0, 0, 100, 20]}]
        )

        metrics = evaluator.evaluate_single_document(truth_path, pred_path)

        assert 0.0 < metrics["cer"] < 1.0
        assert 0.0 < metrics["wer"] <= 1.0

    def test_punctuation_differences_are_ignored(self, evaluator, tmp_path):
        """Normalization should make 'Hello, World!' and 'hello world'
        evaluate as a perfect match."""
        truth_path = self._write_funsd(
            tmp_path, "truth.json", [{"text": "Hello, World!", "box": [0, 0, 100, 20]}]
        )
        pred_path = self._write_funsd(
            tmp_path, "pred.json", [{"text": "hello world", "box": [0, 0, 100, 20]}]
        )

        metrics = evaluator.evaluate_single_document(truth_path, pred_path)

        assert metrics["cer"] == 0.0
        assert metrics["wer"] == 0.0

    def test_missing_file_returns_none(self, evaluator, tmp_path):
        truth_path = tmp_path / "does_not_exist.json"
        pred_path = self._write_funsd(tmp_path, "pred.json", [])

        assert evaluator.evaluate_single_document(truth_path, pred_path) is None


# ---------------------------------------------------------------------------
# evaluate_batch
# ---------------------------------------------------------------------------

class TestEvaluateBatch:
    def _write_funsd(self, directory, filename, form_elements):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"form": form_elements}, f)
        return path

    def test_averages_across_matching_files(self, evaluator, tmp_path, capsys):
        truth_dir = tmp_path / "truth"
        pred_dir = tmp_path / "pred"

        # Doc 1: perfect match (cer=0, wer=0)
        self._write_funsd(truth_dir, "doc1.json", [{"text": "Hello", "box": [0, 0, 50, 20]}])
        self._write_funsd(pred_dir, "doc1.json", [{"text": "Hello", "box": [0, 0, 50, 20]}])

        # Doc 2: completely wrong (cer=1, wer=1)
        self._write_funsd(truth_dir, "doc2.json", [{"text": "Hello", "box": [0, 0, 50, 20]}])
        self._write_funsd(pred_dir, "doc2.json", [])

        result = evaluator.evaluate_batch(truth_dir, pred_dir)

        assert result["avg_cer"] == pytest.approx(0.5)
        assert result["avg_wer"] == pytest.approx(0.5)

    def test_skips_predictions_missing_ground_truth(self, evaluator, tmp_path, capsys):
        truth_dir = tmp_path / "truth"
        pred_dir = tmp_path / "pred"

        self._write_funsd(pred_dir, "no_truth.json", [{"text": "Hello", "box": [0, 0, 50, 20]}])
        truth_dir.mkdir(parents=True, exist_ok=True)  # exists but empty

        result = evaluator.evaluate_batch(truth_dir, pred_dir)

        captured = capsys.readouterr()
        assert "Warning: Ground truth not found" in captured.out
        assert result is None  # no valid files evaluated

    def test_empty_pred_dir_returns_none(self, evaluator, tmp_path):
        truth_dir = tmp_path / "truth"
        pred_dir = tmp_path / "pred"
        truth_dir.mkdir()
        pred_dir.mkdir()

        assert evaluator.evaluate_batch(truth_dir, pred_dir) is None
