"""
test_parser.py

Unit tests for AzureToFUNSDParser (src/parser.py).

These tests focus on the pure-logic pieces of the parser: geometry
conversion (polygon -> box), the overlap/containment heuristic used to
de-duplicate KVP elements vs. page lines, text formatting for checkboxes,
and the end-to-end translation of a small mock Azure response into FUNSD
format.

Run with:
    pytest tests/test_parser.py -v
"""

import pytest
from src.parser import AzureToFUNSDParser


@pytest.fixture
def parser():
    """A fresh parser instance for each test (resets id counter/state)."""
    return AzureToFUNSDParser()


# ---------------------------------------------------------------------------
# _convert_polygon_to_box
# ---------------------------------------------------------------------------

class TestConvertPolygonToBox:
    def test_simple_rectangle(self, parser):
        """An axis-aligned rectangle should map directly to [left, top, right, bottom]."""
        polygon = [10, 20, 110, 20, 110, 70, 10, 70]  # x1,y1, x2,y2, x3,y3, x4,y4
        assert parser._convert_polygon_to_box(polygon) == [10, 20, 110, 70]

    def test_rotated_polygon_uses_min_max_envelope(self, parser):
        """A slightly rotated/skewed quad should produce the bounding envelope,
        not just the first two points."""
        polygon = [12, 5, 100, 8, 98, 60, 10, 58]
        box = parser._convert_polygon_to_box(polygon)
        assert box == [10, 5, 100, 60]

    def test_coordinates_are_truncated_to_int(self, parser):
        """Azure returns floats; FUNSD boxes should be ints (floored, not rounded)."""
        polygon = [10.9, 20.9, 110.1, 20.1, 110.1, 70.9, 10.1, 70.1]
        box = parser._convert_polygon_to_box(polygon)
        assert all(isinstance(coord, int) for coord in box)
        assert box == [10, 20, 110, 70]

    def test_empty_polygon_returns_fallback(self, parser):
        assert parser._convert_polygon_to_box([]) == [0, 0, 0, 0]

    def test_none_polygon_returns_fallback(self, parser):
        assert parser._convert_polygon_to_box(None) == [0, 0, 0, 0]

    def test_too_few_points_returns_fallback(self, parser):
        """A polygon needs at least 4 (x, y) pairs (8 numbers)."""
        polygon = [10, 20, 110, 20, 110, 70]  # only 3 points
        assert parser._convert_polygon_to_box(polygon) == [0, 0, 0, 0]


# ---------------------------------------------------------------------------
# _boxes_overlap
# ---------------------------------------------------------------------------

class TestBoxesOverlap:
    def test_identical_boxes_overlap(self, parser):
        box = [10, 10, 110, 30]
        assert parser._boxes_overlap(box, box) is True

    def test_disjoint_boxes_do_not_overlap(self, parser):
        box_a = [0, 0, 50, 50]
        box_b = [200, 200, 250, 250]
        assert parser._boxes_overlap(box_a, box_b) is False

    def test_high_iou_jitter_counts_as_overlap(self, parser):
        """Sub-pixel jitter between a KVP bounding region and a page-line
        polygon for the same text should be treated as the same element."""
        box_a = [100, 100, 300, 120]
        box_b = [101, 99, 299, 121]  # ~1px shift on each edge
        assert parser._boxes_overlap(box_a, box_b, iou_threshold=0.5) is True

    def test_low_iou_below_threshold_no_overlap(self, parser):
        """Two boxes that touch but mostly don't intersect should not count."""
        box_a = [0, 0, 100, 100]
        box_b = [90, 90, 200, 200]  # small corner intersection only
        assert parser._boxes_overlap(box_a, box_b, iou_threshold=0.5) is False

    def test_multiline_value_contains_single_line(self, parser):
        """A multi-line KVP value box should 'contain' an individual line's
        box even though their IoU is low (different total areas)."""
        merged_value_box = [100, 50, 500, 150]   # spans two lines vertically
        single_line_box = [105, 55, 480, 95]      # fully inside the merged box
        assert parser._boxes_overlap(
            merged_value_box, single_line_box,
            iou_threshold=0.5, containment_threshold=0.8
        ) is True

    def test_partial_containment_below_threshold_no_overlap(self, parser):
        """A box only partially contained (below containment_threshold) and
        with low IoU should not be flagged as the same element."""
        box_a = [0, 0, 100, 100]
        box_b = [50, 50, 200, 200]  # only 25% of box_b's area is inside box_a
        assert parser._boxes_overlap(
            box_a, box_b, iou_threshold=0.5, containment_threshold=0.8
        ) is False

    def test_zero_area_box_does_not_overlap(self, parser):
        """Degenerate (zero-width/height) boxes should never report overlap,
        avoiding a division-by-zero in the IoU/containment math."""
        zero_box = [10, 10, 10, 50]  # zero width
        normal_box = [0, 0, 100, 100]
        assert parser._boxes_overlap(zero_box, normal_box) is False


# ---------------------------------------------------------------------------
# _format_text
# ---------------------------------------------------------------------------

class TestFormatText:
    def test_selected_checkbox_replaced_with_filled_box(self, parser):
        assert parser._format_text(":selected:") == "☑"

    def test_unselected_checkbox_replaced_with_empty_box(self, parser):
        assert parser._format_text(":unselected:") == "☐"

    def test_checkbox_marker_within_larger_string(self, parser):
        text = "Approved :selected: by manager"
        assert parser._format_text(text) == "Approved ☑ by manager"

    def test_plain_text_passes_through_unchanged(self, parser):
        assert parser._format_text("COMPOUND") == "COMPOUND"

    def test_empty_string_returns_empty_string(self, parser):
        assert parser._format_text("") == ""

    def test_none_returns_empty_string(self, parser):
        assert parser._format_text(None) == ""


# ---------------------------------------------------------------------------
# _generate_id
# ---------------------------------------------------------------------------

class TestGenerateId:
    def test_ids_are_sequential_starting_at_zero(self, parser):
        assert parser._generate_id() == 0
        assert parser._generate_id() == 1
        assert parser._generate_id() == 2

    def test_fresh_parser_resets_counter(self):
        p1 = AzureToFUNSDParser()
        p1._generate_id()
        p1._generate_id()

        p2 = AzureToFUNSDParser()
        assert p2._generate_id() == 0


# ---------------------------------------------------------------------------
# parse_azure_result (end-to-end with small mock payloads)
# ---------------------------------------------------------------------------

class TestParseAzureResult:
    def test_single_kvp_creates_linked_question_and_answer(self, parser):
        mock_response = {
            "key_value_pairs": [
                {
                    "key": {
                        "content": "COMPOUND",
                        "bounding_regions": [{"polygon": [84, 109, 136, 109, 136, 119, 84, 119]}],
                    },
                    "value": {
                        "content": "3-Hydroxy-3-methylbutanoic acid",
                        "bounding_regions": [{"polygon": [145, 98, 507, 98, 507, 116, 145, 116]}],
                    },
                }
            ]
        }

        result = parser.parse_azure_result(mock_response)
        form = result["form"]

        assert len(form) == 2

        question = next(el for el in form if el["label"] == "question")
        answer = next(el for el in form if el["label"] == "answer")

        assert question["text"] == "COMPOUND"
        assert answer["text"] == "3-Hydroxy-3-methylbutanoic acid"

        # Both elements should record the link between question <-> answer
        assert [question["id"], answer["id"]] in question["linking"]
        assert [question["id"], answer["id"]] in answer["linking"]

    def test_kvp_with_missing_key_still_creates_answer(self, parser):
        """If Azure returns a value with no matching key content, the answer
        should still be created (unlinked)."""
        mock_response = {
            "key_value_pairs": [
                {
                    "key": {"content": "", "bounding_regions": []},
                    "value": {
                        "content": "Standalone value",
                        "bounding_regions": [{"polygon": [0, 0, 100, 0, 100, 20, 0, 20]}],
                    },
                }
            ]
        }

        result = parser.parse_azure_result(mock_response)
        form = result["form"]

        assert len(form) == 1
        assert form[0]["label"] == "answer"
        assert form[0]["text"] == "Standalone value"
        assert form[0]["linking"] == []

    def test_unmapped_line_near_top_is_labeled_header(self, parser):
        mock_response = {
            "key_value_pairs": [],
            "pages": [
                {
                    "lines": [
                        {
                            "content": "BIOLOGICAL SPECIMEN RECORD",
                            "polygon": [10, 5, 300, 5, 300, 25, 10, 25],  # y < 200
                            "words": [],
                        }
                    ]
                }
            ],
        }

        result = parser.parse_azure_result(mock_response)
        form = result["form"]

        assert len(form) == 1
        assert form[0]["label"] == "header"
        assert form[0]["text"] == "BIOLOGICAL SPECIMEN RECORD"

    def test_unmapped_line_below_threshold_is_labeled_other(self, parser):
        mock_response = {
            "key_value_pairs": [],
            "pages": [
                {
                    "lines": [
                        {
                            "content": "Some body text",
                            "polygon": [10, 300, 300, 300, 300, 320, 10, 320],  # y >= 200
                            "words": [],
                        }
                    ]
                }
            ],
        }

        result = parser.parse_azure_result(mock_response)
        form = result["form"]

        assert len(form) == 1
        assert form[0]["label"] == "other"

    def test_line_overlapping_kvp_box_is_not_duplicated(self, parser):
        """A page line covering the same physical text as a KVP element
        should be skipped (not re-added as 'header'/'other')."""
        mock_response = {
            "key_value_pairs": [
                {
                    "key": {
                        "content": "COMPOUND",
                        "bounding_regions": [{"polygon": [84, 109, 136, 109, 136, 119, 84, 119]}],
                    },
                    "value": {},
                }
            ],
            "pages": [
                {
                    "lines": [
                        {
                            # Same region as the KVP key above (sub-pixel jitter)
                            "content": "COMPOUND",
                            "polygon": [84, 109.5, 136.2, 109, 136, 119, 84, 119.4],
                            "words": [],
                        }
                    ]
                }
            ],
        }

        result = parser.parse_azure_result(mock_response)
        form = result["form"]

        # Only the KVP "question" element should exist; the duplicate line is skipped.
        assert len(form) == 1
        assert form[0]["label"] == "question"

    def test_word_level_breakdown_is_mapped(self, parser):
        mock_response = {
            "key_value_pairs": [],
            "pages": [
                {
                    "lines": [
                        {
                            "content": "Hello World",
                            "polygon": [0, 0, 100, 0, 100, 20, 0, 20],
                            "words": [
                                {"content": "Hello", "polygon": [0, 0, 40, 0, 40, 20, 0, 20]},
                                {"content": "World", "polygon": [45, 0, 100, 0, 100, 20, 45, 20]},
                            ],
                        }
                    ]
                }
            ],
        }

        result = parser.parse_azure_result(mock_response)
        line_element = result["form"][0]

        assert len(line_element["words"]) == 2
        assert line_element["words"][0]["text"] == "Hello"
        assert line_element["words"][0]["box"] == [0, 0, 40, 20]
        assert line_element["words"][1]["text"] == "World"

    def test_supports_camelcase_keys(self, parser):
        """Azure's REST API can return camelCase keys (keyValuePairs,
        boundingRegions) instead of the snake_case used by the Python SDK's
        .as_dict(). The parser should handle both."""
        mock_response = {
            "keyValuePairs": [
                {
                    "key": {
                        "content": "DATE",
                        "boundingRegions": [{"polygon": [0, 0, 50, 0, 50, 10, 0, 10]}],
                    },
                    "value": {
                        "content": "2024-01-01",
                        "boundingRegions": [{"polygon": [60, 0, 150, 0, 150, 10, 60, 10]}],
                    },
                }
            ]
        }

        result = parser.parse_azure_result(mock_response)
        form = result["form"]

        assert len(form) == 2
        labels = {el["label"] for el in form}
        assert labels == {"question", "answer"}

    def test_empty_response_returns_empty_form(self, parser):
        result = parser.parse_azure_result({})
        assert result == {"form": []}

    def test_reparsing_resets_state(self, parser):
        """Calling parse_azure_result a second time should not accumulate
        elements/ids from the previous call."""
        mock_response = {
            "key_value_pairs": [
                {
                    "key": {"content": "A", "bounding_regions": [{"polygon": [0, 0, 10, 0, 10, 10, 0, 10]}]},
                    "value": {"content": "B", "bounding_regions": [{"polygon": [20, 0, 30, 0, 30, 10, 20, 10]}]},
                }
            ]
        }

        first = parser.parse_azure_result(mock_response)
        second = parser.parse_azure_result(mock_response)

        assert len(first["form"]) == len(second["form"]) == 2
        # IDs should restart from 0 on the second call
        assert {el["id"] for el in second["form"]} == {0, 1}
