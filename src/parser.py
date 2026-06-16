"""
parser.py

Translates raw Azure Document Intelligence output into the standardized 
FUNSD ground-truth JSON format. 
"""

import json

class AzureToFUNSDParser:
    def __init__(self):
        self.form_elements = []
        self.current_id = 0

    def _generate_id(self):
        """Generates a sequential unique ID for FUNSD linking."""
        this_id = self.current_id
        self.current_id += 1
        return this_id

    def _convert_polygon_to_box(self, polygon):
        """
        Converts Azure's 8-point polygon [x1, y1, x2, y2, x3, y3, x4, y4]
        to FUNSD's 4-point bounding box [left, top, right, bottom].
        """
        if not polygon or len(polygon) < 8:
            return [0, 0, 0, 0] # Fallback for missing coordinate data
            
        x_coords = [polygon[i] for i in range(0, len(polygon), 2)]
        y_coords = [polygon[i] for i in range(1, len(polygon), 2)]
        
        return [
            int(min(x_coords)), 
            int(min(y_coords)), 
            int(max(x_coords)), 
            int(max(y_coords))
        ]

    def _boxes_overlap(self, box_a, box_b, iou_threshold=0.5, containment_threshold=0.8):
        """
        Returns True if two [left, top, right, bottom] boxes refer to the
        same physical text region. Two cases are checked:

        1. IoU overlap above `iou_threshold` — handles the common case of
           sub-pixel float jitter between Azure's KVP bounding_regions and
           page-line polygons for the same single-line text.

        2. Containment above `containment_threshold` — handles multi-line
           KVP values. Azure merges multi-line answers into ONE KVP value
           box, but reports each line separately in `pages[].lines`. A
           single line's box will have low IoU against the larger merged
           box (different areas) but is almost entirely *contained* within
           it, so we check what fraction of the smaller box's area falls
           inside the larger box.
        """
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        if inter_area == 0:
            return False

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        if area_a == 0 or area_b == 0:
            return False

        # Case 1: standard IoU overlap (single-line jitter)
        union_area = area_a + area_b - inter_area
        iou = inter_area / union_area
        if iou >= iou_threshold:
            return True

        # Case 2: containment (multi-line KVP value vs. individual line)
        smaller_area = min(area_a, area_b)
        containment_ratio = inter_area / smaller_area
        if containment_ratio >= containment_threshold:
            return True

        return False

    def _format_text(self, text):
        """Handles edge cases like Azure's checkbox formatting."""
        if not text:
            return ""
        text = text.replace(":selected:", "☑")
        text = text.replace(":unselected:", "☐")
        return text

    def _create_element(self, text, label, polygon, words=None):
        """Builds a single FUNSD-compliant dictionary element."""
        if words is None:
            words = []
            
        element = {
            "box": self._convert_polygon_to_box(polygon),
            "text": self._format_text(text),
            "label": label,
            "words": [],
            "linking": [],
            "id": self._generate_id()
        }

        # If Azure provided word-level breakdowns, map them as well
        for w in words:
            element["words"].append({
                "box": self._convert_polygon_to_box(w.get("polygon", [])),
                "text": self._format_text(w.get("content", ""))
            })
            
        return element

    def parse_azure_result(self, azure_data):
        """
        Main translation engine. 
        Accepts the raw Azure dictionary (prebuilt-document model) and 
        returns a FUNSD-formatted dictionary.
        """
        self.form_elements = [] # Reset state
        self.current_id = 0
        
        # Track boxes of already-mapped Q/A elements so we don't duplicate
        # them as "header"/"other" later. Stored as converted [l, t, r, b]
        # boxes (not raw polygons) since Azure's KVP bounding_regions and
        # page-line polygons for the *same* text differ by sub-pixel float
        # jitter and will never match via exact string/polygon comparison.
        mapped_boxes = []

        # 1. Parse Key-Value Pairs (Questions and Answers)
        kvps = azure_data.get("key_value_pairs") or azure_data.get("keyValuePairs") or []
        for kvp in kvps:
            question_id = None
            answer_id = None
            
            # Helper to safely handle camelCase vs snake_case regions
            def get_poly(node):
                regions = node.get("bounding_regions") or node.get("boundingRegions")
                if regions and len(regions) > 0:
                    return regions[0].get("polygon", [])
                return []

            # Process Key (Question)
            key_node = kvp.get("key", {})
            if key_node and key_node.get("content"):
                poly = get_poly(key_node)
                q_element = self._create_element(
                    text=key_node.get("content", ""),
                    label="question",
                    polygon=poly
                )
                question_id = q_element["id"]
                mapped_boxes.append(q_element["box"])
                self.form_elements.append(q_element)

            # Process Value (Answer)
            value_node = kvp.get("value", {})
            if value_node and value_node.get("content"):
                poly = get_poly(value_node)
                a_element = self._create_element(
                    text=value_node.get("content", ""),
                    label="answer",
                    polygon=poly
                )
                answer_id = a_element["id"]
                mapped_boxes.append(a_element["box"])
                self.form_elements.append(a_element)

            # Link them together if both exist
            if question_id is not None and answer_id is not None:
                link_pair = [question_id, answer_id]
                # Find elements in list and update linking
                for element in self.form_elements:
                    if element["id"] in (question_id, answer_id):
                        element["linking"].append(link_pair)

        # 2. Parse Unmapped Text (Headers / Other)
        # Assuming azure_data contains "pages" with "lines" for block text
        pages = azure_data.get("pages", [])
        for page in pages:
            for line in page.get("lines", []):
                poly = line.get("polygon", [])
                line_box = self._convert_polygon_to_box(poly)

                # Skip this line if it spatially overlaps a box we already
                # emitted as a question/answer (i.e. it's the same text,
                # just seen again via the page's line list).
                already_mapped = any(
                    self._boxes_overlap(line_box, mapped_box)
                    for mapped_box in mapped_boxes
                )

                if not already_mapped:
                    # Basic heuristic: if it's at the very top of the page, call it a header
                    y_coords = [poly[i] for i in range(1, len(poly), 2)] if len(poly) >= 8 else [0]
                    is_header = min(y_coords) < 200  # Adjust threshold based on image height
                    
                    label = "header" if is_header else "other"
                    
                    unmapped_element = self._create_element(
                        text=line.get("content", ""),
                        label=label,
                        polygon=poly,
                        words=line.get("words", [])
                    )
                    mapped_boxes.append(unmapped_element["box"])
                    self.form_elements.append(unmapped_element)

        return {"form": self.form_elements}

    def export_to_json(self, output_path):
        """Saves the fully parsed FUNSD data to a local JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({"form": self.form_elements}, f, indent=4, ensure_ascii=False)


# ==========================================
# Testing block (Only runs if script is executed directly)
# ==========================================
if __name__ == "__main__":
    # Simulated mock data to test the logic
    mock_azure_response = {
        "key_value_pairs": [
            {
                "key": {"content": "COMPOUND", "bounding_regions": [{"polygon": [84, 109, 136, 109, 136, 119, 84, 119]}]},
                "value": {"content": "3-Hydroxy-3-methylbutanoic acid", "bounding_regions": [{"polygon": [145, 98, 507, 98, 507, 116, 145, 116]}]}
            }
        ]
    }
    
    parser = AzureToFUNSDParser()
    parsed_output = parser.parse_azure_result(mock_azure_response)
    print(json.dumps(parsed_output, indent=2))