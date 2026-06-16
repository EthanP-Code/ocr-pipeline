"""
generate_example_output.py

Generates a three-panel comparison image for the README:
  Panel 1 — original scanned document image (no annotations)
  Panel 2 — Azure's parsed output overlaid as colored bounding boxes
  Panel 3 — FUNSD ground-truth annotation overlaid as colored bounding boxes

Run once from the project root after pipeline.py has generated predictions:

    python generate_example_output.py

Output is saved to docs/example_output.png.

The color scheme:
    question  ->  blue
    answer    ->  green
    header    ->  orange
    other     ->  gray
"""

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ============================================================
# Configuration — edit IMAGE_STEM to pick a different example
# ============================================================

# Change this to any image stem that exists in both predictions/ and annotations/.
# Run `ls data/processed/predictions/` to see what's available after pipeline.py.
IMAGE_STEM = "0013255595"

IMAGE_PATH = Path(f"data/raw/training_data/images/{IMAGE_STEM}.png")
PRED_PATH  = Path(f"data/processed/predictions/{IMAGE_STEM}.json")
TRUTH_PATH = Path(f"data/raw/training_data/annotations/{IMAGE_STEM}.json")
OUTPUT_PATH = Path("docs/example_output.png")

LABEL_COLORS = {
    "question": "#2196F3",  # blue
    "answer":   "#4CAF50",  # green
    "header":   "#FF9800",  # orange
    "other":    "#9E9E9E",  # gray
}


def load_image_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_funsd(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("form", [])


def draw_boxes(ax, image: np.ndarray, elements: list[dict], title: str):
    """Draw FUNSD-format [left, top, right, bottom] boxes over an image on ax."""
    ax.imshow(image)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.axis("off")

    for el in elements:
        box = el.get("box", [])
        if len(box) != 4:
            continue

        left, top, right, bottom = box
        width  = right - left
        height = bottom - top
        label  = el.get("label", "other")
        color  = LABEL_COLORS.get(label, "#9E9E9E")

        rect = mpatches.FancyBboxPatch(
            (left, top), width, height,
            boxstyle="square,pad=0",
            linewidth=1.5,
            edgecolor=color,
            facecolor=color,
            alpha=0.15,
        )
        ax.add_patch(rect)

        # Draw edge border at full opacity on top
        border = mpatches.FancyBboxPatch(
            (left, top), width, height,
            boxstyle="square,pad=0",
            linewidth=1.5,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(border)


def build_legend():
    return [
        mpatches.Patch(facecolor=color, edgecolor=color, label=label.capitalize())
        for label, color in LABEL_COLORS.items()
    ]


def main():
    # Validate files exist before doing any work
    for path in (IMAGE_PATH, PRED_PATH, TRUTH_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Make sure pipeline.py has been run and predictions exist."
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    image      = load_image_rgb(IMAGE_PATH)
    pred_elems = load_funsd(PRED_PATH)
    truth_elems = load_funsd(TRUTH_PATH)

    print(f"Image:       {IMAGE_PATH}  ({image.shape[1]}x{image.shape[0]} px)")
    print(f"Predicted:   {len(pred_elems)} elements")
    print(f"Ground truth:{len(truth_elems)} elements")

    fig, axes = plt.subplots(
        1, 3,
        figsize=(22, 10),
        gridspec_kw={"wspace": 0.04}
    )

    # Panel 1: raw image, no boxes
    axes[0].imshow(image)
    axes[0].set_title("Original document", fontsize=13, fontweight="bold", pad=10)
    axes[0].axis("off")

    # Panel 2: predictions
    draw_boxes(axes[1], image, pred_elems,
               f"Azure DI output (parsed)\n{len(pred_elems)} elements")

    # Panel 3: ground truth
    draw_boxes(axes[2], image, truth_elems,
               f"FUNSD ground truth\n{len(truth_elems)} elements")

    # Shared legend below all panels
    legend = build_legend()
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=4,
        fontsize=11,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        f"Pipeline output comparison — {IMAGE_STEM}.png",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )

    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()