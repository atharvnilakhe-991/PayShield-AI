# Phase 3 family templates

`layout_families.json` describes the four supported dark Google Pay-style families and their components. The renderer in `scripts/phase3/renderer.py` draws these families with shared palette, typography hierarchy, and synthetic markers. Layout-specific card structures remain explicit in the renderer so each edited field can return its own maskable pixel zones.

All synthetic rendering is fictional and Pillow-based. This keeps the clean/edited pair on one deterministic drawing path and lets the pixel-difference mask and bounding boxes use the same coordinates. No logos or transaction values are copied from the real dataset.
