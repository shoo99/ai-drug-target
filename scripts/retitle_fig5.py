#!/usr/bin/env python3
"""Retitle fig5 heatmap: 'Top 20 MR genes ...' -> 'Top 20 TWAS genes ...'.

The fig5 heatmap was generated on the KBRI cluster and its per-tissue matrix is
not available locally, so the plot itself cannot be faithfully re-rendered. The
only defect is the baked-in title word 'MR', which contradicts the paper's
'S-PrediXcan is a TWAS, not MR' statement. This script masks the title band and
writes the corrected title — changing ONLY the label text, not any data.
"""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

FIG = Path("/home/sysoft/ai-drug-target/paper4/figures/fig5_tissue_dot.png")
FONT = "/home/sysoft/ai-drug-target/venv/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"
NEW_TITLE = "Top 20 TWAS genes × 13 brain tissues — S-PrediXcan significance"

img = Image.open(FIG).convert("RGB")
W, H = img.size
draw = ImageDraw.Draw(img)

# The matplotlib suptitle/title sits in the top band of the canvas. Mask a
# generous band across the full width and rewrite, centered, in the same dark
# grey matplotlib uses for titles.
band_h = int(H * 0.055)
draw.rectangle([0, 0, W, band_h + 6], fill="white")

# size the font to match the original (~ title fontsize). Try to fit width.
for fs in range(34, 14, -1):
    font = ImageFont.truetype(FONT, fs)
    bbox = draw.textbbox((0, 0), NEW_TITLE, font=font)
    tw = bbox[2] - bbox[0]
    if tw <= W * 0.92:
        break
th = bbox[3] - bbox[1]
x = (W - tw) // 2
y = max(2, (band_h - th) // 2)
draw.text((x, y), NEW_TITLE, fill=(40, 40, 40), font=font)

img.save(FIG)
print(f"Retitled {FIG.name}: '{NEW_TITLE}'  (font {fs}pt)")
