# Notes. digits dataset

## File
- data/digits-data.csv

## What the data represents
This dataset contains 8x8 handwritten digit images (64 features per sample) with a digit label for each sample.

## Columns and types
- digit: integer class label (0–9)
- features: 64 numeric pixel-intensity columns (flattened 8x8)

## Formatting plan
- Ensure digit is integer in [0, 9]
- Ensure all 64 feature columns are numeric
- Save a tidy/cleaned version for visualization

## Visualization plan (non-misleading)
Use a 2D embedding (t-SNE or PCA) where each point is one sample. Color by digit label. Axes will be labeled as embedding dimensions. No clustering is required.

## AI usage
AI helped structure the workflow and checks. I verified file locations, schema assumptions, and ensured the visualization uses correct labels and does not distort comparisons.
