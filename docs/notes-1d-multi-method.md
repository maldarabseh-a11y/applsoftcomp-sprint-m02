# Notes. 1D multi method dataset

## Dataset structure

**File**. `data/1d-multi-method-data.csv`

This dataset contains AUC ROC measurements for multiple methods evaluated on the same number of samples.

- `method`. Categorical label for the method. One of `Proposed` or `Baseline_1` through `Baseline_9`.
- `AUCROC`. AUC ROC score per sample. Higher is better. Valid range is [0, 1].

Each method has 10 samples, so the dataset has 100 rows total.

## Quick descriptive statistics

The table below summarizes the distribution of AUC ROC per method.

| Method | n | Mean | Median | Std | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Baseline 2 | 10 | 0.917 | 0.908 | 0.047 | 0.859 | 1.000 |
| Baseline 1 | 10 | 0.767 | 0.768 | 0.028 | 0.724 | 0.815 |
| Proposed | 10 | 0.661 | 0.669 | 0.037 | 0.610 | 0.700 |
| Baseline 9 | 10 | 0.557 | 0.573 | 0.035 | 0.503 | 0.594 |
| Baseline 3 | 10 | 0.462 | 0.456 | 0.025 | 0.428 | 0.498 |
| Baseline 7 | 10 | 0.388 | 0.392 | 0.012 | 0.368 | 0.402 |
| Baseline 6 | 10 | 0.321 | 0.324 | 0.017 | 0.298 | 0.341 |
| Baseline 5 | 10 | 0.269 | 0.266 | 0.016 | 0.251 | 0.297 |
| Baseline 4 | 10 | 0.226 | 0.225 | 0.008 | 0.216 | 0.241 |
| Baseline 8 | 10 | 0.192 | 0.192 | 0.011 | 0.177 | 0.209 |


Key observations.

1. `Proposed` is competitive but not the top performer in this synthetic dataset. Baseline 2 and Baseline 1 have higher central tendency.
2. `Proposed` clearly outperforms Baselines 3 to 9 by a wide margin, both in median and overall range.

## Visualization strategy

Goal. Highlight `Proposed` against all baselines using preattentive visual encoding while staying non misleading.

Design decisions.

- I used a **horizontal boxplot** per method with **all individual samples overlaid** as jittered points. This avoids the common trap of bar charts hiding distributional information.
- The x axis spans **0 to 1** because AUC ROC is bounded. This keeps the scale honest and avoids exaggerating small differences by truncating the axis.
- `Proposed` is highlighted via multiple preattentive cues.
  - **Hue**. A saturated color for `Proposed`, neutral gray for baselines.
  - **Stroke weight**. A thicker outline for the `Proposed` box and median.
  - **Position**. `Proposed` is placed at the top of the y axis.
  - **Annotation**. The median of `Proposed` is called out directly to reduce lookup effort.

I ordered the baselines by median performance (descending). This is not a data transformation. It is a readability choice that makes relative performance comparisons faster.

## AI usage disclosure

I used ChatGPT to brainstorm a few visualization options for highlighting one group against many comparators. I then implemented the final design manually, validated the data ranges, and checked that the axes and ordering do not create misleading interpretations.
