Q1 Documentation  


Sprint Project 2 


Adam Horowitz
























Steps: 


1. I open 1d-data.csv in Excel and I confirm the columns are value and group (10 cases, 10 control).

2. I build a small summary table (Group / Mean) and I use AVERAGEIF to compute group means.

3. I get #DIV/0!, so I diagnose it: I use COUNTIF to confirm the criteria match (10 and 10), which tells me the issue is the values aren’t numeric.

4. I convert the value column from text to numbers (Convert to Number / Text-to-Columns), and the AVERAGEIF means update correctly.

5. I create a helper x column to position groups: I assign 1 for cases and 2 for control.

6. I insert an XY Scatter plot using x as the X-values and value as the Y-values.

7. When the plot looks wrong (axes swapped), I edit the series ranges so X = x column and Y = value column.

8. When Excel creates multiple series by accident, I open Select Data Source, remove the extra series, and keep one correct data series with the proper X/Y ranges.

9. I set the Y-axis to logarithmic scale because the values span from single digits to ~35,000, and I label the y-axis as Value (log scale) to keep the visualization non-misleading.