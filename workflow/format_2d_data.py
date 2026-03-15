# Ben Benson Challenge 2: Visualize 2D Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # didn't end up using seaborn but imported it just in case.

# Load the 2D data
data = pd.read_csv('./applsoftcomp-sprint-m02-master/data/2d-data.csv') #Load the 2D data from location

# Inspect the data structure
print(data.head()) #Print the first few (5 by default) rows of the data to understand its structure. This is useful for understand
# This is useful for understanding what plot should be used

# Data is a list of x,y coordinates. Therefore we use a scatterplot
plt.figure(figsize=(8, 6))
plt.scatter(data['x'], data['y'], alpha=0.4) # Create scatter plot. Points have .4 opacity to help with visibility of overlapping points

# Calculate and plot line of best fit
m, b = np.polyfit(data['x'], data['y'], 1)  #use polyfit to calculate the slope (m) and intercept (b) of the line of best fit for the data. 
plt.plot(data['x'], m * data['x'] + b, color='red', label='Best Fit Line') # plots line of best fit


plt.title('2D Data Scatter Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout() # Auto formats plot elements to make it look better and avoid overlap
plt.savefig('./figs/fig-2d-data.png')
plt.show() # displays plot in a pop up window
