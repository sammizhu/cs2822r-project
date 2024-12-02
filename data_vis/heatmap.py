import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV data
file_path = '/Users/sammizhu/cs2822r-project/data_vis/results.csv'  
data = pd.read_csv(file_path)

# Ensure 'User Correct' column is numeric and represents accuracy directly
# (In this data, it appears to already be numeric, so we can proceed directly)

# Pivot the data to create a table for the heatmap
heatmap_data = data.pivot_table(values='User Correct', index='Layer', columns='Features Shown', aggfunc='mean')

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Impact of Hidden Layers and Features Shown on User Accuracy")
plt.xlabel("Features Shown (Top-k)")
plt.ylabel("CNN Layer")
plt.show()