import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# Reading the two files
file_path_1 = 'references/sim_scores.txt'  # Replace with the correct path for file 1
file_path_2 = 'references/sim_scores.txt'  # Replace with the correct path for file 2

data_1 = pd.read_csv(file_path_1, header=None)
data_2 = pd.read_csv(file_path_2, header=None)

# 1. Plot of the similarity scores from both files
plt.figure(figsize=(10,6))
plt.plot(data_1.index, data_1[0], label='File 1 Similarity Scores')
plt.plot(data_2.index, data_2[0], label='File 2 Similarity Scores', linestyle='--')
plt.title('Comparison of Similarity Scores from Two Files')
plt.xlabel('Index')
plt.ylabel('Similarity Score')
plt.legend()
plt.show()

# 2. Histogram comparison of the two files' similarity scores
plt.figure(figsize=(10,6))
plt.hist(data_1[0], bins=20, color='blue', alpha=0.5, label='File 1')
plt.hist(data_2[0], bins=20, color='orange', alpha=0.5, label='File 2')
plt.title('Distribution of Similarity Scores for Both Files')
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# 3. Rolling average of the similarity scores from both files with a window size of 100
rolling_mean_1 = data_1[0].rolling(window=100).mean()
rolling_mean_2 = data_2[0].rolling(window=100).mean()

plt.figure(figsize=(10,6))
plt.plot(data_1.index, rolling_mean_1, label='File 1 Rolling Average (window=100)', color='blue')
plt.plot(data_2.index, rolling_mean_2, label='File 2 Rolling Average (window=100)', color='orange', linestyle='--')
plt.title('Rolling Average of Similarity Scores for Both Files')
plt.xlabel('Index')
plt.ylabel('Similarity Score')
plt.legend()
plt.show()

# 4. Difference between the similarity scores of the two files
plt.figure(figsize=(10,6))
plt.plot(data_1.index, data_1[0] - data_2[0], label='Difference (File 1 - File 2)', color='purple')
plt.title('Difference Between Similarity Scores of File 1 and File 2')
plt.xlabel('Index')
plt.ylabel('Difference in Similarity Score')
plt.legend()
plt.show()

# 5. Linear Regression for both files
slope_1, intercept_1, _, _, _ = stats.linregress(data_1.index, data_1[0])
slope_2, intercept_2, _, _, _ = stats.linregress(data_2.index, data_2[0])

plt.figure(figsize=(10,6))
plt.plot(data_1.index, data_1[0], label='File 1 Similarity Scores', color='blue')
plt.plot(data_1.index, intercept_1 + slope_1*data_1.index, label='File 1 Regression Line', color='blue', linestyle='--')
plt.plot(data_2.index, data_2[0], label='File 2 Similarity Scores', color='orange')
plt.plot(data_2.index, intercept_2 + slope_2*data_2.index, label='File 2 Regression Line', color='orange', linestyle='--')
plt.title('Linear Regression of Similarity Scores for Both Files')
plt.xlabel('Index')
plt.ylabel('Similarity Score')
plt.legend()
plt.show()

# 6. Correlation between the two files' similarity scores
correlation = np.corrcoef(data_1[0], data_2[0])[0, 1]
print(f'Correlation between File 1 and File 2 similarity scores: {correlation:.4f}')

# 7. Mean and Variance comparison
mean_1 = np.mean(data_1[0])
mean_2 = np.mean(data_2[0])
variance_1 = np.var(data_1[0])
variance_2 = np.var(data_2[0])

print(f'File 1 - Mean: {mean_1:.4f}, Variance: {variance_1:.4f}')
print(f'File 2 - Mean: {mean_2:.4f}, Variance: {variance_2:.4f}')

# 8. Scatter plot to show the relationship between similarity scores of the two files
plt.figure(figsize=(10,6))
plt.scatter(data_1[0], data_2[0], alpha=0.5, color='green')
plt.title('Scatter Plot of Similarity Scores between File 1 and File 2')
plt.xlabel('File 1 Similarity Scores')
plt.ylabel('File 2 Similarity Scores')
plt.show()
