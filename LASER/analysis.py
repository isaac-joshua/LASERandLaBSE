import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# Reading the two files
file_path_1 = 'sim/wol-eng-labse-sim_scores.txt'  # Replace with the correct path for file 1
file_path_2 = 'sim/wol-eng-laser-sim-scores.txt'  # Replace with the correct path for file 2

data_1 = pd.read_csv(file_path_1, header=None)
data_2 = pd.read_csv(file_path_2, header=None)

plt.figure(figsize=(10,6))
plt.plot(data_1.index, data_1[0], label='LABSE')
plt.plot(data_2.index, data_2[0], label='LASER', linestyle='--')
plt.title('Comparison of Similarity Scores from Two Files')
plt.xlabel('Index')
plt.ylabel('Similarity Score')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
plt.hist(data_1[0], bins=20, color='blue', alpha=0.5, label='LABSE')
plt.hist(data_2[0], bins=20, color='orange', alpha=0.5, label='LASER')
plt.title('Distribution of Similarity Scores for Both Files')
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()


correlation = np.corrcoef(data_1[0], data_2[0])[0, 1]
print(f'Correlation between File 1 and File 2 similarity scores: {correlation:.4f}')

mean_1 = np.mean(data_1[0])
mean_2 = np.mean(data_2[0])
variance_1 = np.var(data_1[0])
variance_2 = np.var(data_2[0])

print(f'LABSE - Mean: {mean_1:.4f}, Variance: {variance_1:.4f}')
print(f'LASER - Mean: {mean_2:.4f}, Variance: {variance_2:.4f}')

