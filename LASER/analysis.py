import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact
import ipywidgets as widgets

# Load the LabSE and LASER scores
labse_file_path = 'sim/wol-eng-labse-sim_scores.txt'
laser_file_path = 'sim/wol-eng-laser-sim-scores.txt'
vref_file_path = 'references/vref.txt'

# Read files
labse_scores = pd.read_csv(labse_file_path, header=None)
laser_scores = pd.read_csv(laser_file_path, header=None)
vref = pd.read_csv(vref_file_path, sep=" ", header=None)

# Prepare the scatter plot data
verses = vref[0]  # Assuming the first column is the verse reference
labse_scores.columns = ['LaBSE']
laser_scores.columns = ['LASER']

# Prepare the merged dataframe
df = pd.DataFrame({'Verses': verses, 'LaBSE': labse_scores['LaBSE'], 'LASER': laser_scores['LASER']})

# Function to create a plot with slider input
def plot_interactive(labse_threshold, laser_threshold):
    # Define the highlight condition based on the thresholds
    highlight_condition = (df['LaBSE'] > labse_threshold) & (df['LASER'] > laser_threshold)
    
    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['LaBSE'], df['LASER'], c='blue', alpha=0.5, label='Regular Verses')
    
    # Highlight the specific verses
    plt.scatter(df['LaBSE'][highlight_condition], df['LASER'][highlight_condition], 
                c='red', alpha=0.8, label='Highlighted Verses')
    
    # Add titles and labels
    plt.title(f"Scatter Plot of LaBSE vs LASER Scores (Thresholds: LaBSE > {labse_threshold}, LASER > {laser_threshold})")
    plt.xlabel("LaBSE Scores")
    plt.ylabel("LASER Scores")
    plt.legend()
    
    # Show the plot
    plt.show()

# Create sliders for LaBSE and LASER thresholds
interact(plot_interactive, 
         labse_threshold=widgets.FloatSlider(value=0.9, min=0, max=1, step=0.01, description='LaBSE Threshold'),
         laser_threshold=widgets.FloatSlider(value=0.9, min=0, max=1, step=0.01, description='LASER Threshold'))
