import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

# --- Configuration ---

# The CSV file containing the recall data.
# This should match the output of your evaluation script.
RECALL_CSV_FILE = "query_10_varying_k_results.csv"

# The name of the file to save the plot
OUTPUT_PLOT_FILE = "recall_3d_surface_plot.png"

# --- End Configuration ---


def create_dummy_data(file_path):
    """
    Creates a dummy CSV file if the target file is not found.
    This generates a 10x10 grid (100 entries) for k1 and k2.
    """
    print(f"Warning: File not found at '{file_path}'.")
    print("Creating a dummy file for demonstration...")
    
    k_values = list(range(100, 1001, 100))
    k1_list = np.repeat(k_values, 10) # [100, 100, ..., 200, 200, ...]
    k2_list = np.tile(k_values, 10)   # [100, 200, ..., 1000, 100, 200, ...]
    
    # Generate a plausible recall surface (e.g., a sine wave)
    # This makes for a more interesting plot than pure random noise.
    def mock_recall(k1, k2):
        # A sample function where recall peaks around k1=500, k2=500
        k1_norm = (k1 - 550) / 500
        k2_norm = (k2 - 550) / 500
        return 0.4 + 0.5 * (np.sin(1 - k1_norm**2) * np.cos(1 - k2_norm**2))
    
    recall_list = [mock_recall(k1, k2) for k1, k2 in zip(k1_list, k2_list)]
    
    dummy_df = pd.DataFrame({
        'query': '1912 films set in England',
        'k1': k1_list,
        'k2': k2_list,
        'recall_score': recall_list
    })
    
    dummy_df.to_csv(file_path, index=False)
    print(f"Dummy file '{file_path}' created.")

def plot_3d_surface(csv_path, output_image_path):
    """
    Loads data from the CSV and generates a 3D surface plot.
    """
    if not os.path.exists(csv_path):
        create_dummy_data(csv_path)

    # Load the data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # 1. Reshape the data for a 3D surface
    # We pivot the table to create a 2D grid of recall scores,
    # with k1 as rows and k2 as columns.
    try:
        pivot_table = df.pivot(index='k1', columns='k2', values='recall_score')
    except Exception as e:
        print(f"Error pivoting data. Is the CSV format correct?")
        print("Expected columns: 'k1', 'k2', 'recall_score'")
        print(f"Error details: {e}")
        return

    # 2. Get the X (k1) and Y (k2) coordinates
    k1_values = pivot_table.index.values
    k2_values = pivot_table.columns.values
    
    # 3. Create a 2D grid (mesh) for the plot
    # K1 and K2 will be 2D arrays that provide the (x, y) coordinates for each
    # point on the 3D surface.
    K1, K2 = np.meshgrid(k1_values, k2_values)
    
    # 4. Get the Z (Recall) values
    # We must transpose the pivot table's values to match the shape
    # created by np.meshgrid.
    Z = pivot_table.values.T
    
    # 5. Create the plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(K1, K2, Z, cmap='viridis', edgecolor='none')
    
    # 6. Customize the plot
    ax.set_xlabel('k1 ("1912 films")')
    ax.set_ylabel('k2 ("films set in England")')
    ax.set_zlabel('Recall Score')
    ax.set_title('Recall Score vs. k1 and k2 for Query 10', pad=20)
    
    # Add a color bar to map values to colors
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Recall Score')
    
    # Set a good viewing angle
    ax.view_init(elev=30, azim=135)
    
    # 7. Save and show the plot
    try:
        plt.savefig(output_image_path, bbox_inches='tight')
        print(f"3D surface plot saved to: {output_image_path}")
        # plt.show() # Uncomment this line if you want the plot to pop up
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == "__main__":
    plot_3d_surface(RECALL_CSV_FILE, OUTPUT_PLOT_FILE)
