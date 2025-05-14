#%%
%matplotlib qt
#%%
import pickle
import numpy as np
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%%
with open(f'data/noise/results_tables/directions.txt') as f:
    noise_vectors_angles = f.readlines()
with open(f'data/noise/results_tables/success.txt') as f:
    success_list = f.readlines()
# %%
noise_vectors_angles_array = np.empty((10,3))
for i, line in enumerate(noise_vectors_angles):
    noise_vectors_angles_array[i, :] = line.split()
# %%
success_array = np.empty((10,3))
for i, line in enumerate(success_list):
    success_array[i, :] = line.split()
# %%
# Define magnitudes for each column
magnitudes = [0.05, 0.1, 0.15]

# Compute x and y coordinates for the endpoints of each vector
x = np.cos(noise_vectors_angles_array) * magnitudes  # x = magnitude * cos(angle)
y = np.sin(noise_vectors_angles_array) * magnitudes  # y = magnitude * sin(angle)

# Create the plot
plt.figure(figsize=(8, 8))

# Loop through columns to plot points
for i in range(3):  # Loop through each column
    for j in range(10):  # Loop through each row
        color = 'green' if success_array[j, i] == 1 else 'red'  # Green for success, red for failure
        plt.scatter(x[j, i], y[j, i], color=color, s=50, marker='x')

circle_radii = [0.05, 0.1, 0.15]
for radius in circle_radii:
    circle = plt.Circle((0, 0), radius, color='black', alpha=0.3, fill=False, linestyle='--', linewidth=1)
    plt.gca().add_patch(circle)

# Add formatting
plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)  # x-axis
plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)  # y-axis
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Endpoints of Vectors with Success (Green) and Failure (Red)")
plt.axis("equal")  # Ensures equal scaling on both axes
plt.show()
# %%
from matplotlib.patches import Wedge

# Create the plot
plt.figure(figsize=(8, 8))

# Loop through columns to plot points
for i in range(3):  # Loop through each column
    for j in range(10):  # Loop through each row
        color = 'green' if success_array[j, i] == 1 else 'red'  # Green for success, red for failure
        plt.scatter(x[j, i], y[j, i], color=color, s=50, marker='x')

circle_radii = [0.05, 0.1, 0.15]
for radius in circle_radii:
    circle = plt.Circle((0, 0), radius, color='black', alpha=0.3, fill=False, linestyle='--', linewidth=1)
    plt.gca().add_patch(circle)

# Define the radii for each ring (e.g., center 0.05, 0.1, 0.15) and their thickness
magnitudes = [0.05, 0.1, 0.15]
ring_width = 0.045  # Slightly smaller than spacing to avoid touching

angles_deg = np.degrees(noise_vectors_angles_array) % 360
n_sectors = 12
sector_angle = 360 / n_sectors

from matplotlib.patches import Wedge

# Define the *edges* of the rings instead of center radii
ring_edges = [0.0, 0.05, 0.1, 0.15]
n_rings = len(ring_edges) - 1
n_sectors = 12
sector_angle = 360 / n_sectors

angles_deg = np.degrees(noise_vectors_angles_array) % 360

ax = plt.gca()
for ring_idx in range(n_rings):
    inner_radius = ring_edges[ring_idx]
    outer_radius = ring_edges[ring_idx + 1]
    for sector in range(n_sectors):
        start_angle = sector * sector_angle
        end_angle = (sector + 1) * sector_angle

        # For this ring, get the relevant column in data
        ring_column = ring_idx  # matches the magnitude index
        in_sector = (angles_deg[:, ring_column] >= start_angle) & (angles_deg[:, ring_column] < end_angle)
        sector_values = success_array[:, ring_column][in_sector]

        color = None
        if np.any(sector_values == 0):
            color = 'red'
        elif np.any(sector_values == 1):
            color = 'green'

        if color:
            wedge = Wedge(
                center=(0, 0),
                r=outer_radius,
                theta1=start_angle,
                theta2=end_angle,
                width=outer_radius - inner_radius,
                facecolor=color,
                edgecolor=None,
                alpha=0.25
            )
            ax.add_patch(wedge)

# Add formatting
plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)  # x-axis
plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)  # y-axis
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Endpoints of Vectors with Success (Green) and Failure (Red)")
plt.axis("equal")  # Ensures equal scaling on both axes
plt.show()
# %%
