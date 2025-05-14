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
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"]})
#%%
sizes = [0,1,2,3]
masses = [75,150,225,300]
ids = [0,14]
n_trials = 10
heatmap_data = {size_n : {mass : {id : [] for id in ids} for mass in masses} for size_n in sizes}
heatmap_successes = {size_n : {mass : [] for mass in masses} for size_n in sizes}
table_data_list = []
for i in range(4):
    for j, mass in enumerate(masses):
        if i == 0 and mass == 300:
            continue
        for k in range(10):
            with open(f'data/heatmap/size_{i}/{mass}/target_{k}.pkl', 'rb') as f:
                marker_data = pickle.load(f)
                for marker in marker_data:
                    id = marker.id[0]
                    xyz_position = marker.pose.pose.pose.pose.position
                    position_trial = np.array([xyz_position.x, xyz_position.y, xyz_position.z])
                    heatmap_data[i][mass][id].append(position_trial)
    with open(f'data/heatmap/results_tables/size_{i}.txt', 'r') as f:
        success_list = f.readlines()
        for j, mass in enumerate(masses):
            for n_trial, line in enumerate(success_list):
                heatmap_successes[i][mass] = (line.split())

                
# %%
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# Define your colors for IDs
id_symbols = {0: '*', 14: 'x'}
size_colors = {0 : 'yellow', 1: 'red', 2: 'purple', 3 : 'green'}

# Function to plot heatmap data
def plot_heatmap(heatmap_data, sizes=None, masses=None, ids=None):
    """
    Plots heatmap data with filtering options.
    
    Parameters:
        heatmap_data (dict): The nested dictionary with position data.
        sizes (list): List of sizes to include in the plot (default: all).
        masses (list): List of masses to include in the plot (default: all).
        ids (list): List of ids to include in the plot (default: all).
    """
    sizes = sizes if sizes is not None else heatmap_data.keys()
    masses = masses if masses is not None else next(iter(heatmap_data.values())).keys()
    ids = ids if ids is not None else next(iter(next(iter(heatmap_data.values())).values())).keys()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for size, mass, id_ in product(sizes, masses, ids):
        if size not in heatmap_data:
            print(f"Size {size} not found, skipping.")
            continue
        if mass not in heatmap_data[size]:
            print(f"Mass {mass} not found for size {size}, skipping.")
            continue
        if id_ not in heatmap_data[size][mass]:
            print(f"ID {id_} not found for size {size} and mass {mass}, skipping.")
            continue
        
        positions = heatmap_data[size][mass][id_]
        if positions:
            positions = np.array(positions)
            ax.scatter(
                positions[:, 0], 
                positions[:, 1], 
                label=f"Size {size}, Mass {mass}, ID {id_}",
                color=size_colors.get(size, 'black'),  # Default color is black
                marker=id_symbols.get(id_, 'O'),
                alpha=0.7
            )
    
    ax.set_title("Heatmap Data")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Legend")
    plt.tight_layout()
    plt.show()

# Example of calling the function
# Plot for sizes [0, 1], masses [75, 150], and both IDs (0 and 14)
plot_heatmap(
    heatmap_data, 
    sizes=[0,1,2,3], 
    masses=[75,150,225,300], 
    ids=[0, 14]
)
# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm

# Constants
radius_scale=0.5
size_to_diameter = {
    0: 0.06*radius_scale,
    1: 0.065*radius_scale,
    2: 0.07*radius_scale,
    3: 0.075*radius_scale
}
mass_list = [75, 150, 225, 300]
mass_to_color = {mass: cm.viridis(i / (len(mass_list) - 1)) for i, mass in enumerate(mass_list)}

# Cylinder generator
def create_cylinder(center_x, center_y, radius, height=0.01, resolution=20):
    z = np.linspace(0, height, 2)
    theta = np.linspace(0, 2 * np.pi, resolution)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = center_x + radius * np.cos(theta_grid)
    y_grid = center_y + radius * np.sin(theta_grid)
    return x_grid, y_grid, z_grid

# Plotting function
def plot_3d_cylinders(heatmap_data, sizes, masses, id_=0, cylinder_height=0.1):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for size in sizes:
        for mass in masses:
            if size not in heatmap_data or mass not in heatmap_data[size] or id_ not in heatmap_data[size][mass]:
                continue
            
            positions = heatmap_data[size][mass][id_]
            if not positions:
                continue

            positions = np.array(positions)
            color = mass_to_color.get(mass, 'gray')
            radius = size_to_diameter[size] / 2

            for pos in positions:
                x, y, _ = pos  # Ignore Z from data
                Xc, Yc, Zc = create_cylinder(x, y, radius, height=cylinder_height)
                ax.plot_surface(Xc, Yc, Zc, color=color, alpha=0.7, linewidth=0, shade=True)

    ax.set_title(f"3D Cylinders for ID {id_}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (Height)")
    ax.view_init(elev=30, azim=120)
    ax.set_zlim(0, 0.05)
    plt.tight_layout()
    plt.show()

# Call the function
plot_3d_cylinders(heatmap_data, sizes=[0,1,2,3], masses=[75,150,225,300], id_=0, cylinder_height=0.01)

# %%
import matplotlib.pyplot as plt
from matplotlib import cm

# Diameter mapping (size -> visual marker size)
size_to_diameter = {
    0: 0.04,
    1: 0.08,
    2: 0.12,
    3: 0.18
}

# Use a colormap for mass
mass_list = [75, 150, 225, 300]
mass_norm = plt.Normalize(min(mass_list), max(mass_list))
cmap = cm.tab10

def plot_2d_markers_by_size_and_mass(heatmap_data, sizes, masses, id_=0):
    fig, ax = plt.subplots(figsize=(10, 8))

    for size in sizes:
        for i, mass in enumerate(masses):
            if size not in heatmap_data or mass not in heatmap_data[size] or id_ not in heatmap_data[size][mass]:
                continue
            
            positions = heatmap_data[size][mass][id_]
            if not positions:
                continue

            positions = np.array(positions)
            diameter = size_to_diameter.get(size, 0.06)
            marker_size = (diameter * 1000)  # squared for visibility scaling

            ax.scatter(
                positions[:, 0],
                positions[:, 1],
                s=marker_size,
                c=cmap(i),
                alpha=0.7,
                label=f"Size {size}, Mass {mass}"
            )

    ax.set_title(f"2D Position Plot for ID {id_}")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_aspect('equal', 'box')
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Size/Mass")
    # plt.colorbar(cm.ScalarMappable(norm=mass_norm, cmap=cmap), ax=ax, label='Mass')
    plt.tight_layout()
    plt.show()

# Call the function
plot_2d_markers_by_size_and_mass(heatmap_data, sizes=[0,1,2,3], masses=[75,150,225,300], id_=0)


# %%
