#%%
%matplotlib qt
#%%
import pickle
import numpy as np
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#%%
#%%
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"]})
#%%
sizes = [6.5,7,7.5,8]
n_trials = 10
heatmap_data = {size_n : {} for size_n in sizes}
table_data_list = []
for i in range(4):
    with open(f'data/heatmap/results_tables/size_{i}.txt') as f:
        table_data_list.append(f.readlines())
#%%
masses = [75,150,225,300]
# %%
for k, size in enumerate(sizes):
    for i, mass  in enumerate(masses):
        heatmap_data[size][mass] = np.empty(0)
        for j in range(n_trials):
            score = table_data_list[k][j].split()[i]
            if not (i == 3 and k == 0):
                score = float(score)
            else:
                score = np.nan
            heatmap_data[size][mass] = np.concatenate((heatmap_data[size][mass], np.array([score])))
# %%
heatmap_averages = copy.deepcopy(heatmap_data)
heatmap_data_no_half_points = copy.deepcopy(heatmap_data)
heatmap_averages_no_half_points = copy.deepcopy(heatmap_averages)
for size in sizes:
    for mass in masses:
        for i, score in enumerate(heatmap_data_no_half_points[size][mass]):
            if score == 0.5:
                heatmap_data_no_half_points[size][mass][i] = 0.0
#%%
for size in sizes:
    for mass in masses:
        heatmap_averages[size][mass] = heatmap_data[size][mass].mean()
        heatmap_averages_no_half_points[size][mass] = heatmap_data_no_half_points[size][mass].mean()
# %%
df = pd.DataFrame(heatmap_averages)
df_reversed = df.iloc[::-1]
# Optional: Transpose the DataFrame to align rows and columns
#df = df.T  # Transpose if you want rows to be categories
annot_vals = df_reversed.applymap(lambda x: r"$\mathbf{" + f"{x:.2f}" + "}$")# Step 2: Create the heatmap
plt.figure(figsize=(8, 6))  # Set the figure size

# Create the heatmap
ax = sns.heatmap(
    df_reversed, 
    annot=annot_vals, 
    cmap="viridis_r",  # Reversed colormap
    linewidths=0.5, 
    vmin=0.0, 
    vmax=1.0, 
    fmt="", 
    annot_kws={"size": "20", "weight": "bold"}
)

# Add black patch in top-right corner (corresponds to NaN at 300g and 6.5cm)
# This is row=0 (since reversed) and col=0
# ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=True, color='black', lw=0))
# === Draw an "X" in the top-left square (row=0, col=0) ===
# x0, y0 = 0, 0  # Top-left cell in heatmap coordinates
# ax.plot([x0, x0 + 1], [y0, y0 + 1], color='black', linewidth=2)
# ax.plot([x0, x0 + 1], [y0 + 1, y0], color='black', linewidth=2)
# === Hatch the top-left square (row=0, col=0) ===
hatch_rect = patches.Rectangle(
    (0, 0), 1, 1,
    hatch='///',  # Diagonal lines
    fill=False,
    edgecolor='black',
    linewidth=0.0
)
ax.add_patch(hatch_rect)
# Draw black rectangle around (mass=150, size=7)
# In df_reversed: mass=150 is at row index 2 (from bottom), size=7 is col index 1
ax.add_patch(plt.Rectangle((1, 2), 1, 1, fill=False, edgecolor='black', lw=4))

# Labeling
plt.xlabel("Diameter [cm]", fontsize=20)
plt.ylabel("Mass [g]", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
cbar.set_label("Success rate [-]", fontsize=20, labelpad=10)

plt.show()

# %%
# %%
df = pd.DataFrame(heatmap_averages_no_half_points)
df_reversed = df.iloc[::-1]
# Optional: Transpose the DataFrame to align rows and columns
#df = df.T  # Transpose if you want rows to be categories

# Step 2: Create the heatmap
plt.figure(figsize=(8, 6))  # Set the figure size
sns.heatmap(df_reversed, annot=True, cmap="viridis", linewidths=0.5, vmin=0.0, vmax=1.0)

# Add title
plt.title("Heatmap from Nested Dictionary")

# Show the plot
plt.show()
# %%
