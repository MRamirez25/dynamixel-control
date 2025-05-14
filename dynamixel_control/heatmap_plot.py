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
#%%
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"]})
#%%
sizes = [0,1,2,3]
n_trials = 10
heatmap_data = {size_n : {} for size_n in sizes}
table_data_list = []
for i in range(4):
    with open(f'data/heatmap/results_tables/size_{i}.txt') as f:
        table_data_list.append(f.readlines())
#%%
masses = [75,150,225,300]
# %%
for size in sizes:
    for i, mass  in enumerate(masses):
        heatmap_data[size][mass] = np.empty(0)
        for j in range(n_trials):
            score = table_data_list[size][j].split()[i]
            if not (i == 3 and size == 0):
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

# Step 2: Create the heatmap
plt.figure(figsize=(8, 6))  # Set the figure size
sns.heatmap(df_reversed, annot=True, cmap="viridis", linewidths=0.5, vmin=0.0, vmax=1.0)

# Add title
plt.title("Heatmap from Nested Dictionary")

# Show the plot
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
