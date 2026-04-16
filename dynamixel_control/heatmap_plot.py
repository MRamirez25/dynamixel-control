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
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from rembg import remove
#%%
#%%
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"]})
#%%
sizes = [4.5,5.5,6.5,7.5]
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
fig = plt.figure(figsize=(6, 6), dpi=300)  # Set DPI for the figure

# Create the heatmap
ax = sns.heatmap(
    df_reversed, 
    annot=annot_vals, 
    cmap="viridis_r",  # Reversed colormap
    linewidths=0.5, 
    vmin=0.0, 
    vmax=1.0, 
    fmt="", 
    annot_kws={"size": "16", "weight": "bold"}
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
plt.xlabel("Diameter [cm]", fontsize=16, labelpad=54)
plt.ylabel("Mass [g]", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=16)
cbar.set_label("Success rate [-]", fontsize=16, labelpad=10)

# === Helper function to process icons ===
def process_icon(icon_path):
    """
    Open image with PIL, remove background with rembg, 
    and crop to bounding box of non-transparent pixels.
    """
    # Open and convert to RGBA
    img = Image.open(icon_path).convert("RGBA")
    
    # Remove background
    img = remove(img)
    
    # Convert to numpy to find bounding box
    img_arr = np.array(img)
    
    # Use a much higher threshold to exclude faint/semi-transparent pixels
    # that rembg might leave behind
    alpha_threshold = 100  # Only keep nearly-opaque pixels (0-255 scale)
    mask = img_arr[:, :, 3] > alpha_threshold
    
    print(f"Alpha range: min={img_arr[:,:,3].min()}, max={img_arr[:,:,3].max()}")
    print(f"Pixels with alpha > {alpha_threshold}: {mask.sum()}")
    
    # Find bounding box coordinates
    # np.argwhere returns (row, col) which is (y, x)
    coords = np.argwhere(mask)
    if len(coords) == 0:
        # No pixels above threshold, try with lower threshold
        print(f"Warning: No pixels above threshold {alpha_threshold}, trying lower threshold")
        alpha_threshold = 100
        mask = img_arr[:, :, 3] > alpha_threshold
        coords = np.argwhere(mask)
        if len(coords) == 0:
            print("Still no pixels found, returning original")
            return img
    
    # Get min/max for rows (y) and columns (x)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    print(f"Bounding box: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
    
    # PIL crop expects (left, upper, right, lower) = (x_min, y_min, x_max, y_max)
    img_cropped = img.crop((x_min, y_min, x_max+1, y_max+1))
    
    print(f"Cropped from {img.size} to {img_cropped.size}\n")
    
    return img_cropped

# === Add icons below x-axis labels ===
# Icon image paths for each diameter category (replace with your actual paths)
icon_paths = [
    "/home/mariano/phd_code/cup_images/IMG_0140.png",  # Icon for 6.5cm
    "/home/mariano/phd_code/cup_images/IMG_0141.png",    # Icon for 7cm
    "/home/mariano/phd_code/cup_images/IMG_0142.png",  # Icon for 7.5cm
    "/home/mariano/phd_code/cup_images/IMG_0143.png"     # Icon for 8cm
]

# Adjust plot to make room for icons below BEFORE adding them
plt.subplots_adjust(bottom=0.25)  # Make room at bottom for icons

# Load icons and add them below the x-axis
icon_zoom = 0.03  # Adjust size of icons

# Get the x-tick positions in data coordinates
xtick_positions = ax.get_xticks()
print(f"X-tick positions: {xtick_positions}")

for i, icon_path in enumerate(icon_paths):
    try:
        # Process icon: remove background and crop to bounding box
        img_pil = process_icon(icon_path)
        
        # Convert PIL Image to numpy array for matplotlib
        img = np.array(img_pil)
        
        imagebox = OffsetImage(img, zoom=icon_zoom, interpolation='bilinear', resample=True, dpi_cor=True)
        
        # Use axes fraction coordinates for better control
        # Transform data x-position to axes fraction
        x_data = xtick_positions[i]  # Center of each heatmap column (now properly cropped)
        
        # Create annotation box with blended transform:
        # x in data coords, y in axes fraction (0 = bottom, 1 = top)
        from matplotlib.transforms import blended_transform_factory
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        
        ab = AnnotationBbox(imagebox, (x_data, -0.17),  # -0.15 in axes fraction = below axis
                            frameon=False, 
                            box_alignment=(0.5, 0.5),  # Align center of box to point (horizontally and vertically)
                            xycoords=trans,
                            clip_on=False)  # Allow drawing outside axes
        ax.add_artist(ab)
        print(f"Added icon {i} at x={x_data} (data), y=-0.15 (axes fraction)")
    except FileNotFoundError:
        print(f"Warning: Icon not found at {icon_path}")
    except Exception as e:
        print(f"Error processing icon {icon_path}: {e}")

# Save with high quality
plt.savefig("heatmap_with_icons.pdf", dpi=300, bbox_inches="tight")
plt.savefig("heatmap_with_icons.png", dpi=300, bbox_inches="tight")
print("Saved heatmap_with_icons.pdf and .png at 300 DPI")

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
