#%%
%matplotlib qt
#%%
from gpt_tentacle import GPT_tag_tentacle
import time
from geometry_msgs.msg import PoseStamped
import rospy
from sklearn.gaussian_process.kernels import RBF, Matern,WhiteKernel, ConstantKernel as C
import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt
import dynamic_reconfigure.client
import os
import random
from mpl_toolkits.mplot3d import Axes3D
from policy_transportation import GaussianProcessTransportation as Transport
from tag_detector import convert_distribution
import pathlib
# %%
transport = Transport()
#%%
skill_name = "dynamic_grasp"
# %%
def load_distributions(filename='source'):
    try:
        with open(f"distributions/{filename}.pkl","rb") as source:
            source_distribution = pickle.load(source)
    except:
        print("No source distribution saved")

    try:
        with open("distributions/target.pkl","rb") as target:
            target_distribution = pickle.load(target)
    except:
        print("No target distribution saved")
    return source_distribution, target_distribution    
#%%
source_distribution, target_distribution = load_distributions(skill_name)
#%%
target_dist_list = []
n_trajs = 10
for i in range(n_trajs):
    if skill_name == 'stack_two':
        if i == 12 or i == 18 or i == 19:
            continue
    with open(f'data/{skill_name}/target_{i}.pkl', 'rb') as f:
        marker_data = pickle.load(f)
        target_dist_list.append(marker_data)
# %%
transported_trajs = []
converted_targets_dists = []
converted_targets_dists_proj = []
tentacle_actuations = []
for i in range(n_trajs):
    converted_source_dist, converted_target_dist, dist = convert_distribution(source_distribution, target_dist_list[i])
    converted_targets_dists.append(converted_target_dist)
    ##%%
    try:
        del transport.training_delta
        del transport.training_dK
    except:
        print('already deleted')
    ##%%
    transport.source_distribution = converted_source_dist
    transport.target_distribution = converted_target_dist
    ## %%
    data =np.load(str(pathlib.Path().resolve())+'/data/'+str(skill_name)+'.npz')
    ## %%
    transport.training_traj = data['training_traj']
    transport.training_ori = data['training_ori']
    ##%%
    original_source_distribution = copy.deepcopy(transport.source_distribution)
    original_target_distribution = copy.deepcopy(transport.target_distribution)
    distances = np.empty((len(transport.training_traj), 0))
    for j in range(transport.source_distribution.shape[0]):
        distance_frame_i = np.linalg.norm(transport.training_traj - transport.source_distribution[j], keepdims=True, axis=-1)
        distances = np.hstack((distances, distance_frame_i))
    ##%%
    closest_points_idxs = np.argmin(distances, axis=0)
    projected_source_distribution = transport.training_traj[closest_points_idxs]
    transport.source_distribution = projected_source_distribution
    transport.source_distribution[:,-1] = transport.target_distribution[:,-1]
    ##%%
    ##%%
    source_target_diff = transport.target_distribution - original_source_distribution
    projected_target_distribution = projected_source_distribution + source_target_diff
    transport.target_distribution = projected_target_distribution
    transport.target_distribution[:,-1] = transport.source_distribution[:,-1]
    converted_targets_dists_proj.append(transport.target_distribution)
    ##%%
    transport.fit_transportation()
    ## %%
    transport.apply_transportation()
    ## %%
    transported_trajs.append(transport.training_traj)
    tentacle_actuations.append(data["training_dynamixels"])
# %%
color_values = np.linspace(0, 1, len(target_dist_list))  # Generate values from 0 to 1
cmap = plt.get_cmap("tab20")  # Use HSV colormap for distinct colors

colors = cmap(color_values)  # Map colors to colormap
#%%
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"]})
#%%
# # Create a 3D plot
fig = plt.figure(figsize=(9,6))
axs=[]
for n in range(9):
    temp_ax = fig.add_subplot(3,3,n+1, projection='3d')
    axs.append(temp_ax)
# ax = fig.add_subplot(331, projection='3d')
old_traj = transport.training_traj_old
start_idx = 0
end_idx = -1
# Plot the trajectories
# ax.plot(old_traj[start_idx:end_idx,0], old_traj[start_idx:end_idx,1], old_traj[start_idx:end_idx,2], label='Demonstration', color='black', linestyle='--', linewidth=2)
for n in range(9):
    axs[n].plot(transported_trajs[n][start_idx:end_idx,0], transported_trajs[n][start_idx:end_idx,1],transported_trajs[n][start_idx:end_idx,2], color=colors[n])
# ax.scatter3D(transport.source_distribution[0,0], transport.source_distribution[0,1],transport.source_distribution[0,2],marker="$x$", s=100, color='black')
# ax.scatter3D(transport.source_distribution[1,0], transport.source_distribution[1,1],transport.source_distribution[1,2],marker="$1$", s=100)
# ax.scatter3D(original_source_distribution[0,0], original_source_distribution[0,1],original_source_distribution[0,2],marker="*", s=100)

# ax.scatter3D(gpt.source_distribution[2,0], gpt.source_distribution[2,1],gpt.source_distribution[2,2],marker="$3$", s=100)
for n in range(9):
    axs[n].scatter3D(converted_targets_dists_proj[n][0,0], converted_targets_dists_proj[n][0,1],converted_targets_dists_proj[n][0,2]-converted_targets_dists_proj[n][1,2],marker="*", s=100, color=colors[n])
# ax.scatter3D(original_target_distribution[0,0], original_target_distribution[0,1],original_target_distribution[0,2],marker="*", s=100)

# ax.scatter3D(gpt.target_distribution[2,0], gpt.target_distribution[2,1],gpt.target_distribution[2,2],marker="$3$", s=100)
# ax.scatter3D([], [], [], marker='x', color='black', label='Marker 2')
# ax.scatter3D(0, 0, 0, marker='*', color='black', label='Marker 1', s=100)

# Customize the plot
# ax.set_title('Transported stacking trajectories')
# ax.set_xlabel('x [m]')
# ax.set_ylabel('y [m]')
# ax.set_zlabel('z [m]')
# ax.legend()

# Show the plot
plt.show()
#%%
# # Create a 3D plot
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')
old_traj = transport.training_traj_old
start_idx = 40
end_idx = -250
# Plot the trajectories
ax.plot(old_traj[start_idx:end_idx,0], old_traj[start_idx:end_idx,1], old_traj[start_idx:end_idx,2], label='Demonstration', color='black', linestyle='--', linewidth=2)
for n in range(len(transported_trajs)):
    ax.plot(transported_trajs[n][start_idx:end_idx,0], transported_trajs[n][start_idx:end_idx,1],transported_trajs[n][start_idx:end_idx,2], color=colors[n], linewidth=2)
ax.scatter3D(transport.source_distribution[0,0], transport.source_distribution[0,1],0,marker="$*$", s=120, color='black')
# ax.scatter3D(transport.source_distribution[1,0], transport.source_distribution[1,1],transport.source_distribution[1,2],marker="$1$", s=100)
# ax.scatter3D(original_source_distribution[0,0], original_source_distribution[0,1],original_source_distribution[0,2],marker="*", s=100)

# ax.scatter3D(gpt.source_distribution[2,0], gpt.source_distribution[2,1],gpt.source_distribution[2,2],marker="$3$", s=100)
for n in range(len(converted_targets_dists)):
    ax.scatter3D(converted_targets_dists_proj[n][0,0], converted_targets_dists_proj[n][0,1],0,marker="*", s=200, color=colors[n])
    # ax.scatter3D(converted_targets_dists_proj[n][1,0], converted_targets_dists_proj[n][1,1],converted_targets_dists_proj[n][1,2],marker="$1$", s=100, color=colors[n])
# ax.scatter3D(original_target_distribution[0,0], original_target_distribution[0,1],original_target_distribution[0,2],marker="*", s=100)

# ax.scatter3D(gpt.target_distribution[2,0], gpt.target_distribution[2,1],gpt.target_distribution[2,2],marker="$3$", s=100)
# ax.scatter3D(0, 0, 0, marker='*', color='black', label='Initial object locations', s=100)
ax.scatter3D([], [], [], marker='*', color='black', label='Initial object locations', s=100)

# Customize the plot
# ax.set_title('Transported stacking trajectories')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.legend()

# Show the plot
plt.show()
# %%
###################################### Create a 2D plot
fig, ax = plt.subplots(figsize=(9,6),dpi=300)
# Plot the trajectories
# ax.plot(old_traj[start_idx:end_idx,0]-transport.source_distribution[1,0], old_traj[start_idx:end_idx,1]-transport.source_distribution[1,1], old_traj[start_idx:end_idx,2]-transport.source_distribution[1,2], label='Demonstration', color='black', linestyle='--', linewidth=2)
# for n in range(len(transported_trajs)):
    # ax.plot(transported_trajs[n][start_idx:end_idx,0]-converted_targets_dists_proj[n][1,0], transported_trajs[n][start_idx:end_idx,1]-converted_targets_dists_proj[n][1,1],transported_trajs[n][start_idx:end_idx,2]-converted_targets_dists_proj[n][1,2], color=colors[n])
# ax.scatter3D(transport.source_distribution[0,0]-transport.source_distribution[1,0], transport.source_distribution[0,1]-transport.source_distribution[1,1],transport.source_distribution[0,2]-transport.source_distribution[1,2],marker="$x$", s=100, color='black')
# ax.scatter3D(transport.source_distribution[1,0], transport.source_distribution[1,1],transport.source_distribution[1,2],marker="$1$", s=100)
# ax.scatter3D(original_source_distribution[0,0], original_source_distribution[0,1],original_source_distribution[0,2],marker="*", s=100)

# ax.scatter3D(gpt.source_distribution[2,0], gpt.source_distribution[2,1],gpt.source_distribution[2,2],marker="$3$", s=100)
for n in range(len(converted_targets_dists)):
    if converted_targets_dists_proj[n][0,1]>0:
        continue
    ax.scatter(converted_targets_dists_proj[n][0,0], converted_targets_dists_proj[n][0,1],marker="x", color=colors[n], s=500, linewidths=4.5)
    # ax.scatter(converted_targets_dists_proj[n][1,0], converted_targets_dists_proj[n][1,1],marker="*", color=colors[n], s=110)
    # ax.annotate('', 
    #         xy=(converted_targets_dists_proj[n][1,0], converted_targets_dists_proj[n][1,1]), 
    #         xytext=(converted_targets_dists_proj[n][0,0], converted_targets_dists_proj[n][0,1]),
    #         arrowprops=dict(arrowstyle='->,head_length=0.9,head_width=0.4', color=colors[n], lw=2))
# ax.scatter3D(original_target_distribution[0,0], original_target_distribution[0,1],original_target_distribution[0,2],marker="*", s=100)

# ax.scatter3D(gpt.target_distribution[2,0], gpt.target_distribution[2,1],gpt.target_distribution[2,2],marker="$3$", s=100)
# ax.scatter3D([], [], [], marker='x', color='black', label='Marker 2')
# ax.scatter3D(0, 0, 0, marker='*', color='black', label='Marker 1', s=100)

# Customize the plot
# ax.set_title('Transported stacking trajectories')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_ylim((-0.45, 0.03))
ax.scatter([],[], marker='x', color='black', label='Initial object location', s=200, linewidths=3)
# ax.set_zlabel('Z axis')
ax.legend()
plt.savefig("/media/mariano/Windows2/phd/hybrid_arxiv_paper/example_plots/dynamic_2d_new.pdf")
plt.savefig("/media/mariano/Windows2/phd/hybrid_arxiv_paper/example_plots/dynamic_2d_new.png")


# Show the plot
# plt.savefig("")
plt.show()
# %%
# Show the plot
plt.show()
# %%
# # Create a 3D plot
fig, axs = plt.subplots(2,1)
# Plot the trajectories
# ax.plot(old_traj[start_idx:end_idx,0]-transport.source_distribution[1,0], old_traj[start_idx:end_idx,1]-transport.source_distribution[1,1], old_traj[start_idx:end_idx,2]-transport.source_distribution[1,2], label='Demonstration', color='black', linestyle='--', linewidth=2)
# for n in range(len(transported_trajs)):
    # ax.plot(transported_trajs[n][start_idx:end_idx,0]-converted_targets_dists_proj[n][1,0], transported_trajs[n][start_idx:end_idx,1]-converted_targets_dists_proj[n][1,1],transported_trajs[n][start_idx:end_idx,2]-converted_targets_dists_proj[n][1,2], color=colors[n])
# ax.scatter3D(transport.source_distribution[0,0]-transport.source_distribution[1,0], transport.source_distribution[0,1]-transport.source_distribution[1,1],transport.source_distribution[0,2]-transport.source_distribution[1,2],marker="$x$", s=100, color='black')
# ax.scatter3D(transport.source_distribution[1,0], transport.source_distribution[1,1],transport.source_distribution[1,2],marker="$1$", s=100)
# ax.scatter3D(original_source_distribution[0,0], original_source_distribution[0,1],original_source_distribution[0,2],marker="*", s=100)

# ax.scatter3D(gpt.source_distribution[2,0], gpt.source_distribution[2,1],gpt.source_distribution[2,2],marker="$3$", s=100)
for n in range(len(converted_targets_dists)):
    axs[1].scatter(converted_targets_dists_proj[n][0,0], converted_targets_dists_proj[n][0,1],marker="x", color=colors[n])
    axs[0].scatter(converted_targets_dists_proj[n][1,0], converted_targets_dists_proj[n][1,1],marker="*", color=colors[n])
    # ax.plot([converted_targets_dists_proj[n][0,0], converted_targets_dists_proj[n][1,0]], [converted_targets_dists_proj[n][0,1], converted_targets_dists_proj[n][1,1]], color=colors[n])
# ax.scatter3D(original_target_distribution[0,0], original_target_distribution[0,1],original_target_distribution[0,2],marker="*", s=100)

# ax.scatter3D(gpt.target_distribution[2,0], gpt.target_distribution[2,1],gpt.target_distribution[2,2],marker="$3$", s=100)
# ax.scatter3D([], [], [], marker='x', color='black', label='Marker 2')
# ax.scatter3D(0, 0, 0, marker='*', color='black', label='Marker 1', s=100)

# Customize the plot
# ax.set_title('Transported stacking trajectories')
axs[1].set_xlabel('x [m]')
axs[0].set_ylabel('y [m]')
axs[1].set_ylabel('y [m]')

axs[0].set_xlim(0.4,0.65)
axs[1].set_xlim(0.4,0.65)
axs[0].set_ylim(-0.4,0.4)
axs[1].set_ylim(-0.4,0.4)
axs[0].set_title("Pick locations")
axs[1].set_title("Place locations")

# ax.set_zlabel('Z axis')
# ax.legend()
  
# Show the plot
plt.show()
# %4# %%

# %%
fig, ax = plt.subplots(figsize=(9,6))
plt.xlabel("Index [-]")
plt.ylabel("Servo actuations [-]")
plt.plot(tentacle_actuations[0])
# %%
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(transported_trajs[0][:,0], transported_trajs[0][:,1], tentacle_actuations[0][:,0])
ax.set_zlabel("Servo actuations [-]")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
# %%
# Create a 3D trajectory
t = tentacle_actuations[0][:,0]
x = transported_trajs[0][:,0]
y = transported_trajs[0][:,1]
z = transported_trajs[0][:,2]

# Normalize the color range to [0, 1] for colormap
colors = plt.cm.viridis((t - t.min()) / (t.max() - t.min()))

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each segment with a different color
for i in range(len(t) - 1):
    ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=colors[i])

# Optionally, add a colorbar
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=t.min(), vmax=t.max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
cbar.set_label('Servo actuation')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.tight_layout()
plt.show()
# %%

#%%
# import pickle

# with open("old_traj.pkl", "wb") as f:
#     pickle.dump(old_traj, f)

# with open("transported_trajs.pkl", "wb") as f:
#     pickle.dump(transported_trajs, f)

# with open("converted_targets_dists.pkl", "wb") as f:
#     pickle.dump(converted_targets_dists, f)

# with open("converted_targets_dists_proj.pkl", "wb") as f:
#     pickle.dump(converted_targets_dists_proj, f)

# with open("source_distribution.pkl", "wb") as f:
#     pickle.dump(transport.source_distribution, f)
#%%
import pickle

with open("old_traj.pkl", "rb") as f:
    old_traj = pickle.load(f)

with open("transported_trajs.pkl", "rb") as f:
    transported_trajs = pickle.load(f)

with open("converted_targets_dists.pkl", "rb") as f:
    converted_targets_dists = pickle.load(f)

with open("converted_targets_dists_proj.pkl", "rb") as f:
    converted_targets_dists_proj = pickle.load(f)

with open("source_distribution.pkl", "rb") as f:
    source_distribution = pickle.load(f)
#%%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, AuxTransformBox, VPacker
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter1d
import numpy as np
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    'font.size': 20,               # Base font size
    'axes.titlesize': 20,          # Title size
    'axes.labelsize': 20,          # X/Y axis labels
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 16})
#%%
class ImageHandler(HandlerBase):
    """
    Draw a PNG (or any RGBA array) centred in the legend handle box.
    """
    def __init__(self, img, zoom=0.5):
        super().__init__()
        self.img  = img
        self.zoom = zoom

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # logical centre of the little handle slot
        x_center = xdescent + width  * 0.5
        y_center = ydescent + height * 0.5

        oi = OffsetImage(self.img, zoom=self.zoom, interpolation='hanning')
        ab = AnnotationBbox(oi, (x_center, y_center),
                            xycoords=trans, frameon=False)
        return [ab]        # ← must return a *list* of artists
#%%
color_values = np.linspace(0, 1, len(converted_targets_dists))  # Generate values from 0 to 1
cmap = plt.get_cmap("tab20")  # Use HSV colormap for distinct colors

colors = cmap(color_values)  # Map colors to colormap
#%%
# # Create a 3D plot
fig = plt.figure(figsize=(9,6), dpi=300)
ax = fig.add_subplot(111, projection='3d')

# rotate labe
start_idx = 40
end_idx = -250
# Plot the trajectories
sigma = 15
ax.plot(old_traj[start_idx:end_idx,0], old_traj[start_idx:end_idx,1], old_traj[start_idx:end_idx,2], label='Demonstration', color='black', linestyle='--', linewidth=3.5)
for n in range(len(transported_trajs)):
    if converted_targets_dists_proj[n][0,1] > 0:
        continue
    ax.plot(gaussian_filter1d(transported_trajs[n][start_idx:end_idx,0], sigma=sigma), gaussian_filter1d(transported_trajs[n][start_idx:end_idx,1], sigma=sigma),gaussian_filter1d(transported_trajs[n][start_idx:end_idx,2], sigma=sigma), color=colors[n], lw=3.5)
# ax.scatter3D(transport.source_distribution[0,0]-transport.source_distribution[1,0], transport.source_distribution[0,1]-transport.source_distribution[1,1],0,marker="$x$", s=300, color='black')
# ax.scatter3D(transport.source_distribution[1,0], transport.source_distribution[1,1],transport.source_distribution[1,2],marker="$1$", s=100)
# ax.scatter3D(original_source_distribution[0,0], original_source_distribution[0,1],original_source_distribution[0,2],marker="*", s=100)

# ax.scatter3D(gpt.source_distribution[2,0], gpt.source_distribution[2,1],gpt.source_distribution[2,2],marker="$3$", s=100)
for n in range(len(converted_targets_dists)):
    if converted_targets_dists_proj[n][0,1] > 0:
        continue
    ax.scatter3D(converted_targets_dists_proj[n][0,0], converted_targets_dists_proj[n][0,1],0.025,marker="x", s=300, color=colors[n], linewidths=4)
    # ax.scatter3D(converted_targets_dists_proj[n][1,0], converted_targets_dists_proj[n][1,1],converted_targets_dists_proj[n][1,2],marker="$1$", s=100, color=colors[n])
# ax.scatter3D(original_target_distribution[0,0], original_target_distribution[0,1],original_target_distribution[0,2],marker="*", s=100)

# ax.scatter3D(gpt.target_distribution[2,0], gpt.target_distribution[2,1],gpt.target_distribution[2,2],marker="$3$", s=100)
ax.scatter3D([], [], [], marker='x', color='black', label='Initial object locations', s=200, linewidths=3)
ax.scatter3D(source_distribution[0,0], source_distribution[0,1], 0.025, marker='x', color='black', s=300, linewidths=4)
ax.view_init(elev=25, azim=-45)  # 30° from horizontal, 45° around z-axis
# Customize the plot
# ax.set_title('Transported stacking trajectories')
ax.set_xlabel(r'$x$ [m]', labelpad=15)
ax.set_ylabel(r'$y$ [m]', labelpad=15)
ax.set_zlabel(r'$z$ [m]', labelpad=10)
ax.set_xlim((0.2,0.6))
ax.set_ylim((-0.39,0.55))
ax.set_zlim((-0.05,0.35))
# --- Helper to place images on scatter plot ---
def add_image_marker(ax, x, y, img, zoom):
    ab = AnnotationBbox(OffsetImage(img, zoom=zoom, interpolation="spline36"), (x, y),
                        frameon=False, xycoords='data')
    ax.add_artist(ab)
img_start = mpimg.imread("/home/mariano/phd_code/red_cylinder2.png")
# add_image_marker(ax, -0.018, -0.041, img_start, 0.2)
ax.legend()
# legend = ax.get_legend()
# handles, labels = legend.legend_handles, [text.get_text() for text in legend.texts]

# start_handle = object()
# handles = list(handles) + [start_handle]
# labels = list(labels) + ['Initial object position']


# # -- build the legend ------------------------------------------------
# ax.legend(handles, labels,
#           handler_map={
#               start_handle: ImageHandler(img_start, zoom=0.19)
#           },
#           frameon=True, loc="upper left")  # pick a loc you like
ax.zaxis.set_ticks_position('lower')
ax.zaxis.set_label_position('lower')
plt.tight_layout()
# plt.savefig("/media/mariano/Windows2/phd/hybrid_arxiv_paper/example_plots/dynamic_3d_new.pdf")
# plt.savefig("/media/mariano/Windows2/phd/hybrid_arxiv_paper/example_plots/dynamic_3d_new.png")
#%%

#%%
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np      # only for dummy data

# ---–  Level 1: outer split (1/3 : 2/3)  –---
fig        = plt.figure(figsize=(9, 6), constrained_layout=True, dpi=300)
subfig_L, subfig_R = fig.subfigures(
    1, 2, width_ratios=[2, 3],  # 1⁄3  :  2⁄3
)

# ---–  Level 2-L: two rows on the left  –---
axs_left = subfig_L.subplots(
    2, 1, sharex=False, sharey=False, height_ratios=[1, 1]
)
img_top = mpimg.imread("/media/mariano/Windows2/phd/hybrid_arxiv_paper/example_plots/dynamic_3d_new.png")
img_top = img_top[50:-80, 300:-450]
axs_left[0].imshow(img_top)
axs_left[0].set_axis_off()

img_bottom = mpimg.imread("/media/mariano/Windows2/phd/hybrid_arxiv_paper/example_plots/dynamic_2d_new.png")
axs_left[1].imshow(img_bottom[40:-100,0:-80])
axs_left[1].set_axis_off()
# axs_left[1].set_title("Bottom plot")

axs_left[0].text(
    -0.0, 1.0, r"$A$", transform=axs_left[0].transAxes,
    fontsize=16, fontweight='bold', va='top', ha='left'
)

axs_left[1].text(
    -0.0, 1.04, r"$B$", transform=axs_left[1].transAxes,
    fontsize=16, fontweight='bold', va='top', ha='left'
)

n_images = 5
# --- Right: 3 subfigures (columns), each with 4 stacked image axes ---
subfigs_cols = subfig_R.subfigures(1, 3, wspace=0.05)

start_frame = 1
imgs = [
    [mpimg.imread(f"/media/mariano/Windows2/phd/hybrid_arxiv_paper/frames_nb/dynamic_grasp/frame{k}.png") for k in range(start_frame,start_frame+n_images)],
    [mpimg.imread(f"/media/mariano/Windows2/phd/hybrid_arxiv_paper/frames_nb/dynamic_grasp2/frame{k}.png") for k in range(start_frame,start_frame+n_images)],
    [mpimg.imread(f"/media/mariano/Windows2/phd/hybrid_arxiv_paper/frames_nb/dynamic_grasp3/frame{k}.png") for k in range(start_frame,start_frame+n_images)],
]

crop_left = 200
crop_right = 100
crop_top = 0
crop_bottom = 30
overlay_alpha = 0.6

column_labels = [r"$C_1$", r"$C_2$", r"$C_3$"]

for i, (subfig_col, imgset, label) in enumerate(zip(subfigs_cols, imgs, column_labels)):
    subfig_col.set_facecolor("lightgray")  # ← add gray background per column
    axs = subfig_col.subplots(n_images, 1, sharex=True, sharey=True, 
                              gridspec_kw={ 'hspace':0.0, 'wspace':0.0})
    
    for row, ax in enumerate(axs):
        base_img = imgset[row][crop_top:-crop_bottom, crop_left:-crop_right]
        ax.imshow(base_img, zorder=3)

        if row > 0:
            overlay = imgset[row-1][crop_top:-crop_bottom, crop_left:-crop_right]
            ax.imshow(overlay, alpha=overlay_alpha, zorder=2)

        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Label the top axis
    axs[0].text(
        0.5, 1.04, label, transform=axs[0].transAxes,
        fontsize=16, fontweight='bold', ha='center', va='bottom'
    )

plt.savefig("/media/mariano/Windows2/phd/hybrid_arxiv_paper/example_plots/full_figure_dynamic.pdf")
# %%
