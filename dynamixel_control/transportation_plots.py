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
from scipy.ndimage import gaussian_filter1d
# %%
transport = Transport()
#%%
skill_name = "stack_two"
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
n_trajs = 20
for i in range(25):
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
    "font.serif": ["Times New Roman"],
    'font.size': 20,               # Base font size
    'axes.titlesize': 20,          # Title size
    'axes.labelsize': 20,          # X/Y axis labels
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 16})
#%%
# # Create a 3D plot
fig = plt.figure(figsize=(9,6))
axs=[]
for n in range(9):
    temp_ax = fig.add_subplot(3,3,n+1, projection='3d')
    axs.append(temp_ax)
# ax = fig.add_subplot(331, projection='3d')
old_traj = transport.training_traj_old
start_idx = 200
end_idx = 800
# Plot the trajectories
# ax.plot(old_traj[start_idx:end_idx,0], old_traj[start_idx:end_idx,1], old_traj[start_idx:end_idx,2], label='Demonstration', color='black', linestyle='--', linewidth=2)
for n in range(9):
    axs[n].plot(transported_trajs[n][start_idx:end_idx,0], transported_trajs[n][start_idx:end_idx,1],transported_trajs[n][start_idx:end_idx,2], color=colors[n])
# ax.scatter3D(transport.source_distribution[0,0], transport.source_distribution[0,1],transport.source_distribution[0,2],marker="$x$", s=100, color='black')
# ax.scatter3D(transport.source_distribution[1,0], transport.source_distribution[1,1],transport.source_distribution[1,2],marker="$1$", s=100)
# ax.scatter3D(original_source_distribution[0,0], original_source_distribution[0,1],original_source_distribution[0,2],marker="*", s=100)

# ax.scatter3D(gpt.source_distribution[2,0], gpt.source_distribution[2,1],gpt.source_distribution[2,2],marker="$3$", s=100)
for n in range(9):
    axs[n].scatter3D(converted_targets_dists_proj[n][0,0], converted_targets_dists_proj[n][0,1],converted_targets_dists_proj[n][0,2]-converted_targets_dists_proj[n][1,2],marker="x", s=100, color=colors[n])
    axs[n].scatter3D(converted_targets_dists_proj[n][1,0], converted_targets_dists_proj[n][1,1],converted_targets_dists_proj[n][1,2],marker="*", s=100, color=colors[n])
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
import pickle

with open("old_traj.pkl", "wb") as f:
    pickle.dump(old_traj, f)

with open("transported_trajs.pkl", "wb") as f:
    pickle.dump(transported_trajs, f)

with open("converted_targets_dists.pkl", "wb") as f:
    pickle.dump(converted_targets_dists, f)

with open("converted_targets_dists_proj.pkl", "wb") as f:
    pickle.dump(converted_targets_dists_proj, f)

with open("source_distribution.pkl", "wb") as f:
    pickle.dump(transport.source_distribution, f)
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
# # Create a 3D plot
fig = plt.figure(figsize=(9,6), dpi=300)
ax = fig.add_subplot(111, projection='3d')

# rotate labe
start_idx = 100
end_idx = -150
# Plot the trajectories
sigma = 15
ax.plot(old_traj[start_idx:end_idx,0]-source_distribution[1,0], old_traj[start_idx:end_idx,1]-source_distribution[1,1], old_traj[start_idx:end_idx,2]-source_distribution[1,2], label='Demonstration', color='black', linestyle='--', linewidth=2)
for n in range(len(transported_trajs)):
    ax.plot(gaussian_filter1d(transported_trajs[n][start_idx:end_idx,0]-converted_targets_dists_proj[n][1,0], sigma=sigma), gaussian_filter1d(transported_trajs[n][start_idx:end_idx,1]-converted_targets_dists_proj[n][1,1], sigma=sigma),gaussian_filter1d(transported_trajs[n][start_idx:end_idx,2]-converted_targets_dists_proj[n][1,2], sigma=sigma), color=colors[n])
# ax.scatter3D(transport.source_distribution[0,0]-transport.source_distribution[1,0], transport.source_distribution[0,1]-transport.source_distribution[1,1],0,marker="$x$", s=120, color='black')
# ax.scatter3D(transport.source_distribution[1,0], transport.source_distribution[1,1],transport.source_distribution[1,2],marker="$1$", s=100)
# ax.scatter3D(original_source_distribution[0,0], original_source_distribution[0,1],original_source_distribution[0,2],marker="*", s=100)

# ax.scatter3D(gpt.source_distribution[2,0], gpt.source_distribution[2,1],gpt.source_distribution[2,2],marker="$3$", s=100)
for n in range(len(converted_targets_dists)):
    ax.scatter3D(converted_targets_dists_proj[n][0,0]-converted_targets_dists_proj[n][1,0], converted_targets_dists_proj[n][0,1]-converted_targets_dists_proj[n][1,1],0,marker="x", s=120, color=colors[n])
    # ax.scatter3D(converted_targets_dists_proj[n][1,0], converted_targets_dists_proj[n][1,1],converted_targets_dists_proj[n][1,2],marker="$1$", s=100, color=colors[n])
# ax.scatter3D(original_target_distribution[0,0], original_target_distribution[0,1],original_target_distribution[0,2],marker="*", s=100)

# ax.scatter3D(gpt.target_distribution[2,0], gpt.target_distribution[2,1],gpt.target_distribution[2,2],marker="$3$", s=100)
# ax.scatter3D(0, 0, 0, marker='*', color='black', label='Initial object locations', s=100)
ax.scatter3D([], [], [], marker='x', color='black', label='Goal object location', s=100)
ax.view_init(elev=25, azim=-45)  # 30° from horizontal, 45° around z-axis
# Customize the plot
# ax.set_title('Transported stacking trajectories')
ax.set_xlabel(r'$x - t_x^1$ [m]', labelpad=15)
ax.set_ylabel(r'$y - t_y^1$ [m]', labelpad=15)
ax.set_zlabel(r'$z - t_z^1$ [m]', labelpad=10)
ax.set_xlim((-0.25,0.2))
ax.set_ylim((-0.7,0.6))
ax.set_zlim((-0.05,0.3))
# --- Helper to place images on scatter plot ---
def add_image_marker(ax, x, y, img, zoom):
    ab = AnnotationBbox(OffsetImage(img, zoom=zoom, interpolation="spline36"), (x, y),
                        frameon=False, xycoords='data')
    ax.add_artist(ab)
img_start = mpimg.imread("/home/mariano/phd_code/red_cylinder2.png")
add_image_marker(ax, 0.008, -0.026, img_start, 0.2)
ax.legend()
legend = ax.get_legend()
handles, labels = legend.legend_handles, [text.get_text() for text in legend.texts]

start_handle = object()
handles = list(handles) + [start_handle]
labels = list(labels) + ['Initial object position']


# -- build the legend ------------------------------------------------
ax.legend(handles, labels,
          handler_map={
              start_handle: ImageHandler(img_start, zoom=0.19)
          },
          frameon=True, loc="upper left")  # pick a loc you like
ax.zaxis.set_ticks_position('lower')
ax.zaxis.set_label_position('lower')
plt.tight_layout()
plt.savefig("/media/mariano/Windows2/phd/hybrid_arxiv_paper/example_plots/stack_3d_new.pdf")
plt.savefig("/media/mariano/Windows2/phd/hybrid_arxiv_paper/example_plots/stack_3d_new.png")

# Show the plot
plt.show()

# %%
# # Create 2D plot
fig, ax = plt.subplots(figsize=(9,6), dpi=300)
# Plot the trajectories
# ax.plot(old_traj[start_idx:end_idx,0]-transport.source_distribution[1,0], old_traj[start_idx:end_idx,1]-transport.source_distribution[1,1], old_traj[start_idx:end_idx,2]-transport.source_distribution[1,2], label='Demonstration', color='black', linestyle='--', linewidth=2)
# for n in range(len(transported_trajs)):
    # ax.plot(transported_trajs[n][start_idx:end_idx,0]-converted_targets_dists_proj[n][1,0], transported_trajs[n][start_idx:end_idx,1]-converted_targets_dists_proj[n][1,1],transported_trajs[n][start_idx:end_idx,2]-converted_targets_dists_proj[n][1,2], color=colors[n])
# ax.scatter3D(transport.source_distribution[0,0]-transport.source_distribution[1,0], transport.source_distribution[0,1]-transport.source_distribution[1,1],transport.source_distribution[0,2]-transport.source_distribution[1,2],marker="$x$", s=100, color='black')
# ax.scatter3D(transport.source_distribution[1,0], transport.source_distribution[1,1],transport.source_distribution[1,2],marker="$1$", s=100)
# ax.scatter3D(original_source_distribution[0,0], original_source_distribution[0,1],original_source_distribution[0,2],marker="*", s=100)

# ax.scatter3D(gpt.source_distribution[2,0], gpt.source_distribution[2,1],gpt.source_distribution[2,2],marker="$3$", s=100)
for n in range(len(converted_targets_dists)):
    ax.scatter(converted_targets_dists[n][0,0], converted_targets_dists[n][0,1],marker="*", color=colors[n], s=150)
    ax.scatter(converted_targets_dists[n][1,0], converted_targets_dists[n][1,1],marker="x", color=colors[n], s=150)
    ax.annotate('', 
            xy=(converted_targets_dists[n][0,0], converted_targets_dists[n][0,1]), 
            xytext=(converted_targets_dists[n][1,0], converted_targets_dists[n][1,1]),
            arrowprops=dict(arrowstyle='->,head_length=0.9,head_width=0.4', color=colors[n], lw=2))
# ax.scatter3D(original_target_distribution[0,0], original_target_distribution[0,1],original_target_distribution[0,2],marker="*", s=100)

# ax.scatter3D(gpt.target_distribution[2,0], gpt.target_distribution[2,1],gpt.target_distribution[2,2],marker="$3$", s=100)
# ax.scatter3D([], [], [], marker='x', color='black', label='Marker 2')
# ax.scatter3D(0, 0, 0, marker='*', color='black', label='Marker 1', s=100)

# Customize the plot
# ax.set_title('Transported stacking trajectories')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.scatter([],[], marker='x', color='black', label='Initial object locations')
ax.scatter([], [], marker='*', color='black', label='Goal object locations')
# ax.set_zlabel('Z axis')
ax.legend()

# Show the plot
plt.savefig("/media/mariano/Windows2/phd/hybrid_arxiv_paper/example_plots/plot_dpi300.pdf")

# Show the plot
plt.show()
#%%
###################### New 2d plot
#%%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, AuxTransformBox, VPacker
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle

# --- Load images ---
img_goal = mpimg.imread("/home/mariano/phd_code/yellow_cylinder.png")
img_start = mpimg.imread("/home/mariano/phd_code/red_cylinder2.png")

ZOOM = 0.2  # Tweak as needed

# --- Helper to place images on scatter plot ---
def add_image_marker(ax, x, y, img, zoom):
    ab = AnnotationBbox(OffsetImage(img, zoom=zoom, interpolation="spline36"), (x, y),
                        frameon=False, xycoords='data')
    ax.add_artist(ab)

def add_image_marker3d(ax, x, y, z, img, zoom):
    ab = AnnotationBbox(OffsetImage(img, zoom=zoom, interpolation="spline36"), (x, y, z),
                        frameon=False, xycoords='data')
    ax.add_artist(ab)

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

# --- Plot ---
# --- normal line / scatter work first --------------------------------
dpi=300
fig, ax = plt.subplots(figsize=(9, 6), dpi=dpi)
ax.set_xlim((0.325,0.6))
ax.set_ylim((-0.4,0.4))
ax.set_autoscale_on(False)        # <- stop autoscaling
# xlim0, ylim0 = ax.get_xlim(), ax.get_ylim()    # ① remember limits

# --- now drop the PNGs ------------------------------------------------
for n in range(len(converted_targets_dists)):
    gx, gy = converted_targets_dists[n][0][:-1]      # goal
    sx, sy = converted_targets_dists[n][1][:-1]      # start
    add_image_marker(ax, gx, gy, img_goal, ZOOM-0.01)
    add_image_marker(ax, sx, sy, img_start, (ZOOM+0.01))
    start = converted_targets_dists[n][1][:-1]
    goal = converted_targets_dists[n][0][:-1]

    # Compute direction unit vector
    direction = goal[:2] - start[:2]
    norm = np.linalg.norm(direction)
    if norm != 0:
        direction = direction / norm
    else:
        direction = np.array([0.0, 0.0])

    # Shorten arrow by a small amount (e.g., 0.1 units)
    goal_arrow = goal[:2] - direction * 0.005

    # Draw arrow to shifted point
    ax.annotate('', 
                xy=goal_arrow, 
                xytext=start[:2],
                arrowprops=dict(arrowstyle="->,head_length=0.6,head_width=0.3", lw=3, color=colors[n]), zorder=10)
    # ax.scatter(converted_targets_dists[n][0,0], converted_targets_dists[n][0,1],marker="*", color=colors[n], s=150, zorder=20)
    # ax.scatter(converted_targets_dists[n][1,0], converted_targets_dists[n][1,1],marker="x", color=colors[n], s=150, zorder=20)

# ax.set_xlim(xlim0); ax.set_ylim(ylim0)          # ② restore limits

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")

# -- dummy handles ---------------------------------------------------
goal_handle  = object()   # any hashable object is fine
start_handle = object()

# -- build the legend ------------------------------------------------
ax.legend([start_handle, goal_handle, ],
          ["Initial object position", "Goal stack position", ],
          handler_map={
              goal_handle:  ImageHandler(img_goal,  zoom=0.18),
              start_handle: ImageHandler(img_start, zoom=0.19)
          },
          frameon=True, loc="upper left")  # pick a loc you like

plt.tight_layout()
# plt.savefig(f"/media/mariano/Windows2/phd/hybrid_arxiv_paper/example_plots/stack_2d_new_dpi{dpi}.pdf")
# plt.savefig(f"/media/mariano/Windows2/phd/hybrid_arxiv_paper/example_plots/stack_2d_new_dpi{dpi}.png")

# plt.show()
#%%
img = img_start
fig, ax = plt.subplots()
for i, interp in enumerate(['nearest', 'bilinear', 'hanning', 'lanczos']):
    imbox = OffsetImage(img, zoom=0.4, interpolation=interp)
    ab = AnnotationBbox(imbox, (i, 0), frameon=True)
    ax.add_artist(ab)
    ax.text(i, -0.2, interp, ha='center')
ax.set_xlim(-1, 4)
ax.set_ylim(-1, 1)
plt.axis('off')
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
img_top = mpimg.imread("/media/mariano/Windows2/phd/hybrid_arxiv_paper/example_plots/stack_3d_new.png")
img_top = img_top[100:-100, 500:-350]
axs_left[0].imshow(img_top)
axs_left[0].set_axis_off()

img_bottom = mpimg.imread("/media/mariano/Windows2/phd/hybrid_arxiv_paper/example_plots/stack_2d_new_dpi300.png")
axs_left[1].imshow(img_bottom[100:-80,100:-100])
axs_left[1].set_axis_off()
# axs_left[1].set_title("Bottom plot")

axs_left[0].text(
    -0.05, 1.02, r"$A$", transform=axs_left[0].transAxes,
    fontsize=16, fontweight='bold', va='top', ha='left'
)

axs_left[1].text(
    -0.05, 1.02, r"$B$", transform=axs_left[1].transAxes,
    fontsize=16, fontweight='bold', va='top', ha='left'
)

# ---–  Level 2-R: 4 rows × 3 cols on the right  –---
axs_right = subfig_R.subplots(4, 3, sharex=True, sharey=True, 
                              gridspec_kw={'wspace': 0.1, 'hspace': 0.05})
imgs = [
    [mpimg.imread(f"/media/mariano/Windows2/phd/hybrid_arxiv_paper/frames_nb/stack1_frames/frame{k}.png") for k in range(1,5)],      # column 0: 4 RGBA images
    [mpimg.imread(f"/media/mariano/Windows2/phd/hybrid_arxiv_paper/frames_nb/stack2_frames/frame{k}.png") for k in range(1,5)],      # column 1
    [mpimg.imread(f"/media/mariano/Windows2/phd/hybrid_arxiv_paper/frames_nb/stacks_left_right2/frame{k}.png") for k in range(1,5)],      # column 2
]

# ---------------------------------------------------------------------------
# ➤ 2.  Display them with the overlay rule you specified:
#     row 0: img0
#     row 1: img1  + faded (img0)
#     row 2: img2  + faded (img1)
#     row 3: img3  + faded (img2)
# ---------------------------------------------------------------------------
overlay_alpha = 0.6        # transparency of the faded layer
crop_left = 600
crop_right = 250
crop_top = 0
crop_bottom = 50
for col in range(3):                    # for each column
    for row in range(4):                # for each row
        ax = axs_right[row, col]
        ax.imshow(imgs[col][row][crop_top:-crop_bottom, crop_left:-crop_right], zorder=1)       # base image

        if row > 0:      
            overlaid_img = imgs[col][row-1][crop_top:-(crop_bottom), crop_left:-crop_right]  # rows 1-3: add previous image on top
            ax.imshow(overlaid_img, alpha=overlay_alpha)

        # ax.axis('off')                  # no ticks / frame
column_labels = [r"$C_1$", r"$C_2$", r"$C_3$"]
for i, label in enumerate(column_labels):
    axs_right[0, i].text(
        0.5, 1.04, label, transform=axs_right[0, i].transAxes,
        fontsize=16, fontweight='bold', ha='center', va='bottom'
    )
    

for ax in axs_right.flat:
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.savefig("/media/mariano/Windows2/phd/hybrid_arxiv_paper/example_plots/full_figure_stack.pdf")
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
img_top = mpimg.imread("/media/mariano/Windows2/phd/hybrid_arxiv_paper/example_plots/stack_3d_new.png")
img_top = img_top[100:-100, 350:-450]
axs_left[0].imshow(img_top)
axs_left[0].set_axis_off()

img_bottom = mpimg.imread("/media/mariano/Windows2/phd/hybrid_arxiv_paper/example_plots/stack_2d_new_dpi300.png")
axs_left[1].imshow(img_bottom[90:-50,50:-10])
axs_left[1].set_axis_off()
# axs_left[1].set_title("Bottom plot")

axs_left[0].text(
    -0.02, 1.02, r"$A$", transform=axs_left[0].transAxes,
    fontsize=16, fontweight='bold', va='top', ha='left'
)

axs_left[1].text(
    -0.02, 1.04, r"$B$", transform=axs_left[1].transAxes,
    fontsize=16, fontweight='bold', va='top', ha='left'
)

# --- Right: 3 subfigures (columns), each with 4 stacked image axes ---
subfigs_cols = subfig_R.subfigures(1, 3, wspace=0.05)

imgs = [
    [mpimg.imread(f"/media/mariano/Windows2/phd/hybrid_arxiv_paper/frames_nb/stack1_frames/frame{k}.png") for k in range(1,5)],
    [mpimg.imread(f"/media/mariano/Windows2/phd/hybrid_arxiv_paper/frames_nb/stack2_frames/frame{k}.png") for k in range(1,5)],
    [mpimg.imread(f"/media/mariano/Windows2/phd/hybrid_arxiv_paper/frames_nb/stacks_left_right2/frame{k}.png") for k in range(1,5)],
]

crop_left = 600
crop_right = 250
crop_top = 0
crop_bottom = 50
overlay_alpha = 0.6

column_labels = [r"$C_1$", r"$C_2$", r"$C_3$"]

for i, (subfig_col, imgset, label) in enumerate(zip(subfigs_cols, imgs, column_labels)):
    subfig_col.set_facecolor("lightgray")  # ← add gray background per column
    axs = subfig_col.subplots(4, 1, sharex=True, sharey=True, 
                              gridspec_kw={'hspace': 0.05})
    
    for row, ax in enumerate(axs):
        base_img = imgset[row][crop_top:-crop_bottom, crop_left:-crop_right]
        ax.imshow(base_img, zorder=1)

        if row > 0:
            overlay = imgset[row-1][crop_top:-crop_bottom, crop_left:-crop_right]
            ax.imshow(overlay, alpha=overlay_alpha, zorder=2)

        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Label the top axis
    axs[0].text(
        0.5, 1.04, label, transform=axs[0].transAxes,
        fontsize=16, fontweight='bold', ha='center', va='bottom'
    )

plt.savefig("/media/mariano/Windows2/phd/hybrid_arxiv_paper/example_plots/full_figure_stack.pdf")
#%%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox

# Create the full figure and subfigures
fig = plt.figure(figsize=(10, 5), constrained_layout=True)
subfig_L, subfig_R = fig.subfigures(1, 2, width_ratios=[1, 2])

# Left side plots (just filler)
axs_left = subfig_L.subplots(2, 1)
for ax in axs_left:
    ax.plot([0, 1], [0, 1])

# Right side: 4 rows x 3 columns of image axes
axs_right = subfig_R.subplots(4, 3, sharex=True, sharey=True)
for row in axs_right:
    for ax in row:
        ax.imshow(np.random.rand(10, 10))
        ax.set_xticks([]); ax.set_yticks([])

# >>> FORCE LAYOUT to finalize positions <<<
fig.canvas.draw()
renderer = fig.canvas.get_renderer()

# Add background rectangles behind each column
for i in range(3):
    ax_top = axs_right[0][i]
    ax_bot = axs_right[-1][i]

    # Get tight bounding boxes in figure pixels
    bbox_top = ax_top.get_tightbbox(renderer)
    bbox_bot = ax_bot.get_tightbbox(renderer)

    # Union of top and bottom axes for this column
    bbox_col = Bbox.union([bbox_top, bbox_bot])

    # Convert to figure-relative (0-1) coords
    bbox_fig = bbox_col.transformed(fig.transFigure.inverted())

    # Add light gray background rectangle in figure coords
    rect = Rectangle(
        (bbox_fig.x0, bbox_fig.y0),
        bbox_fig.width, bbox_fig.height,
        transform=fig.transFigure,
        facecolor='black',
        alpha=0.8,
        zorder=0  # send to background
    )
    fig.patches.append(rect)

plt.show()
print(f"Column {i}: x={bbox_fig.x0:.2f}, y={bbox_fig.y0:.2f}, w={bbox_fig.width:.2f}, h={bbox_fig.height:.2f}")
#%%
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg
import numpy as np  # only for dummy data if needed

# --- Level 1: outer split (2/5 : 3/5) ---
fig = plt.figure(figsize=(9, 6), constrained_layout=True, dpi=300)
subfig_L, subfig_R = fig.subfigures(1, 2, width_ratios=[2, 3])  # same as before

# --- Level 2-L: two rows on the left ---
axs_left = subfig_L.subplots(2, 1, sharex=False, sharey=False, height_ratios=[1, 1])

img_top = mpimg.imread("/media/mariano/Windows2/phd/hybrid_arxiv_paper/example_plots/stack_3d_new.png")
img_top = img_top[100:-100, 500:-310]
axs_left[0].imshow(img_top)
axs_left[0].set_axis_off()

img_bottom = mpimg.imread("/media/mariano/Windows2/phd/hybrid_arxiv_paper/example_plots/stack_2d_new_dpi300.png")
axs_left[1].imshow(img_bottom[100:-80, 100:-90])
axs_left[1].set_axis_off()

axs_left[0].text(
    -0.05, 1.02, r"$A$", transform=axs_left[0].transAxes,
    fontsize=16, fontweight='bold', va='top', ha='left'
)
axs_left[1].text(
    -0.05, 1.02, r"$B$", transform=axs_left[1].transAxes,
    fontsize=16, fontweight='bold', va='top', ha='left'
)

# --- Level 2-R: Create 3 subfigures, one per column ---
col_subfigs = subfig_R.subfigures(1, 3, wspace=0.05)

# Load images (same as your original)
imgs = [
    [mpimg.imread(f"/media/mariano/Windows2/phd/hybrid_arxiv_paper/frames_nb/stack1_frames/frame{k}.png") for k in range(1, 5)],
    [mpimg.imread(f"/media/mariano/Windows2/phd/hybrid_arxiv_paper/frames_nb/stack2_frames/frame{k}.png") for k in range(1, 5)],
    [mpimg.imread(f"/media/mariano/Windows2/phd/hybrid_arxiv_paper/frames_nb/stacks_left_right2/frame{k}.png") for k in range(1, 5)],
]

overlay_alpha = 0.6
crop_left = 600
crop_right = 250
crop_top = 0
crop_bottom = 50

for col_idx, subfig_col in enumerate(col_subfigs):
    # Light gray background for each column subfigure
    subfig_col.patch.set_facecolor('lightgray')
    subfig_col.patch.set_alpha(0.)

    # 4 rows, 1 column inside each subfigure
    axs = subfig_col.subplots(4, 1, sharex=True, sharey=True,
                              gridspec_kw={'hspace': 0.05})
    
    for row_idx, ax in enumerate(axs):
        base_img = imgs[col_idx][row_idx][crop_top:-crop_bottom, crop_left:-crop_right]
        ax.imshow(base_img, zorder=1)

        if row_idx > 0:
            overlay_img = imgs[col_idx][row_idx - 1][crop_top:-crop_bottom, crop_left:-crop_right]
            ax.imshow(overlay_img, alpha=overlay_alpha)

        ax.set_frame_on(True)              # ensures the axes frame is shown
        ax.tick_params(
            left=False, right=False,
            bottom=False, top=False,
            labelleft=False, labelbottom=False
)

    # Add column label at top center inside the subfigure
    axs[0].text(
        0.5, 1.04, f"$C_{{{col_idx + 1}}}$",
        transform=axs[0].transAxes,
        fontsize=16, fontweight='bold', ha='center', va='bottom'
    )
plt.savefig("/media/mariano/Windows2/phd/hybrid_arxiv_paper/example_plots/full_figure_stack_subfigures.pdf")
plt.show()
#%%