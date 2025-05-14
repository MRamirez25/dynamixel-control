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
n_trajs = 15
for i in range(15):
    if skill_name == 'stack_two':
        if i == 12 or i == 18 or i == 19:
            continue
    with open(f'data/{skill_name}/target_{i}.pkl', 'rb') as f:
        marker_data = pickle.load(f)
        target_dist_list.append(marker_data)
# %%
transported_trajs = []
converted_tragets_dists = []
converted_targets_dists_proj = []
tentacle_actuations = []
for i in range(n_trajs):
    converted_source_dist, converted_target_dist, dist = convert_distribution(source_distribution, target_dist_list[i])
    converted_tragets_dists.append(converted_target_dist)
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
# # Create a 3D plot
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')
old_traj = transport.training_traj_old
start_idx = 100
end_idx = -150
# Plot the trajectories
ax.plot(old_traj[start_idx:end_idx,0]-transport.source_distribution[1,0], old_traj[start_idx:end_idx,1]-transport.source_distribution[1,1], old_traj[start_idx:end_idx,2]-transport.source_distribution[1,2], label='Demonstration', color='black', linestyle='--', linewidth=2)
for n in range(len(transported_trajs)):
    ax.plot(transported_trajs[n][start_idx:end_idx,0]-converted_targets_dists_proj[n][1,0], transported_trajs[n][start_idx:end_idx,1]-converted_targets_dists_proj[n][1,1],transported_trajs[n][start_idx:end_idx,2]-converted_targets_dists_proj[n][1,2], color=colors[n])
ax.scatter3D(transport.source_distribution[0,0]-transport.source_distribution[1,0], transport.source_distribution[0,1]-transport.source_distribution[1,1],0,marker="$x$", s=120, color='black')
# ax.scatter3D(transport.source_distribution[1,0], transport.source_distribution[1,1],transport.source_distribution[1,2],marker="$1$", s=100)
# ax.scatter3D(original_source_distribution[0,0], original_source_distribution[0,1],original_source_distribution[0,2],marker="*", s=100)

# ax.scatter3D(gpt.source_distribution[2,0], gpt.source_distribution[2,1],gpt.source_distribution[2,2],marker="$3$", s=100)
for n in range(len(converted_tragets_dists)):
    ax.scatter3D(converted_targets_dists_proj[n][0,0]-converted_targets_dists_proj[n][1,0], converted_targets_dists_proj[n][0,1]-converted_targets_dists_proj[n][1,1],0,marker="x", s=120, color=colors[n])
    # ax.scatter3D(converted_targets_dists_proj[n][1,0], converted_targets_dists_proj[n][1,1],converted_targets_dists_proj[n][1,2],marker="$1$", s=100, color=colors[n])
# ax.scatter3D(original_target_distribution[0,0], original_target_distribution[0,1],original_target_distribution[0,2],marker="*", s=100)

# ax.scatter3D(gpt.target_distribution[2,0], gpt.target_distribution[2,1],gpt.target_distribution[2,2],marker="$3$", s=100)
ax.scatter3D(0, 0, 0, marker='*', color='black', label='Initial object locations', s=100)
ax.scatter3D([], [], [], marker='x', color='black', label='Goal object locations', s=100)

# Customize the plot
# ax.set_title('Transported stacking trajectories')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.legend()

# Show the plot
plt.show()
# %%
# # Create a 3D plot
fig, ax = plt.subplots(figsize=(9,6))
# Plot the trajectories
# ax.plot(old_traj[start_idx:end_idx,0]-transport.source_distribution[1,0], old_traj[start_idx:end_idx,1]-transport.source_distribution[1,1], old_traj[start_idx:end_idx,2]-transport.source_distribution[1,2], label='Demonstration', color='black', linestyle='--', linewidth=2)
# for n in range(len(transported_trajs)):
    # ax.plot(transported_trajs[n][start_idx:end_idx,0]-converted_targets_dists_proj[n][1,0], transported_trajs[n][start_idx:end_idx,1]-converted_targets_dists_proj[n][1,1],transported_trajs[n][start_idx:end_idx,2]-converted_targets_dists_proj[n][1,2], color=colors[n])
# ax.scatter3D(transport.source_distribution[0,0]-transport.source_distribution[1,0], transport.source_distribution[0,1]-transport.source_distribution[1,1],transport.source_distribution[0,2]-transport.source_distribution[1,2],marker="$x$", s=100, color='black')
# ax.scatter3D(transport.source_distribution[1,0], transport.source_distribution[1,1],transport.source_distribution[1,2],marker="$1$", s=100)
# ax.scatter3D(original_source_distribution[0,0], original_source_distribution[0,1],original_source_distribution[0,2],marker="*", s=100)

# ax.scatter3D(gpt.source_distribution[2,0], gpt.source_distribution[2,1],gpt.source_distribution[2,2],marker="$3$", s=100)
for n in range(len(converted_tragets_dists)):
    ax.scatter(converted_targets_dists_proj[n][0,0], converted_targets_dists_proj[n][0,1],marker="x", color=colors[n], s=110)
    ax.scatter(converted_targets_dists_proj[n][1,0], converted_targets_dists_proj[n][1,1],marker="*", color=colors[n], s=110)
    ax.annotate('', 
            xy=(converted_targets_dists_proj[n][1,0], converted_targets_dists_proj[n][1,1]), 
            xytext=(converted_targets_dists_proj[n][0,0], converted_targets_dists_proj[n][0,1]),
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
for n in range(len(converted_tragets_dists)):
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
