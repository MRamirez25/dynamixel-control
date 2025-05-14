#%%
from policy_transportation import GaussianProcessTransportation as Transport
from tag_detector import convert_distribution
from sklearn.gaussian_process.kernels import RBF, Matern,WhiteKernel, ConstantKernel as C
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
%matplotlib qt
#%%
kernel_transport=C(0.1) * RBF(length_scale=[0.1,0.1,0.1],  length_scale_bounds=[0.1,0.1]) + WhiteKernel(0.0000001, [0.0000001,0.0000001])
gpt=Transport(kernel_transport)
#%%
# f = open("data/source.pkl", "rb")
# source = pickle.load(f)
# f.close()
# %%
f = open("data/traj_demo.pkl", "rb")
demo_data = pickle.load(f)
f.close()
#%%
f = open("data/source.pkl", "rb")
source_distribution = pickle.load(f)
f.close()
#%%
f = open("data/target_2.pkl", "rb")
target_distribution = pickle.load(f)
f.close()
#%%
gpt.training_traj = demo_data
#%%
source_distribution, target_distribution, dist = convert_distribution(source_distribution, target_distribution, use_orientation=False)
#%%
#%%
distances = np.empty((len(gpt.training_traj), 0))
for i in range((source_distribution.shape[0])):
    distance_keypoint_i = np.linalg.norm(gpt.training_traj - source_distribution[i], keepdims=True, axis=-1)
    distances = np.hstack((distances, distance_keypoint_i))
#%%
closest_points_idxs = np.argmin(distances, axis=0)
projected_source_distribution = gpt.training_traj[closest_points_idxs]
gpt.source_distribution = projected_source_distribution
#%%
source_target_diff = target_distribution - source_distribution
projected_target_distribution = projected_source_distribution + source_target_diff
gpt.target_distribution = projected_target_distribution
# gpt.target_distribution = source_distribution + np.random.randint(-2,2, size=(3,2))
#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(demo_data[:,0], demo_data[:,1], demo_data[:,2])
for i in range(source_distribution.shape[0]):
    ax.scatter(source_distribution[i,0], source_distribution[i,1], source_distribution[i,2], marker='*')
for i in range(source_distribution.shape[0]):
    ax.scatter(projected_source_distribution[i,0], projected_source_distribution[i,1], projected_source_distribution[i,2], marker='x', s=100)
#%%
gpt.fit_transportation(do_rotation=False, do_scale=False)
gpt.apply_transportation()
#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(gpt.training_traj[:,0], gpt.training_traj[:,1], gpt.training_traj[:,2])
for i in range(target_distribution.shape[0]):
    ax.scatter(target_distribution[i,0], target_distribution[i,1], target_distribution[i,2], marker='*')
for i in range(target_distribution.shape[0]):
    ax.scatter(projected_target_distribution[i,0], projected_target_distribution[i,1], projected_target_distribution[i,2], marker='x', s=100)
#%%
plt.plot(gpt.training_traj[:,0], gpt.training_traj[:,1])
for i in range(len(frame_keys)):
    plt.scatter(gpt.source_distribution[i,0], gpt.source_distribution[i,1], marker='*', s=100)
    plt.scatter(shifted_source_distribution[i,0], shifted_source_distribution[i,1], marker='x', s=100)

#%%
# %%
plt.plot(gpt.training_traj[:,0], gpt.training_traj[:,1])
for i in range(len(frame_keys)):
    plt.scatter(gpt.target_distribution[i,0], gpt.target_distribution[i,1], marker='*', s=100)
    plt.scatter(target_distribution[i,0], target_distribution[i,1], marker='x', s=100)
# %%
