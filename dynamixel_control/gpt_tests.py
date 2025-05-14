#%%
from policy_transportation import GaussianProcessTransportation as Transport
from sklearn.gaussian_process.kernels import RBF, Matern,WhiteKernel, ConstantKernel as C
import pickle
import numpy as np
import matplotlib.pyplot as plt
#%%
kernel_transport=C(0.1) * RBF(length_scale=[30,30],  length_scale_bounds=[30,30]) + WhiteKernel(0.0000001, [0.0000001,0.0000001])
gpt=Transport(kernel_transport)
#%%
# f = open("data/source.pkl", "rb")
# source = pickle.load(f)
# f.close()
# %%
f = open("2d_data/three_frame_paper_1.pickle", "rb")
demo_data = pickle.load(f)
f.close()
#%%
frame_keys = ['blue_rect', 'red_rect', 'green_rect']
#%%
source_distribution = np.empty((len(frame_keys),2))
for i, key in enumerate(frame_keys):
    source_distribution[i,:] = demo_data['locations'][key]
#%%
shifted_source_distribution = source_distribution + np.array([[10,0], [0,10], [10,0]])
#%%
gpt.training_traj = demo_data['traj']
#%%
distances = np.empty((len(gpt.training_traj), 0))
for i, frame_key in enumerate(frame_keys):
    distance_frame_i = np.linalg.norm(gpt.training_traj - shifted_source_distribution[i], keepdims=True, axis=-1)
    distances = np.hstack((distances, distance_frame_i))
#%%
closest_points_idxs = np.argmin(distances, axis=0)
projected_source_distribution = gpt.training_traj[closest_points_idxs]
gpt.source_distribution = projected_source_distribution
#%%
# f = open("data/target.pkl", "rb")
# target = pickle.load(f)
# f.close()
target_distribution = np.random.randint(-30, 30, size=(3,2))
#%%
source_target_diff = target_distribution - shifted_source_distribution
projected_target_distribution = projected_source_distribution + source_target_diff
gpt.target_distribution = projected_target_distribution
# gpt.target_distribution = source_distribution + np.random.randint(-2,2, size=(3,2))
#%%
#%%
plt.plot(gpt.training_traj[:,0], gpt.training_traj[:,1])
for i in range(len(frame_keys)):
    plt.scatter(gpt.source_distribution[i,0], gpt.source_distribution[i,1], marker='*', s=100)
    plt.scatter(shifted_source_distribution[i,0], shifted_source_distribution[i,1], marker='x', s=100)

#%%
gpt.fit_transportation(do_rotation=False, do_scale=False)
gpt.apply_transportation()
# %%
plt.plot(gpt.training_traj[:,0], gpt.training_traj[:,1])
for i in range(len(frame_keys)):
    plt.scatter(gpt.target_distribution[i,0], gpt.target_distribution[i,1], marker='*', s=100)
    plt.scatter(target_distribution[i,0], target_distribution[i,1], marker='x', s=100)
# %%
