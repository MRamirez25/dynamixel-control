#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt
#%%
positions = []
#%%
for k in range(9):
    with open(f'data/push_red/target_{k}.pkl', 'rb') as f:
        marker_data = pickle.load(f)
        for marker in marker_data:
            id = marker.id[0]
            if id == 14:
                xyz_position = marker.pose.pose.pose.pose.position
                position_trial = np.array([xyz_position.x, xyz_position.y, xyz_position.z])
                positions.append(position_trial)
# %%
for position in positions:
    plt.scatter(position[0], position[1])
# %%
