"""
Authors: Giovanni Franzese and Ravi Prakash
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
This is the code used for the experiment of reshalving 
"""
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
#%%
if __name__ == '__main__':
    rospy.set_param('/dynamixels/ids', [1,2])
    gpt=GPT_tag_tentacle()
    gpt.connect_ROS()
    time.sleep(1)
    gpt.training_delta = None
    gpt.training_dK = None
    gpt.dynamixel_pos_controller.increment = 100
    gpt.rec_freq = 50
    gpt.control_freq = 30
    #%%
    exposure_client = dynamic_reconfigure.client.Client("/camera/stereo_module", timeout=10)
    detection_params = {'exposure': 7000, 'enable_auto_exposure': False}
    exposure_client.update_configuration(detection_params)
    #%%
    # present_offsets = gpt.dynamixel_pos_controller.robot.get_homing_offsets()
    # print(present_offsets)
    # #%%
    # new_offsets = copy.deepcopy(present_offsets)
    # new_offsets[1] = new_offsets[1] -200
    # new_offsets[2] = new_offsets[2] + 3600
    # #%%
    # gpt.dynamixel_pos_controller.robot.set_homing_offsets(new_offsets)
    #%%
    print(gpt.dynamixel_pos_controller.robot.get_positions_sync())
    gpt.speed=1.5
    #%%
    positions = {1: 10400, 2: 14300}
    #%%
    gpt.dynamixel_pos_controller.robot.move_pos_sync(positions, relative_to_init=False)
    # Record the trajectory to scan the environment
    #%%
    gpt.record_traj_tags()
    #%%
    gpt.save_traj_tags()
    #%%
    gpt.load_traj_tags()
    #%% Learn the current tag distribution by moving around the environment
    gpt.source_distribution=[]
    gpt.target_distribution=[]
    gpt.source_distribution=gpt.record_tags(gpt.source_distribution)
    #%%
    print("Source len",len(gpt.source_distribution))
    print("Save the  source distribution data") 
    gpt.save_distributions()  # we are saving both the distributions but only the source is not empty
    # Save source configuration for data analysis later on
    f = open("distributions/draw.pkl","wb")  
    pickle.dump(gpt.source_distribution,f)  
    f.close()   
    #%% Provide the kinesthetic demonstration of the task
    time.sleep(1)

    print("Record of the cartesian trajectory")
    gpt.Record_Demonstration(dynamixels_relative_pos=True)
    #%%  
    gpt.save("draw_phi2")
    #%% Save the teaching trajectory
    f = open("data/traj_demo.pkl","wb")
    pickle.dump(gpt.training_traj,f)
    f.close()  
    f = open("data/traj_demo_ori.pkl","wb")
    pickle.dump(gpt.training_ori,f)
    f.close()
    # #%%
    # f = open("data/currents_red_small.pkl","wb")
    # pickle.dump(gpt.training_currents,f)
    # f.close()
    # f = open("data/dynamixel_positions_red_small.pkl","wb")
    # pickle.dump(gpt.training_dynamixels,f)
    # f.close()
    ###################################################
    #%%

    ###################################################
    #%% Start of the experiments
    i=2
    #%%
    size = "size_1"
    noise_magnitude = 0.15
    mass = f"noise_{noise_magnitude*100}"
    #%%
    gpt.load_distributions(filename='dynamic_grasp') #you need to re-load the distribution because that is in a particular format and then it is coverget after and overwritten inside the class
    gpt.load(file='dynamic_grasp')
    # if gpt.training_tentacle[0] != 0:
    #     gpt.training_tentacle = gpt.training_tentacle - gpt.training_tentacle[0]
    gpt.load_traj_tags()
    gpt.home_gripper()
    #%%
    gpt.target_distribution=[]
    gpt.target_distribution=gpt.record_tags(gpt.target_distribution)
    print("Target len", len(gpt.target_distribution) )
    #%%
    # Save target distribution for data analysis later on
    # Function to prompt user for confirmation
    # def ask_confirmation(filename):
    #     user_input = input(f"File {filename} already exists. Do you want to overwrite it? (y/n): ")
    #     return user_input.lower() == 'y'

    # # Define the file path
    # file_path = f"data/walls_red/target_{i}.pkl"

    # # Ensure the directory exists
    # os.makedirs(f"data/walls_red", exist_ok=True)

    # # Check if the file already exists
    # if os.path.exists(file_path):
    #     if ask_confirmation(file_path):
    #         # If user confirms, overwrite the file
    #         with open(file_path, "wb") as f:
    #             pickle.dump(gpt.target_distribution, f)
    #         print(f"File {file_path} has been overwritten.")
    #     else:
    #         print(f"File {file_path} was not overwritten.")
    # else:
    #     # If the file doesn't exist, create and write to it
    #     with open(file_path, "xb") as f:
    #         pickle.dump(gpt.target_distribution, f)
    #     print(f"File {file_path} has been created.")

    #%%
    # random_direction = random.uniform(0, 2*np.pi)
    # #%%
    # x_direction_noise = noise_magnitude*np.cos(random_direction)
    # y_direction_noise = noise_magnitude*np.sin(random_direction)
    # for detection in gpt.target_distribution:
    #     if detection.id[0] == 14:
    #         detection.pose.pose.pose.pose.position.x = detection.pose.pose.pose.pose.position.x + x_direction_noise
    #         detection.pose.pose.pose.pose.position.y = detection.pose.pose.pose.pose.position.y + y_direction_noise

    #%%
    if type(gpt.target_distribution) != type(gpt.source_distribution):
        raise TypeError("Both the distribution must be a numpy array.")
    elif not(isinstance(gpt.target_distribution, np.ndarray)) and not(isinstance(gpt.source_distribution, np.ndarray)):
        gpt.convert_distribution_to_array(use_orientation=False)

    #%%
    try:
        del gpt.training_delta
        del gpt.training_dK
    except:
        print('already deleted')
    #%%
    original_source_distribution = copy.deepcopy(gpt.source_distribution)
    distances = np.empty((len(gpt.training_traj), 0))
    for j in range(gpt.source_distribution.shape[0]):
        distance_frame_i = np.linalg.norm(gpt.training_traj - gpt.source_distribution[j], keepdims=True, axis=-1)
        distances = np.hstack((distances, distance_frame_i))
    #%%
    closest_points_idxs = np.argmin(distances, axis=0)
    projected_source_distribution = gpt.training_traj[closest_points_idxs]
    gpt.source_distribution = projected_source_distribution
    gpt.source_distribution[:,-1] = original_source_distribution[:,-1]
    #%%
    #%%
    source_target_diff = gpt.target_distribution - original_source_distribution
    projected_target_distribution = projected_source_distribution + source_target_diff
    gpt.target_distribution = projected_target_distribution
    gpt.target_distribution[:,-1] = gpt.source_distribution[:,-1]
    #%%
    old_traj = copy.deepcopy(gpt.training_traj)
    #%%
    time.sleep(1)
    print("Find the transported policy")
    gpt.kernel_transport=C(0.1) * RBF(length_scale=[0.3,0.3, 0.3 ],  length_scale_bounds=[0.1,0.5]) + WhiteKernel(0.0000001, [0.0000001,0.0000001])
    gpt.fit_transportation(do_rotation=False, do_scale=False)
    gpt.apply_transportation()
    #%%
    # # Save transported trajectory and orientation for data analysis
    # f = open("data/traj_"+str(i)+".pkl","wb") 
    # pickle.dump(gpt.training_traj,f)
    # f.close()
    # f = open("data/traj_ori_"+str(i)+".pkl","wb")
    # pickle.dump(gpt.training_ori,f) 
    # f.close()
    #%%
    # # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the trajectories
    ax.plot(old_traj[:,0], old_traj[:,1], old_traj[:,2], label='Trajectory 1', color='blue')
    ax.plot(gpt.training_traj[:,0], gpt.training_traj[:,1],gpt.training_traj[:,2],label='Trajectory 2', color='red')
    ax.scatter3D(gpt.source_distribution[0,0], gpt.source_distribution[0,1],gpt.source_distribution[0,2],marker="$1$", s=100)
    ax.scatter3D(gpt.source_distribution[1,0], gpt.source_distribution[1,1],gpt.source_distribution[1,2],marker="$2$", s=100)
    # ax.scatter3D(gpt.source_distribution[2,0], gpt.source_distribution[2,1],gpt.source_distribution[2,2],marker="$3$", s=100)

    ax.scatter3D(gpt.target_distribution[0,0], gpt.target_distribution[0,1],gpt.target_distribution[0,2],marker="$1$", s=100)
    ax.scatter3D(gpt.target_distribution[1,0], gpt.target_distribution[1,1],gpt.target_distribution[1,2],marker="$2$", s=100)
    # ax.scatter3D(gpt.target_distribution[2,0], gpt.target_distribution[2,1],gpt.target_distribution[2,2],marker="$3$", s=100)

    # Customize the plot
    ax.set_title('3D Trajectories')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()

    # Show the plot
    plt.show()
    #%%
    gpt.go_to_start()
    # if (gpt.dynamixel_pos_controller.cumul_displacements != gpt.training_tentacle[0,:]).any():
        # print("Initial tentacle position might not match initial learned position")
    #%%
    print("Interactive Control Starting")
    gpt.control()
    #%%
    print(i)
    i=i+1
    print(i)
    # print("noise vector angle in rad\n", random_direction)    
    #%% Make the robot passive and reset the environment 
    gpt.Passive()
# %%
f = open("data/currents_smallest.pkl","rb")   
current_smallest = pickle.load(f)
f.close()
f = open("data/dynamixel_positions_smallest.pkl","rb")   
positions_smallest = pickle.load(f)
f.close()
# %%
f = open("data/currents_green.pkl","rb")   
current_largest= pickle.load(f)
f.close()
f = open("data/dynamixel_positions_green.pkl","rb")   
positions_largest = pickle.load(f)
f.close()
# %%
import matplotlib.pyplot as plt
plt.plot(current_smallest[(current_smallest>100)[:,0]])
plt.plot(current_largest[(current_largest>100)[:,0]])
plt.ylim(63450, 65600)
# %%
plt.plot(positions_smallest)
plt.plot(positions_largest)
#%%
gpt.training_dynamixels
# %%
gpt.training_dynamixels[:,:] = gpt.training_dynamixels[:,:] - gpt.training_dynamixels[0,:]
# %%
# %%
f = open("data/heatmap/size_2/225/target_5.pkl","rb")   
target = pickle.load(f)
f.close()
# %%
target

