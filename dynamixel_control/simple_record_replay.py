"""
Authors: Giovanni Franzese and Ravi Prakash
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
This is the code used for the experiment of reshalving 
"""
#%%
import time
from geometry_msgs.msg import PoseStamped
import rospy
from sklearn.gaussian_process.kernels import RBF, Matern,WhiteKernel, ConstantKernel as C
import numpy as np
import pickle
import copy
from simple_dynamixels import SimpleDynamixels
#%%
if __name__ == '__main__':
    rospy.init_node("simple_dynamixels")
    simple = SimpleDynamixels()
    #%%
    simple.connect_ROS()
    time.sleep(1)
    simple.dynamixel_pos_controller.increment = 50
    simple.rec_freq = 50
    simple.control_freq = 50
    #%%
    #%% Provide the kinesthetic demonstration of the task
    time.sleep(1)
    print("Record of the cartesian trajectory")
    simple.Record_Demonstration()
    #%%  
    simple.save()
    #%% Save the teaching trajectory
    f = open("data/traj_demo.pkl","wb")
    pickle.dump(simple.training_traj,f)
    f.close()  
    f = open("data/traj_demo_ori.pkl","wb")
    pickle.dump(simple.training_ori,f)
    f.close()
    #%% Start of the experiments
    i=0
    # #%%
    # try:
    #     del gpt.training_delta
    #     del gpt.training_dK
    # except:
    #     print('already deleted')
    #%%
    #%%
    #%%
    simple.go_to_start()
    #%%
    print("Interactive Control Starting")
    simple.control()
    i=i+1    
# %%
