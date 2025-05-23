from SIMPLe.SIMPLe import SIMPLe
import math
import numpy as np
from ILoSA.data_prep import slerp_sat
from dynamixel_control.pos_controller_node import PosController
import pathlib
import copy

class SimpleDynamixels(SIMPLe):
    def __init__(self):
        super(SimpleDynamixels, self).__init__()
        self.dynamixel_pos_controller = PosController()
        self.dynamixel_pos_controller.robot.start(self.dynamixel_pos_controller.robot.config.OPERATING_MODE_POS_CURRENT, current_limit=100)
        self.dynamixel_pos_controller.robot.present_positions = {id: self.dynamixel_pos_controller.robot.initial_positions[id] for id in self.dynamixel_pos_controller.robot.ids}
        self.dynamixel_pos_controller.present_positions = self.dynamixel_pos_controller.robot.present_positions
        self.dynamixel_pos_controller.present_currents = self.dynamixel_pos_controller.robot.present_currents

    def Record_Demonstration(self, trigger=0.005, dynamixels_relative_pos=False):
        self.Passive()
        self.end = False
        init_pos = self.cart_pos
        vel = 0
        print("Move robot to start recording.")
        while vel < trigger:
            vel = math.sqrt((self.cart_pos[0]-init_pos[0])**2 + (self.cart_pos[1]-init_pos[1])**2 + (self.cart_pos[2]-init_pos[2])**2)

        print("Recording started. Press Esc to stop.")

        self.recorded_traj = self.cart_pos.reshape(1,3)
        self.recorded_ori  = self.cart_ori.reshape(1,4)
        self.recorded_joint= self.joint_pos.reshape(1,7)
        self.recorded_dynamixels = np.empty((1, len(self.dynamixel_pos_controller.robot.initial_positions)))
        self.recorded_currents = np.empty((1, len(self.dynamixel_pos_controller.robot.initial_positions)))
        for i, id in enumerate(self.dynamixel_pos_controller.robot.ids):
            self.recorded_dynamixels[0, i] = copy.deepcopy(self.dynamixel_pos_controller.present_positions[id])
            self.recorded_currents[0,i] = copy.deepcopy(self.dynamixel_pos_controller.robot.present_currents[id])
        while not self.end:
            if self.dynamixel_pos_controller.move_all_cw_state:
                self.dynamixel_pos_controller.move_all(direction=-1)
            elif self.dynamixel_pos_controller.move_all_ccw_state:
                self.dynamixel_pos_controller.move_all(direction=+1)
            elif self.dynamixel_pos_controller.move_id1_state != 0:
                self.dynamixel_pos_controller.move_single(id=1, direction=self.dynamixel_pos_controller.move_id1_state)
            elif self.dynamixel_pos_controller.move_id2_state != 0:
                self.dynamixel_pos_controller.move_single(id=2, direction=self.dynamixel_pos_controller.move_id2_state)

            self.present_positions = np.empty((1, len(self.dynamixel_pos_controller.present_positions)))
            self.present_currents_array = np.empty((1, len(self.dynamixel_pos_controller.present_currents))) 
            for i, id in enumerate(self.dynamixel_pos_controller.robot.ids):
                self.present_positions[0,i] = copy.deepcopy(self.dynamixel_pos_controller.present_positions[id])
                self.present_currents_array[0,i] = copy.deepcopy(self.dynamixel_pos_controller.present_currents[id])
            self.recorded_traj = np.vstack([self.recorded_traj, self.cart_pos])
            self.recorded_ori  = np.vstack([self.recorded_ori,  self.cart_ori])
            self.recorded_joint = np.vstack([self.recorded_joint, self.joint_pos])
            self.recorded_dynamixels = np.vstack([self.recorded_dynamixels, self.present_positions])
            self.recorded_currents = np.vstack([self.recorded_currents, self.present_currents_array])
        
            self.r_rec.sleep()
        print('Recording ended.')
        
        if dynamixels_relative_pos:
            self.recorded_dynamixels[:,:] = self.recorded_dynamixels[:,:] - self.recorded_dynamixels[0,:]

        save_demo = input("Do you want to keep this demonstration? [y/n] \n")
        if save_demo.lower()=='y':
            self.training_traj=np.empty((0,3))
            self.training_ori=np.empty((0,4))
            self.training_dynamixels=np.empty((0,len(self.dynamixel_pos_controller.robot.present_positions)))
            self.training_currents=np.empty((0, len(self.dynamixel_pos_controller.robot.present_positions)))

            self.training_traj=np.vstack([self.training_traj,self.recorded_traj])
            self.training_ori=np.vstack([self.training_ori,self.recorded_ori])
            self.training_dynamixels=np.vstack([self.training_dynamixels,self.recorded_dynamixels])
            self.training_currents=np.vstack([self.training_currents,self.recorded_currents])

                
            print("Demo Saved")
        else:
            print("Demo Discarded")

    def step(self):
        
        i, beta = self.GGP()
        pos_goal  = self.training_traj[i,:]
        pos_goal=self.cart_pos+ np.clip([pos_goal[0]-self.cart_pos[0],pos_goal[1]-self.cart_pos[1],pos_goal[2]-self.cart_pos[2]],-1,1)
        quat_goal = self.training_ori[i,:]
        quat_goal=slerp_sat(self.cart_ori, quat_goal, 0.1)
        dynamixels_goal={id: 0 for id in self.dynamixel_pos_controller.robot.ids}
        for j, id in enumerate(self.dynamixel_pos_controller.robot.ids):
            dynamixels_goal[id] = copy.deepcopy(self.training_dynamixels[i,j])
        
        self.set_attractor(pos_goal,quat_goal)
        self.dynamixel_pos_controller.robot.move_pos_sync(dynamixels_goal, relative_to_init=True)
            
        K_lin_scaled =beta*self.K_cart
        K_ori_scaled =beta*self.K_ori
        pos_stiff = [K_lin_scaled,K_lin_scaled,K_lin_scaled]
        rot_stiff = [K_ori_scaled,K_ori_scaled,K_ori_scaled]

        self.set_stiffness(pos_stiff, rot_stiff, self.null_stiff)

    def save(self, data='last'):
        np.savez(str(pathlib.Path().resolve())+'/data/'+str(data)+'.npz', 
        # nullspace_traj=self.nullspace_traj, 
        # nullspace_joints=self.nullspace_joints, 
        training_traj=self.training_traj,
        training_ori = self.training_ori,
        training_dynamixels=self.training_dynamixels,
        training_currents=self.training_currents)
        print(np.shape(self.training_ori))
        print(np.shape(self.training_traj))  

    def load(self, file='last'):
        data =np.load(str(pathlib.Path().resolve())+'/data/'+str(file)+'.npz')

        # self.nullspace_traj=data['nullspace_traj']
        # self.nullspace_joints=data['nullspace_joints']
        self.training_traj=data['training_traj']
        self.training_ori=data['training_ori']
        self.training_dynamixels=data['training_dynamixels']
        self.training_currents=data['training_currents'] 
        # self.nullspace_traj=self.nullspace_traj
        # self.nullspace_joints=  self.nullspace_joints
        # self.training_traj=self.training_traj
        # self.training_ori=self.training_ori
        # self.training_dynamixels=