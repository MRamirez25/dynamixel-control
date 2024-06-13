#!/usr/bin/env python3
#%%
import rospy
from std_msgs.msg import String, Bool, Float32
from .config import Config
from .robot import Robot
import threading
import numpy as np
import copy
from pynput.keyboard import Listener, Key

class PosController:
    def __init__(self) -> None:
        device_name = rospy.get_param('/dynamixels/device_name', '/dev/ttyUSB0')
        self.increment = rospy.get_param('/dynamixels/control_increment', 50)
        ids = rospy.get_param('/dynamixels/ids', [1])
        self.config = Config(device_name=device_name)
        self.robot = Robot(config=self.config, ids=ids)

        self.move_cw_sub = rospy.Subscriber('/move_dynamixels_cw', Bool, self.move_cw_cb)
        self.move_ccw_sub = rospy.Subscriber('/move_dynamixels_ccw', Bool, self.move_ccw_cb)
        # self.listener = Listener(on_press=self._on_press, on_release=self._on_release)
        # self.listener.start()
        self.left = False
        self.right = False
        self.stop = False
        self.present_positions = {id: self.robot.initial_positions[id] for id in self.robot.ids}

    def move_cw_cb(self, msg):
        # print(self.present_positions)
        self.move_cw_state  = msg.data


    def move_ccw_cb(self, msg):
        # print(self.present_positions)
        self.move_ccw_state = msg.data

    def move_cw(self):
        position_commands = {}
        for id in self.robot.ids:
            position_commands[id] = int(self.present_positions[id] + self.increment)
        self.robot.move_pos_sync(position_commands, relative_to_init=False)
        self.present_positions = self.robot.get_positions_sync()

    def move_ccw(self):
        position_commands = {}
        for id in self.robot.ids:
            position_commands[id] = int(self.present_positions[id] - self.increment)
        self.robot.move_pos_sync(position_commands, relative_to_init=False)
        self.present_positions = self.robot.get_positions_sync()

    def run(self):
        rospy.spin()
#%%
if __name__ == '__main__':
    rospy.init_node('dynamixels_pos_control_node', anonymous=True)
    pos_controller = PosController(device_name='/dev/my_dynamixel')
    pos_controller.robot.start(pos_controller.config.OPERATING_MODE_POS_CURRENT, current_limit=5)
    while not rospy.is_shutdown():
        if pos_controller.move_cw_state:
            pos_controller.move_cw()
        elif pos_controller.move_ccw_state:
            pos_controller.move_ccw()
        
# %%
