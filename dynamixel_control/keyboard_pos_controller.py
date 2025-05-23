#%%
from config import Config
from robot import Robot
import argparse
from pynput.keyboard import Listener, Key
import copy
import numpy as np
import time

class KeyboardPositionController():
    def __init__(self, device_name, control_increment, ids) -> None:
        self.config = Config(device_name=device_name)
        self.robot = Robot(config=self.config, ids=ids)
        self.increment = control_increment

        self.listener = Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()
        self.left = False
        self.right = False
        self.stop = False
        self.up = False
    def _on_press(self, key):
        if key == Key.left:
            self.left = True
        if key == Key.right:
            self.right = True
        if key == Key.up:
            self.up = True
        if key == Key.esc:
            self.robot.stop()
            self.stop = True

    def _on_release(self, key):
        if key == Key.left:
            self.left = False
        if key == Key.right:
            self.right = False
        if key == Key.up:
            self.up = False
    
    def control_loop(self):
        previous_positions = copy.deepcopy(self.robot.initial_positions)
        previous_positions_array = np.array([previous_positions[id] for id in previous_positions])
        increment_fast = 50
        increment_slow = 25
        diff_from_start = 0
        while not self.stop:
            present_positions = self.robot.get_positions_sync()
            if present_positions == -1:
                continue
            else:
                self.present_positions = present_positions
            # print(present_positions_array, previous_positions_array)
            # if np.linalg.norm(present_positions_array - previous_positions_array) > np.sqrt(len(self.robot.ids)) * 5:
            #     continue
            previous_positions = copy.deepcopy(self.present_positions)
            previous_positions_array = np.array([previous_positions[id] for id in previous_positions])
            if self.right:
                positions = {}
                iters = 24
                for i in range(iters):
                    diff_from_start = 50
                    print(diff_from_start)
                    for id in self.robot.ids:
                        if id == 2:
                            positions[id] = int(-diff_from_start*(iters-i))
                        if id == 3:
                            positions[id] = int(diff_from_start*(iters-i))
                        if id == 1:
                            positions[id] = int(0)

                    self.robot.move_pos_sync(positions, relative_to_init=True)
                    time.sleep(0.1)
                self.right = False
            if self.up:
                positions = {}
                diff_from_start = 0
                print(diff_from_start)
                for id in self.robot.ids:
                    if id == 2:
                        positions[id] = int(diff_from_start)
                    if id == 3:
                        positions[id] = int(-diff_from_start)
                    if id == 1:
                        positions[id] = int(0)

                self.robot.move_pos_sync(positions, relative_to_init=True)
                self.up = False
            if self.left:
                positions = {}
                diff_from_start = 1200
                for id in self.robot.ids:
                    if id == 2:
                        positions[id] = int(-diff_from_start)
                    if id == 3:
                        positions[id] = int(diff_from_start)
                    if id == 1:
                        positions[id] = int(0)
                self.robot.move_pos_sync(positions, relative_to_init=True)
                self.left = False

#%%
if __name__ == "__main__":
    CLI=argparse.ArgumentParser()
    CLI.add_argument("--device_name", type=str, default='/dev/my_dynamixel')
    CLI.add_argument("--control_increment", type=int, default=50)
    CLI.add_argument("--ids", nargs='*', type=int, default=[1,2,3])
    args = CLI.parse_args()

    device_name = args.device_name
    increment = args.control_increment
    ids = args.ids
    controller = KeyboardPositionController(device_name=device_name, control_increment=increment, ids=[1,2,3])
    controller.robot.start(op_mode=controller.config.OPERATING_MODE_POS_CURRENT, current_limit=30)
    controller.control_loop()

# echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer