#%%
from config import Config
from robot import Robot
import argparse
from pynput.keyboard import Listener, Key
import copy
import numpy as np



class SignalPositionController():
    def __init__(self, device_name, control_increment, ids) -> None:
        self.config = Config(device_name=device_name)
        self.robot = Robot(config=self.config, ids=ids)
        self.increment = control_increment

        self.listener = Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()
        self.left = False
        self.right = False
        self.stop = False
    
    def _on_press(self, key):
        if key == Key.left:
            self.left = True
        if key == Key.right:
            self.right = True
        if key == Key.esc:
            self.robot.stop()
            self.stop = True

    def _on_release(self, key):
        if key == Key.left:
            self.left = False
        if key == Key.right:
            self.right = False
    
    def control_loop(self, control_signal, iterations=1):
        previous_positions = copy.deepcopy(self.robot.initial_positions)
        previous_positions_array = np.array([previous_positions[id] for id in previous_positions])
        while not self.stop:
            present_positions = self.robot.get_positions_sync()
            if present_positions == -1:
                continue
            else:
                self.present_positions = present_positions
            present_positions_array = np.array([self.present_positions[id] for id in self.present_positions])
            # print(present_positions_array, previous_positions_array)
            # if np.linalg.norm(present_positions_array - previous_positions_array) > np.sqrt(len(self.robot.ids)) * 5:
            #     continue
            previous_positions = copy.deepcopy(self.present_positions)
            previous_positions_array = np.array([previous_positions[id] for id in previous_positions])
            # if self.right:
            for _ in range(iterations):
                for j in range(control_signal.shape[0]):
                    print(j)
                    positions = {}
                    increment = control_signal[j]
                    for id in self.robot.ids:
                        positions[id] = int(self.present_positions[id] + increment)
                    self.robot.move_pos_sync(positions, relative_to_init=False)
            if self.left:
                positions = {}
                for id in self.robot.ids:
                    positions[id] = int(self.present_positions[id] - increment)
                self.robot.move_pos_sync(positions, relative_to_init=False)

#%%
if __name__ == "__main__":
    CLI=argparse.ArgumentParser()
    CLI.add_argument("--device_name", type=str, default='/dev/my_dynamixel')
    CLI.add_argument("--control_increment", type=int, default=50)
    CLI.add_argument("--ids", nargs='*', type=int, default=[2,3])
    args = CLI.parse_args()

    device_name = args.device_name
    increment = args.control_increment
    ids = args.ids
    controller = SignalPositionController(device_name=device_name, control_increment=increment, ids=ids)
    controller.robot.start(op_mode=controller.config.OPERATING_MODE_POS_CURRENT, current_limit=5)
    fast_increment = 30 
    total_movement_servos = 30 * 6
    slow_increment = fast_increment // 2
    fast_increment_signal_array = np.repeat(fast_increment, total_movement_servos / fast_increment)
    slow_increment_signal_array = np.repeat(-slow_increment, total_movement_servos / slow_increment)
    control_signal = np.hstack((fast_increment_signal_array, slow_increment_signal_array))
    print(control_signal)
    controller.control_loop(control_signal, iterations=1)

# echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer