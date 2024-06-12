from config import Config
from robot import Robot
import argparse
from pynput.keyboard import Listener, Key

class KeyboardCurrentController():
    def __init__(self, device_name, control_increment, ids) -> None:
        self.max_current = 5
        self.config = Config(device_name=device_name)
        self.robot = Robot(config=self.config, ids=ids)
        self.increment = control_increment

        self.listener = Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()
        self.left_right_counter = 0
    
    def _on_press(self, key):
        pass
        # if key == Key.left:
        #     self.left = True
        # if key == Key.right:
        #     self.right = True
        # if key == Key.esc:
        #     self.stop = True

    def _on_release(self, key):
        if key == Key.left:
            if self.left_right_counter > -self.max_current:
                self.left_right_counter -= 1
        if key == Key.right:
            if self.left_right_counter < self.max_current:
                self.left_right_counter += 1
    
    def control_loop(self):
        while not self.stop:
            if self.last_left_right_counter != self.left_right_counter:
                currents = {}
                for id in self.robot.ids:
                    currents[id] = int(self.left_right_counter)
                self.robot.move_current_sync(currents)
                self.last_left_right_counter = self.left_right_counter


if __name__ == "__main__":
    CLI=argparse.ArgumentParser()
    CLI.add_argument("--device_name", type=str, default='/dev/ttyUSB0')
    CLI.add_argument("--control_increment", type=int, default=1)
    CLI.add_argument("--ids", nargs='*', type=int, default=[1])
    args = CLI.parse_args()

    device_name = args.device_name
    increment = args.control_increment
    ids = args.ids
    controller = KeyboardCurrentController(device_name=device_name, control_increment=increment, ids=ids)
    controller.robot.start(op_mode=controller.config.OPERATING_MODE_CURRENT)
    controller.control_loop()
