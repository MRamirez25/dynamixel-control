from config import Config
from robot import Robot
import argparse
from pynput.keyboard import Listener, Key

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
    
    def _on_press(self, key):
        if key == Key.left:
            self.left = True
        if key == Key.right:
            self.right = True
        if key == Key.esc:
            self.stop = True

    def _on_release(self, key):
        if key == Key.left:
            self.left = False
        if key == Key.right:
            self.right = False
    
    def control_loop(self):
        while not self.stop:
            if self.right:
                positions = {}
                for id in self.robot.ids:
                    positions[id] = int(increment)
                self.robot.move_pos_sync(positions)
            if self.left:
                positions = {}
                for id in self.robot.ids:
                    positions[id] = int(-increment)
                self.robot.move_pos_sync(positions)


if __name__ == "__main__":
    CLI=argparse.ArgumentParser()
    CLI.add_argument("--device_name", type=str, default='/dev/ttyUSB0')
    CLI.add_argument("--control_increment", type=int, default=50)
    CLI.add_argument("--ids", nargs='*', type=int, default=[1])
    args = CLI.parse_args()

    device_name = args.device_name
    increment = args.control_increment
    ids = args.ids
    controller = KeyboardPositionController(device_name=device_name, control_increment=increment, ids=ids)
    controller.robot.start(op_mode=controller.config.OPERATING_MODE_POSITION)
    controller.control_loop()
