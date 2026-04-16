## dynamixel_control package
Implements scripts to control multiple dynamixel servos simultaneously using ROS and the Python dynamixel sdk. 

Also integrates this control in a class, ```SimpleDynamixels``` that combined with the other tools and packages provided in this ROS [Hybrid Workspace](https://github.com/MRamirez25/hybrid_ws) can be used to control a hybrid soft-rigid robot. 

## Usage 

### To teach and control hybrid soft-rigid robot

After following the instructions for the [Hybrid Workspace](https://github.com/MRamirez25/hybrid_ws) to get all the required components (including this repository), you can use
```main_tags_tentacle.py``` to teach the robot skills through kinesthetic demonstrations and record source keypoints for these skills. In a new task configuration, the same script has code to then detect new target points, **re-target** the source and target keypoints as required when using the soft-rigi hybrid, and then use the transportation algorithm from [here](https://github.com/franzesegiovanni/gaussian_process_transportation.git) (which will be installed if you follow the hybrid workspace instructions) to find a new policy, which can then be executed.

### Generic dynamixel control

The code for the dynamixels can be generically used to control multiple dynamixels for any application. ```config.py``` has the settings for the dynamixels, ```robot.py``` is the lower level controller that directly writes commands to the dynamixels to control them, and ```pos_controller_node``` is the higher-level controller that listens to specific topics to receive commands to control the dynamixels.


## Data
The data from our own experiments is saved in the ```data``` (trajectories from demonstrations) and ```distributions``` (corresponding source keypoints for the skills in data).