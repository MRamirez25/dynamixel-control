"""
Authors: Ravi Prakash & Giovanni Franzese, March 2023
Email: g.franzese@tudelft.nl, r.prakash@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
from policy_transportation import GaussianProcessTransportation as Transport
from tag_detector import Tag_Detector

from simple_dynamixels import SimpleDynamixels
import rospy

class GPT_tag_tentacle(SimpleDynamixels, Transport, Tag_Detector):
    def __init__(self):
        rospy.init_node('GPT', anonymous=True)
        rospy.sleep(2)
        super(GPT_tag_tentacle,self).__init__()

# class GPT_2d(Transport, Tag_Detector):
#     def __init__(self):
#         super(GPT_2d, self).__init__()
        
