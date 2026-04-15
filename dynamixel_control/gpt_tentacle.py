"""
Code modified from https://github.com/franzesegiovanni/policy_transportation/blob/devel/ros_ws/script/modules.py,
by Giovanni Franzese, adding the GPT_tag_tentacle class, which is a combination of the GaussianProcessTransportation 
and Tag_Detector modules, but using the SimpleDynamixels class.
"""
from policy_transportation import GaussianProcessTransportation as Transport
from tag_detector import Tag_Detector

from simple_dynamixels import SimpleDynamixels
import rospy
from SIMPLe.SIMPLe import SIMPLe

class GPT_tag_tentacle(SimpleDynamixels, Transport, Tag_Detector):
    def __init__(self):
        rospy.init_node('GPT', anonymous=True)
        rospy.sleep(2)
        super(GPT_tag_tentacle,self).__init__()

# class GPT_2d(Transport, Tag_Detector):
#     def __init__(self):
#         super(GPT_2d, self).__init__()
        
class GPT_tag(Transport, Tag_Detector, SIMPLe):
    def __init__(self):
        rospy.init_node('GPT', anonymous=True)
        rospy.sleep(2)
        super(GPT_tag,self).__init__()