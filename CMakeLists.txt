cmake_minimum_required(VERSION 3.0.2)
project(dynamixel_control)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
)

catkin_package()
catkin_python_setup()