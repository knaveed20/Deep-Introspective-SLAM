#!/bin/bash
cat /dev/null > abc.txt
cat /dev/null > abc2.txt
rm -ff /home/YOUR PC/frames/*
rm -ff /home/YOUR PC/ORB_SLAM/bin/TrackLost.jpg 
rm -ff /home/YOUR PC/ORB_SLAM/bin/Tracking.jpg
clear
echo "Deleting Images"
rm -ff /home/YOUR PC/frames/*
echo "deleted Images"
echo "Starting ROSCORE"
roscore &
sleep 2
echo "Started ROSCORE"
cd /home/YOUR PC/minos #Following command runs MINOS
#python3 -m minos.tools.pygame_client --dataset mp3d --scene_ids 2t7WUuJeko7 --env_config pointgoal_mp3d_s --save_png --width 600 --height 400 --agent_config agent_gridworld -s map --navmap 


cd /home/ORB_SLAM #Following command runs ORB_SLAM after a 10 second delay
sleep 5
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/home/YOUR PC/
roslaunch ORB_SLAM ExampleGroovyOrNewer.launch -v &
sleep 5 #Following command runs the integration algorithm after a 10 second delay
cd /home/catkin_ws
source devel/setup.bash
rosrun merger_node merger_node

echo "$ROS_PACKAGE_PATH"
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/home/YOUR PC/ORB_SLAM
export ROS_PACKAGE_PATH=/home/YOUR PC/ORB_SLAM:${ROS_PACKAGE_PATH}
