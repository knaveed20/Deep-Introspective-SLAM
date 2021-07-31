# Deep-Introspective-SLAM

This Project is an implementation of Deep Reinforcement Learning based ORB-SLAM to predict and avoid unsafe actions leading towards SLAM failures.

# Youtube Link
Watch our DQN SLAM in action
https://www.youtube.com/channel/UCG5Bm4AGkQB7l0VPTh9krrA

# Dependencies
1. OpenCV (v2.4 or above)
2. Eigen3
3. CUDA(≥ 10.2) 
4. python
5. pygame

# Dataset Format
Currently the DQN is trained and tested on [Minos](https://github.com/minosworld/minos#readme) which is a simulator designed to support the development of multisensory models for goal-directed navigation in complex indoor environments. The [ORB-SLAM] (https://github.com/raulmur/ORB_SLAM) is integrated with Minos before training using Ros Nodes. Use Merger function to fully integrate and calliberate the camera of Orb-SLAM with Minos.

# Training and Testing

Once the complete integration is performed Training fuction can be used with following commands:

    python DQNorb.py --mode=dqn_train
    python DQNorb.py --mode=dqn_test

In order to test any pre-trained algorithm a rosbagfile of Minos is also provided which can be easily used after using camera calliberation settings of ORB-SLAM from Merger Function 

