import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import zmq
import random
import time
import rospy
from std_msgs.msg import String, Float32
import sensor_msgs
import pickle
import ros_numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import math
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os

import argparse

random.seed(2020)



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
EPISODE_STEPS = 100


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)


        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class Agent():
    def __init__(self):
        self.state = None
        self.done = None
        self.reward = 0
        context = zmq.Context()
        self.action_sender = context.socket(zmq.PUSH)
        self.action_sender.bind("tcp://127.0.0.1:5557")
        self.state_reciever = context.socket(zmq.PULL)
        self.state_reciever.connect("tcp://127.0.0.1:5558")
        self.slam_interact = rospy.Publisher('/ORB_SLAM/AGENT_SLAM', String, queue_size=10)
        rospy.Subscriber("/ORB_SLAM/Episode", String, self.update_episode)
        rospy.Subscriber("/ORB_SLAM/State", sensor_msgs.msg.Image, self.update_observation)
        rospy.Subscriber("/ORB_SLAM/Frame", sensor_msgs.msg.Image, self.update_minos_img)
        rospy.Subscriber("/ORB_SLAM/Reward", Float32, self.update_reward)

        self.actions = ['forwards', 'backwards', 'turnLeft', 'turnRight', 'strafeRight', 'strafeLeft']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_actions = len(self.actions)
        self.policy_net = DQN(80, 80, self.n_actions).to(self.device)
        self.target_net = DQN(80, 80, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(5000)

        self.rcvd_state = False
        self.rcvd_reward = False
        self.rcvd_done = False
        self.rcvd_minos = False
        self.distance = None

        self.svm_model = None


        self.steps_done = 0

    def select_action(self,state, inference=False):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold or inference:
            with torch.no_grad():
                outputs = self.policy_net(state)
                print("Estimated Q values:", self.actions[0], outputs[0][0].item(), self.actions[1], outputs[0][1].item(), self.actions[2], outputs[0][2].item(), self.actions[3], outputs[0][3].item(), self.actions[4], outputs[0][4].item(), self.actions[5], outputs[0][5].item())
                return outputs.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def evaluate_actions(self,state):
        with torch.no_grad():
            outputs = self.policy_net(state)
            q_values = {}
            for i in range(len(self.actions)):
                q_values[self.actions[i]] = outputs[0][i].item()
            return q_values

    def generate_actions(self):
        distance = self.distance
        actions = []
        if(distance == None):
            actions.append("forward")
            return actions
        if(distance[0] > 0 ):
            actions.append("turnLeft")
        elif(distance[0] < 0):
            actions.append("turnRight")

        if(distance[2] > 0):
            actions.append("backwards")
        else:
            actions.append("forwards")

        return actions

    def recovery_actions(self, action):

        return "backwards"


    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)


        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))


        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def initialize(self):


        print("Initializing the Map ..")

        t = 0

        while(self.done == "initializing"):
            self.rcvd_done = False
            self.rcvd_reward = False
            self.rcvd_state = False
            action = random.choice(["backwards","strafeLeft", "strafeRight"])
            data = {"initialized" : 0, "action": action}
            self.action_sender.send_json(data)
            while(not self.rcvd_done or not self.rcvd_reward or not self.rcvd_state or not self.rcvd_minos):
                pass
            rcvd = self.state_reciever.recv_json()
            t += 1
            if(t == 100):
                data["action"] = "reset"
                self.action_sender.send_json(data)
                self.slam_interact.publish("reset")
                rcvd = self.state_reciever.recv_json()
                t = 0
                time.sleep(10)

        print("Finished Initializing the map")

    def initialize_no_action(self):

        print("Initializing the Map ..")

        while(self.done == "initializing"):
            continue

        print("Finished Initializing the map")


    def restore(self):
        print("Loading pretrained model")
        state_dict = torch.load("agent.pt")
        self.policy_net.load_state_dict(state_dict["weights"])
        self.target_net.load_state_dict(state_dict["weights"])
        self.steps_done = state_dict["steps_done"]


    def train(self, resume):
        num_episodes = 100
        resize = T.Compose([T.ToPILImage(),
                            T.Resize((80,80), interpolation=Image.CUBIC),
                            T.ToTensor()])

        if(resume):
            self.restore()
        
        for i_episode in range(num_episodes):
            # Initialize the environment and state

            self.initialize()

            print("Started Episode {}".format(i_episode))

            state = self.state
            img = state


            avg_loss = 0
            total_reward = 0

            for t in count():
                # Select and perform an action
                state = resize(state).to(self.device).unsqueeze(0)
                action_idx = self.select_action(state)
                action = self.actions[action_idx.item()]

                self.rcvd_done = False
                self.rcvd_reward = False
                self.rcvd_state = False

                data = {"initialized": 1, "action": action}

                self.action_sender.send_json(data)


                while(not self.rcvd_done or not self.rcvd_reward or not self.rcvd_state or not self.rcvd_minos):
                    pass

                total_reward += self.reward
                reward = torch.tensor([self.reward], device=self.device)

                print("Reward : {:.2f}".format(reward.item()))
                
                # Observe new state
    
                if self.done != "initializing":
                    next_state = resize(self.state).to(self.device).unsqueeze(0)
                else:
                    next_state = None

                # Store the transition in memory
                self.memory.push(state, action_idx, next_state, reward)

                # Move to the next state
                state = self.state

                # Perform one step of the optimization (on the target network)
                loss = self.optimize_model()
                if(loss):
                    avg_loss += loss
                if self.done == "initializing" or t == EPISODE_STEPS:
                    avg_loss = avg_loss / (t+1)
                    avg_reward = total_reward / (t+1)
                    self.writer = open("data.csv", "a+")
                    self.writer.write("{}\n".format(t))
                    self.writer.close()

                    print("Episode {}, Avg_loss = {:.4f}, Avg reward = {:.2f}, total reward = {}".format(i_episode +1, avg_loss, avg_reward, total_reward))
                    data = {"initialized": 1, "action": "reset"}

                    self.action_sender.send_json(data)
                    self.slam_interact.publish("reset")
                    time.sleep(5)
                    break

            torch.save({"weights": self.policy_net.state_dict(), "steps_done": self.steps_done}, "agent.pt")

            # Update the target network, copying all weights and biases in DQN
            if i_episode % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def collect_data(self, samples):

        resize = T.Compose([T.ToPILImage(),
                            T.Resize((80,80), interpolation=Image.CUBIC),
                            T.ToTensor()])

        collected = 0

        while collected < samples:
            # Initialize the environment and state

            self.initialize()

            state = self.state
            img = state



            for t in count():
                # Select and perform an action
                state = resize(state).to(self.device).unsqueeze(0)
                action = random.choice(self.actions)

                self.rcvd_done = False
                self.rcvd_reward = False
                self.rcvd_state = False
                
                data = {"initialized" : 1, "action": action}
                self.action_sender.send_json(data)


                while(not self.rcvd_done or not self.rcvd_reward or not self.rcvd_state or not self.rcvd_minos):
                    pass

                print("Collected {} samples ..".format(collected))

                img = Image.fromarray(img)
                img.save("data_svm/{}.png".format(collected))
                collected+=1

                state = self.state
                img = state

                f = open("labels.txt", "a+")

                if self.done == "initializing" or t == EPISODE_STEPS:
                    f.write("{},{}\n".format(action, 0))
                    f.close()
                    break

                else:
                    f.write("{},{}\n".format(action, 1))
                    f.close()

    def inference_dqn(self, slam_avoidance):
        steps = 0
        while True:
            resize = T.Compose([T.ToPILImage(),
                                T.Resize((80,80), interpolation=Image.CUBIC),
                                T.ToTensor()])
                # Initialize the environment and state

            self.initialize()

            data = {"initialized": 1, "action": "idle"}

            self.action_sender.send_json(data)
            rcvd = self.state_reciever.recv_json()
            self.distance = rcvd["dir"]
            curr_pos = rcvd["pos"]

            state = self.state
            img = self.minos_img

            self.policy_net.load_state_dict(torch.load("agent.pt", map_location=torch.device('cpu'))["weights"])
                
            print("Loaded the agent successfully")

            for t in count():
                steps += 1
                # Select and perform an action
                if(len(state.shape) != 4):
                    state = resize(state).to(self.device).unsqueeze(0)

                actions = self.generate_actions()

                for action in actions:

                    self.rcvd_done = False
                    self.rcvd_reward = False
                    self.rcvd_state = False
                    
                    q_values = self.evaluate_actions(state)

                    for a, q_val in q_values.items():
                        print("{} : {} , {}".format(a, q_val, "Safe" if q_val > -28 else "Non Safe"))


                    data = {"initialized" : 1, "action": action}

                    img = Image.fromarray(img)
                    img.save("imgs/{}.png".format(steps))

                    if(slam_avoidance):
                        #print("Taking %s"%action)
                        if(q_values[action] < -28):


                            img.save("imgs/fail_{}.png".format(steps))

                            f = open("critical_positions.csv", "a+")
                            f.write("{},{},{},{},{},{}\n".format(curr_pos[0], curr_pos[1], curr_pos[2], q_values[action], steps, action))
                            f.close()

                            safe_actions = []
                            
                            if(action == "turnRight"):
                                for i in range(5):
                                    safe_actions.append("strafeLeft")
                                safe_actions.append("turnRight")
                            elif(action == "turnLeft"):
                                for i in range(5):
                                    safe_actions.append("strafeRight")
                                safe_actions.append("turnLeft")

                            for a in safe_actions:
                                steps += 1
                                data["action"] = a
                                
                                self.rcvd_done = False
                                self.rcvd_reward = False
                                self.rcvd_state = False

                                self.action_sender.send_json(data)

                                while(not self.rcvd_done or not self.rcvd_reward or not self.rcvd_state or not self.rcvd_minos):
                                    pass

                                rcvd = self.state_reciever.recv_json()
                                self.distance = rcvd["dir"]
                                curr_pos = rcvd["pos"]

                                distance_to_goal = rcvd["distance"]

                                print("Distance to goal {}".format(distance_to_goal))

                        else:
                            self.rcvd_done = False
                            self.rcvd_reward = False
                            self.rcvd_state = False
                            
                            self.action_sender.send_json(data)

                            while(not self.rcvd_done or not self.rcvd_reward or not self.rcvd_state or not self.rcvd_minos):
                                pass
                            rcvd = self.state_reciever.recv_json()
                            self.distance = rcvd["dir"]

                            distance_to_goal = rcvd["distance"]

                            print("Distance to goal {}".format(distance_to_goal))
                    else:
                        self.rcvd_done = False
                        self.rcvd_reward = False
                        self.rcvd_state = False
                        
                        self.action_sender.send_json(data)

                        while(not self.rcvd_done or not self.rcvd_reward or not self.rcvd_state or not self.rcvd_minos):
                            pass 
                                               
                        rcvd = self.state_reciever.recv_json()
                        self.distance = rcvd["dir"]

                        distance_to_goal = rcvd["distance"]

                        print("Distance to goal {}".format(distance_to_goal))


                    while(not self.rcvd_done or not self.rcvd_reward or not self.rcvd_state or not self.rcvd_minos):
                        pass

                    state = self.state
                    prev_img = img
                    img = self.minos_img
                    state = resize(state).to(self.device).unsqueeze(0)
                    print("=============================")


                if self.done == "initializing":
                    print("SLAM FAILURE HAPPENED after {} steps".format(steps))
                    break


    def evaluate_dqn(self):
        output_csv = open("frames.csv", "w+")
        output_csv.write("frame_id," + ",".join(self.actions) + "\n")
        steps = 0
        resize = T.Compose([T.ToPILImage(),
                            T.Resize((80,80), interpolation=Image.CUBIC),
                            T.ToTensor()])
        # Initialize the environment and state

        self.initialize_no_action()

        state = self.state
        to_plot = self.minos_img

        img = Image.fromarray(to_plot)
        img.save("imgs/{}.png".format(0))

        self.policy_net.load_state_dict(torch.load("agent.pt", map_location=torch.device('cpu'))["weights"])
            
        print("Loaded the agent successfully")

        for t in count():
            
            steps += 1
            # Select and perform an action
            if(len(state.shape) != 4):
                state = resize(state).to(self.device).unsqueeze(0)
   
            q_values = self.evaluate_actions(state)

            print("Frame: {}".format(t))
            output_csv.write(str(t) + ",")
            for i in range(len(self.actions)):
                a = self.actions[i]
                print("{} : {} , {}".format(a, q_values[a], "Safe" if q_values[a] > -28 else "Non Safe"))
                if i == len(self.actions) - 1:
                    output_csv.write(str(q_values[a]) + "\n")
                else:
                    output_csv.write(str(q_values[a]) + ",")
                
  

            state = self.state
            to_plot = self.minos_img
            img = Image.fromarray(to_plot)
            img.save("imgs/{}.png".format(t+1))
            prev_img = img
            img = self.minos_img
            state = resize(state).to(self.device).unsqueeze(0)
            print("=============================")


            if self.done == "initializing":
                print("SLAM FAILURE HAPPENED after {} steps".format(steps))
                output_csv.close()
                break

    def svm_evaluate(self, state, actions, action_to_idx):
        outputs = {}

        for action in actions:
            action_idx = np.array(action_to_idx[action]).reshape(1,1)
            inp = np.concatenate([action_idx, state], axis=-1)
            outputs[action] = self.svm_model.predict(inp)

        return outputs

    def inference_svm(self):

        steps = 0
        while True:

            self.initialize()

            all_actions = ['turnRight', 'backwards', 'turnLeft', 'strafeLeft', 'forwards', 'strafeRight']

            action_to_idx = dict((all_actions[x], x) for x in range(len(all_actions)))

            data = {"initialized": 1, "action": "idle"}

            self.action_sender.send_json(data)
            rcvd = self.state_reciever.recv_json()
            self.distance = rcvd["dir"]
            curr_pos = rcvd["pos"]

            state = self.state
            img = state


            self.svm_model = pickle.load(open("svm.pkl", 'rb'))

                
            print("Loaded the svm model successfully")

            for t in count():
                steps += 1
                # Select and perform an action
                if(len(state.shape) != 4):

                    state = np.array(Image.fromarray(state).convert("RGB").resize((80,80))).reshape(1,-1)

                actions = self.generate_actions()

                for action in actions:

                    self.rcvd_done = False
                    self.rcvd_reward = False
                    self.rcvd_state = False
                    
                    evaluated_actions = self.svm_evaluate(state, all_actions, action_to_idx)

                    data = {"initialized" : 1, "action": action}

                    img = Image.fromarray(self.minos_img)
                    img.save("imgs/{}.png".format(steps))
        
                    #print("Taking %s"%action)
                    if(evaluated_actions[action] == 0):

                        img.save("imgs/fail_{}.png".format(steps))

                        f = open("critical_positions.csv", "a+")
                        f.write("{},{},{}\n".format(curr_pos[0], curr_pos[1], curr_pos[2]))
                        f.close()

                        safe_actions = []
                        
                        if(action == "turnRight"):
                            for i in range(5):
                                safe_actions.append("strafeLeft")
                            safe_actions.append("turnRight")
                        elif(action == "turnLeft"):
                            for i in range(5):
                                safe_actions.append("strafeRight")
                            safe_actions.append("turnLeft")

                        else:
                            safe_actions.append(random.choice(actions))

                        for a in safe_actions:
                            steps += 1
                            data["action"] = a
                            
                            self.rcvd_done = False
                            self.rcvd_reward = False
                            self.rcvd_state = False

                            self.action_sender.send_json(data)

                            while(not self.rcvd_done or not self.rcvd_reward or not self.rcvd_state or not self.rcvd_minos):
                                pass

                            rcvd = self.state_reciever.recv_json()
                            self.distance = rcvd["dir"]
                            curr_pos = rcvd["pos"]

                            distance_to_goal = rcvd["distance"]

                            print("Distance to goal {}".format(distance_to_goal))

                    else:
                        self.rcvd_done = False
                        self.rcvd_reward = False
                        self.rcvd_state = False
                        
                        self.action_sender.send_json(data)

                        while(not self.rcvd_done or not self.rcvd_reward or not self.rcvd_state or not self.rcvd_minos):
                            pass 
                                            
                        rcvd = self.state_reciever.recv_json()
                        self.distance = rcvd["dir"]

                        distance_to_goal = rcvd["distance"]

                        print("Distance to goal {}".format(distance_to_goal))
                        

                    self.rcvd_done = False
                    self.rcvd_reward = False
                    self.rcvd_state = False

                    while(not self.rcvd_done or not self.rcvd_reward or not self.rcvd_state or not self.rcvd_minos):
                        pass

                    state = self.state
                    prev_img = img
                    img = self.minos_img
                    state = np.array(Image.fromarray(state).convert("RGB").resize((80,80))).reshape(1,-1)


                if self.done == "initializing":
                    try:
                        prev_img = Image.fromarray(prev_img)
                    except:
                        pass
                    prev_img.save("imgs/failure.png")
                    print("SLAM FAILURE HAPPENED after {} steps".format(steps))
                    break




    def update_episode(self,data):
        self.done = data.data
        self.rcvd_done = True

    def update_reward(self, data):
        self.reward = data.data
        self.rcvd_reward = True

    def update_observation(self, data):
        self.state = ros_numpy.numpify(data)
        self.rcvd_state = True

    def update_minos_img(self, data):
        self.minos_img = ros_numpy.numpify(data)
        self.rcvd_minos = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="running mode, train, test, collect_data", default="test")
    parser.add_argument("--samples", type= int, help="number of samples to collect", required=False)
    args = parser.parse_args()
    rospy.init_node('agent')
    agent = Agent()


    if(args.mode == "collect_data"):
        agent.collect_data(args.samples)

    elif args.mode == "svm_test":
        agent.inference_svm()

    elif args.mode == "dqn_test":
        agent.inference_dqn(slam_avoidance=True)

    elif args.mode == "dqn_evaluate":
        agent.evaluate_dqn()

    elif args.mode == "naive_slam":
        agent.inference_dqn(slam_avoidance=False)
    
    elif args.mode == "dqn_train":
        agent.train(False)

        

    # if args.test:
    #     print("Interacting with the trained agent ...")
    #     agent.inference(args.slam_avoidance)
    # else:
    #     print("Training the agent from scratch ..")
    #     agent.train(args.resume)

if __name__ == '__main__':
    main()
