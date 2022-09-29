import sys
sys.path.append(sys.path[0][:-17])
from flow_solver import FlowSolver
import numpy as np
import yaml
from dolfin import *
set_log_active(False)
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay
from tqdm import tqdm
import time
from torch import nn
from torch import optim
from airfoilgcnn import AirfoilGCNN, NodeRemovalNet
from ParallelMultiSnapshotEnv2DAirfoil import ParallelMultiSnapshotEnv2DAirfoil as Env2DAirfoil
from itertools import count
import random
import torch
from torch.nn import functional as F
from collections import namedtuple
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import joblib
import ray

@ray.remote
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


steps_done = 0
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-steps_done/EPS_DECAY)
    eps_threshs.append(eps_threshold)
    steps_done += 1
    if(steps_done%100 == 0):
        np.save("./{}/{}eps.npy".format(save_dir, PREFIX), eps_threshs)
    if(sample > eps_threshold): # Exploit
        with torch.no_grad():
            return torch.tensor([[policy_net_1(state).argmax()]]).to(device)
    else: # Explore
        if(flow_config['agent_params']['do_nothing']):
            return torch.tensor([random.sample(range(n_actions+1), 1)], dtype=torch.long).to(device)
        else:
            return torch.tensor([random.sample(range(n_actions), 1)], dtype=torch.long).to(device)

def optimize_model(optimizer):
    if len(memory) < BATCH_SIZE:
        return
    #print("OPTIMIZING MODEL...")
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)
    non_final_next_states = [s for s in batch.next_state if s is not None]

    # Get batch
    state_batch = batch.state
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net

    # Easiest way to batch this
    loader = DataLoader(state_batch, batch_size=BATCH_SIZE)
    for data in loader:
        try:
            output = policy_net_1(data)
        except RuntimeError:
            print("\n\n")
            #print(data)
            #print(data.x)
            #print(data.edge_index)
            #print(data.edge_attr)
            print("\n\n")
            raise
    state_action_values = output[:,action_batch[:,0]].diag()

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE).to(device).float()
    loader = DataLoader(non_final_next_states, batch_size=BATCH_SIZE)

    # get batched output
    for data in loader:
        try:
            output = policy_net_2(data).max(1)[0]
        except RuntimeError:
            print("\n\n")
            #print(data)
            #print(data.x)
            #print(data.edge_index)
            #print(data.edge_attr)
            print("\n\n")
            raise
    try:
        next_state_values[non_final_mask] = output
    except RuntimeError:
        return

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = criterion(state_action_values.float(), expected_state_action_values.float()).float()
    losses.append(loss.item())
    if((len(memory)%25) == 0):
        np.save("./{}/{}losses.npy".format(save_dir, PREFIX), losses)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def _movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma


if __name__ == '__main__':
    start_ep = len(ep_reward) if(RESTART) else 0
    for episode in range(start_ep, num_episodes):
        # Analysis
        episode_actions = []
        episode_rewards = []
    
        print("\nEPISODE: {}\n".format(episode))
        acc_rew = 0.0
        acc_rews = []
        if(episode != 0):
            env = Env2DAirfoil(flow_config)
    
        state = env.get_state()
        for t in tqdm(count()):
    
            action = select_action(state)
            next_state, reward, done, _ = env.step(action.item())
    
            # Analysis
            episode_actions.append(action.item())
            episode_rewards.append(reward)
    
            acc_rew += reward
            reward = torch.tensor([reward])
    
            # Observe new state
            if(done):
                next_state = None
    
            if(next_state is not None):
                memory.push(state.to(device), action.to(device), next_state.to(device), reward.to(device))
            else:
                memory.push(state.to(device), action.to(device), next_state, reward.to(device))
    
            state = next_state
    
            optimize_model(optimizer)
            if(done):
                ep_reward.append(acc_rew)
                break
    
        # Analysis
        all_actions.append(np.array(episode_actions))
        all_rewards.append(np.array(episode_rewards))
    
        if((episode % TARGET_UPDATE) == 0):
            #target_net.load_state_dict(policy_net.state_dict())
            if(first):
                optimizer = optimizer_fn(policy_net_1.parameters())
                first = False
            else:
                optimizer = optimizer_fn(policy_net_2.parameters())
                first = True
        
            fig, ax = plt.subplots()
            ax.plot(ep_reward)
            if(len(ep_reward) >= 25):
                ax.plot(list(range(len(ep_reward)))[24:], _movingaverage(ep_reward, 25))
    
            if(len(ep_reward) >= 200):
                ax.plot(list(range(len(ep_reward)))[199:], _movingaverage(ep_reward, 200))
    
            ax.set(xlabel="Episode", ylabel="Reward")
            ax.set_title("DQN Training Reward")
            plt.savefig("./{}/{}reward.png".format(save_dir, PREFIX))
            plt.close()
    
        np.save("./{}/{}reward.npy".format(save_dir, PREFIX), ep_reward)
    
        if(len(ep_reward)%1 == 0):
            np.save("./{}/{}actions.npy".format(save_dir, PREFIX), np.array(all_actions, dtype=object))
            np.save("./{}/{}rewards.npy".format(save_dir, PREFIX), np.array(all_rewards, dtype=object))
            torch.save(policy_net_1.state_dict(), "./{}/{}policy_net_1.pt".format(save_dir, PREFIX))
            torch.save(policy_net_2.state_dict(), "./{}/{}policy_net_2.pt".format(save_dir, PREFIX))
            #joblib.dump(env.model, "./{}/{}surrogate_model.joblib".format(save_dir, PREFIX))
    
    print("ACTION: {}".format(np.mean(action_times)))
    print("STEP: {}".format(np.mean(step_times)))
    print("PUSH: {}".format(np.mean(push_times)))
    print("OPTIMIZE: {}".format(np.mean(optimize_times)))
    
