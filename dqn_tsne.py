from gym import Env, spaces
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

# TODO: These are misnomers
from airfoilgcnn import AirfoilGCNN, NodeRemovalNet
#from airfoilgcnn_training import PredictionNet
from Env2DAirfoil import Env2DAirfoil
from ClosestEnv2DAirfoil import ClosestEnv2DAirfoil
from ClosestSmoothingEnv2DAirfoil import ClosestSmoothingEnv2DAirfoil

from itertools import count

import random

import torch
from torch.nn import functional as F

from collections import namedtuple

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from sklearn.manifold import TSNE

SEED = 137
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if(torch.cuda.is_available()):
    print("USING GPU")
    device = torch.device('cuda:0')
else:
    print("USING CPU")
    device = torch.device('cpu')
#device = torch.device('cpu')

#PREFIX = 'ag11_18_'
#PREFIX = 'restart_ag11_18_'
PREFIX = 'ys930_13_'
#PREFIX = 'ys930_16_'
#PREFIX = 'ys930_16_graphsage_'


# Set up environment
#with open("./configs/ag11.yaml", 'r') as stream:
with open("./configs/ys930.yaml", 'r') as stream:
    flow_config = yaml.safe_load(stream)
env = ClosestSmoothingEnv2DAirfoil(flow_config)

# Hold on to ground truth values
flow_config['agent_params']['gt_drag'] = env.gt_drag
flow_config['agent_params']['gt_time'] = env.gt_time
flow_config['agent_params']['u'] = env.u.copy(deepcopy=True)
flow_config['agent_params']['p'] = env.p.copy(deepcopy=True)

n_actions = env.action_space.n
print("N CLOSEST: {}".format(n_actions))

# Set up for DQN
policy_net_1 = NodeRemovalNet(n_actions+1, conv_width=256, topk=0.1).to(device).float()
policy_net_2 = NodeRemovalNet(n_actions+1, conv_width=256, topk=0.1).to(device).float()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = 0.5
    if(sample > eps_threshold): # Exploit
        with torch.no_grad():
            return torch.tensor([[policy_net_1(state).argmax()]]).to(device)
    else: # Explore
        if(flow_config['agent_params']['do_nothing']):
            return torch.tensor([random.sample(range(n_actions+1), 1)], dtype=torch.long).to(device)
        else:
            return torch.tensor([random.sample(range(n_actions), 1)], dtype=torch.long).to(device)


# Prime policy nets
policy_net_1.set_num_nodes(flow_config['agent_params']['N_closest'])
policy_net_2.set_num_nodes(flow_config['agent_params']['N_closest'])

# Load policy net parameters
policy_net_1.load_state_dict(torch.load("./trained_dqns/{}policy_net_1.pt".format(PREFIX)))
policy_net_2.load_state_dict(torch.load("./trained_dqns/{}policy_net_2.pt".format(PREFIX)))
print("SUCCESSFULLY LOADED POLICY NETS")


embeddings, drags = [], []
for episode in range(10000):
    # Analysis
    episode_actions = []
    episode_rewards = []

    print("\nEPISODE: {}\n".format(episode))
    acc_rew = 0.0
    acc_rews = []
    if(episode != 0):
        env = ClosestSmoothingEnv2DAirfoil(flow_config)

    env.set_plot_dir("small_learning_mesh_plots/episode_{}".format(episode))
    state = env.get_state()
    for t in tqdm(count()):

        if((episode == 0) and (t==0)):
            _ = env.calculate_reward()
            drags.append(env.new_drag)
            embeddings.append(policy_net_1(state, embedding=True).cpu().detach().numpy()[0])

        if(len(embeddings) == 100000): 
            break

        action = select_action(state)
        next_state, reward, done, _ = env.step(action.item())

        try:
            drags.append(env.new_drag)
            embeddings.append(policy_net_1(state, embedding=True).cpu().detach().numpy()[0])
        except RuntimeError:
            print("CALCULATION BROKE")
            break

        # Analysis
        episode_actions.append(action.item())
        episode_rewards.append(reward)

        acc_rew += reward
        reward = torch.tensor([reward])
        # Observe new state
        if(done):
            next_state = None
            break

        state = next_state

        if(len(embeddings)%100 == 0):
            np.save("./tsne/{}original_embeddings.npy".format(PREFIX), embeddings)
            np.save("./tsne/{}drag.npy".format(PREFIX), drags)

    if(len(embeddings) == 100000): 
        break

embeddings = np.array(embeddings)
print(embeddings.shape)
np.save("./tsne/{}original_embeddings.npy".format(PREFIX), embeddings)
np.save("./tsne/{}drag.npy".format(PREFIX), drags)

tsne_model = TSNE(n_components=2, random_state=137)
t_embeddings = tsne_model.fit_transform(embeddings)
print(t_embeddings.shape)
np.save("./tsne/{}embeddings.npy".format(PREFIX), t_embeddings)
