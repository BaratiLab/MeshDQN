import torch
import os
import sys
import yaml
from ray_dqn import DQNConfig
from ray import tune, air, train
from ray.air import session, Checkpoint
import ray
from ray.air.config import ScalingConfig
from ray.train.torch import TorchTrainer
from ray.tune.registry import register_env
from ParallelMultiSnapshotEnv2DAirfoil import ParallelMultiSnapshotEnv2DAirfoil as Env2DAirfoil
from parallel_airfoilgcnn import NodeRemovalNet
from tqdm import tqdm
import time
#from dolfin import *
from itertools import count
import random
import numpy as np
from torch import optim
from matplotlib import pyplot as plt
from collections import namedtuple
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import math


#ray.init()
ray.init(log_to_driver=False)
SEED = 1370
#SEED = 137*137
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
device = torch.device('cpu')

Transition = namedtuple('Transition',
                       ('state', 'action', 'next_state', 'reward'))
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

    #def __len__(self):
    def size(self):
        return len(self.memory)


def _movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma


@ray.remote
class DataHandler(object):
    def __init__(self, save_dir):
        from parallel_airfoilgcnn import NodeRemovalNet
        self.save_dir = save_dir
        self.rewards = []
        self.ep_rewards = []
        self.losses = []
        self.actions = []
        self.epss = []

        if(RESTART):
            for i in range(RESTART_NUM-1):
                self.save_dir += "RESTART_"
            try:
                self.rewards = list(np.load(self.save_dir + "reward.npy", allow_pickle=True))
            except OSError:
                self.rewards = []
            try:
                self.ep_rewards = list(np.load(self.save_dir + "rewards.npy", allow_pickle=True))
            except OSError:
                self.ep_rewards = []
            try:
                self.losses = list(np.load(self.save_dir + "losses.npy", allow_pickle=True))
            except OSError:
                self.losses = []
            try:
                self.actions = list(np.load(self.save_dir + "actions.npy", allow_pickle=True))
            except OSError:
                self.actions = []
            try:
                self.epss = list(np.load(self.save_dir + "eps.npy", allow_pickle=True))
            except OSError:
                self.epss = []
            self.save_dir += "RESTART_"
            print("\n\nWRITING\n\n")
            self.write()

    def add_eps(self, eps):
        self.epss.append(eps)

    def num_eps(self):
        return len(self.epss)

    def add_loss(self, loss):
        self.losses.append(loss)

    def add_episode(self, ep_rew, ep_action):
        self.rewards.append(sum(ep_rew))
        self.ep_rewards.append(ep_rew)
        self.actions.append(ep_action)

    def write(self):
        np.save(self.save_dir + "reward.npy", self.rewards)
        np.save(self.save_dir + "rewards.npy", self.ep_rewards)
        np.save(self.save_dir + "losses.npy", self.losses)
        np.save(self.save_dir + "actions.npy", self.actions)
        np.save(self.save_dir + "eps.npy", self.epss)

    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.rewards)
        if(len(self.rewards) >= 25):
            ax.plot(list(range(len(self.rewards)))[24:], _movingaverage(self.rewards, 25))

        if(len(self.rewards) >= 200):
            ax.plot(list(range(len(self.rewards)))[199:], _movingaverage(self.rewards, 200))

        ax.set(xlabel="Episode", ylabel="Reward")
        ax.set_title("DQN Training Reward")
        plt.savefig(self.save_dir + "reward.png".format(save_dir, PREFIX))
        plt.close()


grad_steps = 0
@ray.remote
class ParameterServer(object):
    def __init__(self, save_dir, PREFIX):
        self.save_dir = save_dir
        self.PREFIX = PREFIX
        self.policy_net_1 = NodeRemovalNet(n_actions+1, conv_width=128, topk=0.1).float()
        self.policy_net_2 = NodeRemovalNet(n_actions+1, conv_width=128, topk=0.1).float()
        self.policy_net_1.set_num_nodes(NUM_INPUTS)
        self.policy_net_2.set_num_nodes(NUM_INPUTS)

        if(RESTART):
            for i in range(RESTART_NUM-1):
                self.PREFIX = "restart_" + self.PREFIX
            self.policy_net_1.load_state_dict(torch.load(
                                    "./{}/{}policy_net_1.pt".format(save_dir, PREFIX)))
            self.policy_net_2.load_state_dict(torch.load(
                                    "./{}/{}policy_net_2.pt".format(save_dir, PREFIX)))
            self.PREFIX = "restart_" + self.PREFIX

        self.optimizer_fn = lambda parameters: optim.Adam(parameters, lr=LEARNING_RATE,
                                                          weight_decay=WEIGHT_DECAY)
        self.optimizer = self.optimizer_fn(self.policy_net_1.parameters())
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                       milestones=[500000, 1000000, 1500000], gamma=0.1)
        self.num_grads = 0
        self.select = True

    def apply_gradients(self, *gradients):
        if((self.num_grads % TARGET_UPDATE) == 0):
            self.select = not(self.select)

        #self.optimizer.zero_grad()
        self.optimizer.step()
        self.scheduler.step()
        self.num_grads += 1

        if(self.select):
            self.policy_net_1.set_gradients(gradients[0])
            self.optimizer = self.optimizer_fn(self.policy_net_1.parameters())
            return self.policy_net_1.get_weights()
        else:
            self.policy_net_2.set_gradients(gradients[0])
            self.optimizer = self.optimizer_fn(self.policy_net_2.parameters())
            return self.policy_net_2.get_weights()

    def get_weights(self):
        if(self.select):
            return self.policy_net_1.get_weights()
        else:
            return self.policy_net_2.get_weights()

    def select_action(self, state):
        return torch.tensor([[self.policy_net_1(state).argmax()]]).to(device)

    def select(self):
        return self.select

    def write(self):
        torch.save(self.policy_net_1.state_dict(),
                    "/home/fenics/drl_projects/MeshDQN/parallel_training/{}/{}policy_net_1.pt".format(self.save_dir, self.PREFIX))
        torch.save(self.policy_net_2.state_dict(),
                    "/home/fenics/drl_projects/MeshDQN/parallel_training/{}/{}policy_net_2.pt".format(self.save_dir, self.PREFIX))


@ray.remote
class DataWorker(object):
    def __init__(self):
        self.policy_net_1 = NodeRemovalNet(n_actions+1, conv_width=128, topk=0.1).float()
        self.policy_net_2 = NodeRemovalNet(n_actions+1, conv_width=128, topk=0.1).float()
        self.policy_net_1.set_num_nodes(NUM_INPUTS)
        self.policy_net_2.set_num_nodes(NUM_INPUTS)
        if(RESTART):
            self.policy_net_1.load_state_dict(torch.load(
                                    "./{}/{}policy_net_1.pt".format(save_dir, PREFIX)))
            self.policy_net_2.load_state_dict(torch.load(
                                    "./{}/{}policy_net_2.pt".format(save_dir, PREFIX)))
        self.num_grads = 0
        self.select = True


    #def _get_data(self, replay):
    def _get_data(self):
        #transitions = ray.get(replay.sample.remote(BATCH_SIZE))
        transitions = ray.get(memory.sample.remote(BATCH_SIZE))
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), dtype=torch.bool)
        non_final_next_states = [s for s in batch.next_state if s is not None]

        # Get batch
        state_batch = batch.state
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)

        # Easiest way to batch this
        loader = DataLoader(state_batch, batch_size=BATCH_SIZE)
        for data in loader:
            if(self.select):
                output = self.policy_net_1(data)
            else:
                with torch.no_grad():
                    output = self.policy_net_1(data)
            #output = policy_net_1(data)
        state_action_values = output[:,action_batch[:,0]].diag()

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE).to(device)#.float()
        loader = DataLoader(non_final_next_states, batch_size=BATCH_SIZE)

        # get batched output
        for data in loader:
            if(self.select):
                with torch.no_grad():
                    output = self.policy_net_2(data).max(1)[0].float()
            else:
               output = self.policy_net_2(data).max(1)[0].float()
            #output = policy_net_2(data).max(1)[0].float()
        next_state_values[non_final_mask] = output

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        return state_action_values.float(), expected_state_action_values.float()


    def compute_gradients(self, weights, select):
        self.select = select

        # Select model to optimize
        if(self.select):
            self.policy_net_1.set_weights(weights)
        else:
            self.policy_net_2.set_weights(weights)

        # Compute prediction and zero gradients
        pred, target = self._get_data()
        if(self.select):
            self.policy_net_1.zero_grad()
        else:
            self.policy_net_2.zero_grad()

        # Compute loss and backprop
        criterion = torch.nn.HuberLoss()
        loss = criterion(pred, target)
        loss.backward()
        handler.add_loss.remote(loss.item())
        if(self.select):
            return self.policy_net_1.get_gradients()
        else:
            return self.policy_net_2.get_gradients()

        
criterion = torch.nn.HuberLoss()
losses = []
def optimize_model():
    if ray.get(memory.size.remote()) < BATCH_SIZE:
    #if ray.get(memory.size.remote()) < BATCH_SIZE:
    #if ray.get(memory.size.remote()) < 10*BATCH_SIZE:
        return

    # Get gradients
    current_weights = ps.get_weights.remote()
    select = ray.get(ps.select.remote())
    gradients = {}
    losses = []
    for worker in workers:
        gradients[worker.compute_gradients.remote(current_weights, select)] = worker

    # Apply gradients
    #for i in range(NUM_WORKERS):
    ready_gradient_list, _ = ray.wait(list(gradients))
    ready_gradient_id = ready_gradient_list[0]
    worker = gradients.pop(ready_gradient_id)

    # Compute and apply gradients.
    current_weights = ps.apply_gradients.remote(*[ready_gradient_id])
    select = ray.get(ps.select.remote())
    #gradients[worker.compute_gradients.remote(current_weights, select)] = worker

    return np.mean(losses)


RESTART = False

# Prefix used for saving results
#PREFIX = 'ys930_ray_parallel_training_'
#PREFIX = 'ys930_ray_small_lr_parallel_batch_training_'
#PREFIX = 'ys930_ray_recreate_'
#PREFIX = 'ys930_ray_8parallel_'
#PREFIX = 'ys930_ray_more_exploit_'
#PREFIX = 'ys930_ray_faster_learning_'
#PREFIX = 'ys930_ray_small_lr_tuning_'
#PREFIX = 'ys930_ray_small_lr_tuning_'

#PREFIX = 'ys930_ray_tuned_'  # NEED TO IMPLEMENT RESTARTING FOR THIS

#PREFIX = 'ys930_ray_original_hyperparameters_'
#PREFIX = 'ys930_ray_try_again_'
#PREFIX = 'ys930_ray_try_again_'
#PREFIX = 'ys930_ray_last_try_'
PREFIX = 'ys930_ray_scheduler_'

# Save directory
save_dir = 'training_results'
if(not os.path.exists("./{}".format(save_dir))):
    os.makedirs(save_dir)
if(not os.path.exists("./{}/{}".format(save_dir, PREFIX[:-1]))):
    os.makedirs(save_dir + "/" + PREFIX[:-1])

save_dir += '/' + PREFIX[:-1]

# Load config
if(RESTART):
    print("./{}/config.yaml".format(save_dir))
    with open("./{}/config.yaml".format(save_dir, PREFIX.split("_")[0]), 'r') as stream:
        flow_config = yaml.safe_load(stream)
    RESTART_NUM = 0
    for f in os.listdir(save_dir):
        RESTART_NUM += int("{}policy_net_1.pt".format(PREFIX) in f)
    print("\n\nRESTART NUM: {}\n\n".format(RESTART_NUM))
else:
    with open("../configs/ray_{}.yaml".format(PREFIX.split("_")[0]), 'r') as stream:
        flow_config = yaml.safe_load(stream)

# Hyperparameters to tune
GAMMA = flow_config['epsilon']['gamma']
EPS_START = flow_config['epsilon']['start']
EPS_END = flow_config['epsilon']['end']
EPS_DECAY = flow_config['epsilon']['decay']

LEARNING_RATE = float(flow_config['optimizer']['lr'])
WEIGHT_DECAY = float(flow_config['optimizer']['weight_decay'])
BATCH_SIZE = int(flow_config['optimizer']['batch_size'])

TARGET_UPDATE = int(flow_config['agent_params']['target_update'])
NUM_WORKERS = int(flow_config['agent_params']['num_workers'])
NUM_PARALLEL = int(flow_config['agent_params']['num_parallel'])

eps_threshs = []

# Save config to directory
if(not RESTART):
    with open(save_dir + "/config.yaml", 'w') as fout:
        yaml.dump(flow_config, fout)
    fout.close()

# Set up environment
env = Env2DAirfoil(flow_config)
env.set_plot_dir(save_dir)
env.plot_state()
flow_config['agent_params']['plot_dir'] = save_dir

# Hold on to ground truth values
flow_config['agent_params']['gt_drag'] = env.gt_drag
flow_config['agent_params']['gt_time'] = env.gt_time
n_actions = flow_config['agent_params']['N_closest']
print("N CLOSEST: {}".format(n_actions))

# Set up for DQN
try:
    NUM_INPUTS = 2 + 3 * int(flow_config['agent_params']['solver_steps']/flow_config['agent_params']['save_steps'])
except:
    NUM_INPUTS = 5

# Set up replay memory and data handler
memory = ReplayMemory.remote(10000)
save_str = "/home/fenics/drl_projects/MeshDQN/parallel_training/{}/{}".format(save_dir, PREFIX)
handler = DataHandler.remote(save_str)

print("Running Asynchronous Parameter Server Training.")
ps = ParameterServer.remote(save_dir, PREFIX)
workers = [DataWorker.remote() for i in range(NUM_WORKERS)]

# Set up training loop
num_episodes = flow_config['agent_params']['episodes']
ep_reward = []
all_actions = []
all_rewards = []
np.random.seed(137)
def train_loop_per_worker(training_config):
    # Sets random seed for each worker
    seed = int(10000*np.random.random())
    np.random.seed(seed)
    random.seed(seed)

    # Get environment
    env = Env2DAirfoil(training_config['env_config'])
    steps_done = ray.get(handler.num_eps.remote())/14
    start_ep = len(ep_reward) if(RESTART) else 0
    for episode in range(start_ep, num_episodes):
        # Analysis
        episode_actions = []
        episode_rewards = []
        episode_losses = []
    
        print("EPISODE: {}".format(episode))
        acc_rew = 0.0
        acc_rews = []
        if(episode != 0):
            env = Env2DAirfoil(flow_config)
    
        state = env.get_state()
        for t in count():
        #for t in tqdm(count()):
            # Action selection isn't random across workers otherwise
            sample = np.random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-steps_done/EPS_DECAY)
            eps_threshs.append(eps_threshold)
            steps_done += 1
            if(sample > eps_threshold): # Exploit
                with torch.no_grad():
                    #print("\n\nSELECTION NETWORK IN ACTION\n\n")
                    action = ray.get(ps.select_action.remote(state)).to(device)
            else: # Explore
                action = torch.tensor([random.sample(range(n_actions+1), 1)], dtype=torch.long).to(device)
    
            #action, eps = select_action(state)
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
                memory.push.remote(state.to(device), action.to(device), next_state.to(device), reward.to(device))
            else:
                memory.push.remote(state.to(device), action.to(device), next_state, reward.to(device))
    
            state = next_state
    
            loss = optimize_model()
            episode_losses.append(loss)
    
            # Add to data handler
            handler.add_eps.remote(eps_threshold)
    
            if(done):
                ep_reward.append(acc_rew)
                break
    
        # Analysis
        handler.add_episode.remote(episode_rewards, episode_actions)
    
        # Handle data and model saving
        if((episode % 5) == 0):
            handler.plot.remote()
        handler.write.remote()
        ps.write.remote()
    

# If using GPUs, use the below scaling config instead.
#scaling_config = ScalingConfig(num_workers=1)
scaling_config = ScalingConfig(num_workers=NUM_PARALLEL)
trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config={'env_config': flow_config},
    scaling_config=scaling_config,
)
result = trainer.fit()

