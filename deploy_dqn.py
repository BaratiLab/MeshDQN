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
import shutil

SURROGATE_MODEL = False

# TODO: These are misnomers
from airfoilgcnn import AirfoilGCNN, NodeRemovalNet
#from Env2DAirfoil import Env2DAirfoil
if(not SURROGATE_MODEL):
    from MultiSnapshotEnv2DAirfoil import MultiSnapshotEnv2DAirfoil as Env2DAirfoil
else:
    from OnlineInterpolationEnv2DAirfoil import OnlineInterpolationEnv2DAirfoil as Env2DAirfoil

from itertools import count

import random

import torch
from torch.nn import functional as F

from collections import namedtuple

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import os

from shapely.geometry import Polygon, Point
import joblib

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if(torch.cuda.is_available()):
    print("USING GPU")
    device = torch.device('cuda:0')
else:
    print("USING CPU")
    device = torch.device('cpu')


#TODONE: Use last mesh BEFORE reaching accuracy threshold. Significantly better results.
#TODO: Compare interpolated value to actual value on a vertex-by-vertex basis

complete_traj = True
plot_traj = True
best = False

#obj = 'cylinder_multi'
#obj = 'ys930_5000K'
#obj = 'ag11'
#obj = 'ys930_691K'
#obj = 'ys930_multi'
#obj = 'ys930_691K'
#obj = 'ys930_50K'
#obj = 'lwk80120k25_13'

#obj = 'ys930_1386'
#obj = 'ys930_1386_small'
#obj = 'ys930_1386_small'
#obj = 'ys930_1386_goal_vertices'
##obj = 'ys930_1386_goal_vertices_snapshots'

if(not SURROGATE_MODEL):
    obj = 'ys930_1386_goal_vertices_snapshots_interp'
else:
    obj = 'ys930_1386_long_online_interp'

RESTART = True
if(RESTART):
    PREFIX = '{0}/{0}_'.format(obj)
else:
    PREFIX = '{0}/restart_{0}_'.format(obj)

# Set up environment
with open("./configs/{}.yaml".format(obj.split("_")[0]), 'r') as stream:
    flow_config = yaml.safe_load(stream)

print(flow_config)
if(SURROGATE_MODEL):
    flow_config['agent_params']['save_steps'] = flow_config['agent_params']['solver_steps']
env = Env2DAirfoil(flow_config)
env.flow_solver.deploy()

# Hold on to ground truth values
flow_config['agent_params']['gt_drag'] = env.gt_drag
flow_config['agent_params']['gt_time'] = env.gt_time
#flow_config['agent_params']['u'] = env.u.copy(deepcopy=True)
#flow_config['agent_params']['p'] = env.p.copy(deepcopy=True)
flow_config['agent_params']['u'] = [u.copy(deepcopy=True) for u in env.original_u]
flow_config['agent_params']['p'] = [p.copy(deepcopy=True) for p in env.original_p]
#obj = 'lwk80120k25_13'

RESTART = True
if(RESTART):
    PREFIX = '{0}/{0}_'.format(obj)
else:
    PREFIX = '{0}/restart_{0}_'.format(obj)

# Set up environment
with open("./configs/{}.yaml".format(obj.split("_")[0]), 'r') as stream:
    flow_config = yaml.safe_load(stream)
env = Env2DAirfoil(flow_config)
env.flow_solver.deploy()

# Hold on to ground truth values
flow_config['agent_params']['gt_drag'] = env.gt_drag
flow_config['agent_params']['gt_time'] = env.gt_time
#flow_config['agent_params']['u'] = env.u.copy(deepcopy=True)
#flow_config['agent_params']['p'] = env.p.copy(deepcopy=True)
flow_config['agent_params']['u'] = [u.copy(deepcopy=True) for u in env.original_u]
flow_config['agent_params']['p'] = [p.copy(deepcopy=True) for p in env.original_p]
n_actions = env.action_space.n

env.model = joblib.load("./training_results/pretrained_model.joblib")
env.mean_x = np.load("./training_results/mean_x.npy")
env.std_x = np.load("./training_results/std_x.npy")
env.mean_y = np.load("./training_results/mean_y.npy")
env.std_y = np.load("./training_results/std_y.npy")

# Make deployment directory in results
results_dir = 'training_results'
if(not os.path.exists("./{}/{}/deployed/".format(results_dir, obj))):
    os.makedirs("./{}/{}/deployed/".format(results_dir, obj))

# Save models, losses, rewards at tiem of deployment
shutil.copy("./{}/{}losses.npy".format(results_dir, PREFIX),
            "./{}/{}/{}/{}losses.npy".format(results_dir, obj, 'deployed', obj+"_"))

shutil.copy("./{}/{}actions.npy".format(results_dir, PREFIX),
            "./{}/{}/{}/{}actions.npy".format(results_dir, obj, 'deployed', obj+"_"))

shutil.copy("./{}/{}rewards.npy".format(results_dir, PREFIX),
            "./{}/{}/{}/{}rewards.npy".format(results_dir, obj, 'deployed', obj+"_"))

shutil.copy("./{}/{}policy_net_1.pt".format(results_dir, PREFIX),
            "./{}/{}/{}/{}policy_net_1.pt".format(results_dir, obj, 'deployed', obj+"_"))

shutil.copy("./{}/{}policy_net_2.pt".format(results_dir, PREFIX),
            "./{}/{}/{}/{}policy_net_2.pt".format(results_dir, obj, 'deployed', obj+"_"))

if(SURROGATE_MODEL):
    shutil.copy("./{}/{}surrogate_model.joblib".format(results_dir, PREFIX),
        "./{}/{}/{}/{}surrogate_model.joblib".format(results_dir, obj, 'deployed', obj+"_"))

# Set up for DQN
#policy_net_1 = NodeRemovalNet(n_actions+1, conv_width=256, topk=0.1).to(device).float()
policy_net_1 = NodeRemovalNet(n_actions+1, conv_width=128, topk=0.1).to(device).float()
#policy_net_1 = NodeRemovalNet(n_actions+1, conv_width=256, topk=0.5).to(device).float()
def select_action(state):
    return torch.tensor([[policy_net_1(state).argmax()]]).to(device)

# Prime and load policy net
try:
    NUM_INPUTS = 2 + 3 * int(flow_config['agent_params']['solver_steps']/flow_config['agent_params']['save_steps'])
except:
    NUM_INPUTS = 5
policy_net_1.set_num_nodes(NUM_INPUTS)
#policy_net_2.set_num_nodes(NUM_INPUTS)
#policy_net_1.set_num_nodes(env.initial_num_node)
#policy_net_1.set_num_nodes(5)
policy_net_1.load_state_dict(torch.load("./{}/{}/{}/{}policy_net_1.pt".format(
                             results_dir, obj, 'deployed', obj+'_'), map_location=torch.device(device)))
#env.model = joblib.load("./{}/{}/{}/{}surrogate_model.joblib".format(
#                        results_dir, obj, 'deployed', obj+"_"))

if(best):
    actions = np.load("./{}/{}actions.npy".format(results_dir, PREFIX), allow_pickle=True)
    rewards = np.load("./{}/{}rewards.npy".format(results_dir, PREFIX), allow_pickle=True)
    ep_rews = np.empty(len(rewards))
    for idx, r in enumerate(rewards):
        ep_rews[idx] = np.sum(r)
    
    worst = np.argmin(ep_rews)
    best = np.argmax(ep_rews)


# Ground truth 
gt_drag = env.gt_drag
gt_lift = env.gt_lift
gt_time = env.gt_time

# Get meshes
original_mesh = Mesh(env.flow_solver.mesh)
best_mesh = Mesh(env.flow_solver.mesh)

# Fluid simulation function
def run_sim(env):
    drags, lifts = [], []
    for i in tqdm(range(env.solver_steps)):
        u, p, drag, lift = env.flow_solver.evolve()
        drags.append(drag)
        lifts.append(lift)
    return drags[-1], lifts[-1], drags, lifts


def vertex_plot(mesh, name="mesh", title=None, vertex_coord=None):
    coords = mesh.coordinates()
    edges = []
    for c in mesh.cells():
        edges.append([c[0], c[1]])
        edges.append([c[0], c[2]])
        edges.append([c[1], c[2]])

    fig, ax = plt.subplots(figsize=(8,4))
    ax.scatter(coords[:,0], coords[:,1], color='k', s=5, zorder=1)
    for e in edges:
        ax.plot([coords[e[0]][0], coords[e[1]][0]],
                [coords[e[0]][1], coords[e[1]][1]],
                color="#888888", lw=0.5, zorder=0)

    if(title):
        ax.set_title(title + ": {} Vertices".format(len(mesh.coordinates())), fontsize=14)

    plt.savefig("./{}_zoriginal.png".format(name), bbox_inches="tight")
    if(vertex_coord is not None):
        ax.scatter(vertex_coord[0], vertex_coord[1], color='r', s=10)

    plt.savefig("./{}_selected.png".format(name), bbox_inches="tight")
    plt.close()
    

state = env.get_state()
_ = env.calculate_reward()
tactions, vertex_coords = [], []
#traj_vertices, traj_drags, traj_lifts = [len(original_mesh.coordinates())], [gt_drag], [gt_lift]
traj_vertices, traj_drags, traj_lifts = [len(original_mesh.coordinates())], [gt_drag[-1]], [gt_lift[-1]]

est_traj_vertices = [len(original_mesh.coordinates())]
print(env.new_drags)
#print(env.new_lifts)
if(SURROGATE_MODEL):
    est_drag = [env.new_drags[0][-1]]
else:
    est_drag = [env.new_drags[-1]]
    est_lift = [env.new_lifts[-1]]
complete_drags = []
complete_lifts = []

num_steps = flow_config['agent_params']['timesteps']
for t in range(num_steps):
    action = select_action(state)
    if(best):
        try:
            action = torch.tensor(actions[best][t])
            #action = torch.tensor(actions[worst][t])
        except IndexError:
            break

    print("ACTION {}: {}".format(t, action.item()))

    try:
        selected_action = env.coord_map[action.item()]
        vertex_coords.append(env.flow_solver.mesh.coordinates()[selected_action])

        # Plot it if we removed a vertex
        plt_str = str(len(env.flow_solver.mesh.coordinates()))
        while(len(plt_str) < 8):
            plt_str = '0' + plt_str
        if(plot_traj):
            vertex_plot(env.flow_solver.mesh,
            "./{}/{}/{}/{}_{}_mesh_selected".format(results_dir, obj, 'deployed', plt_str, obj),
            "{} Mesh".format(obj), vertex_coords[-1])

    except KeyError:
        print("\nNO REMOVAL\n")
        selected_action = np.nan
        pass

    next_state, reward, done, _ = env.step(action.item())
    #est_drag.append(env.new_drag)
    #est_lift.append(env.new_lift)
    if(not SURROGATE_MODEL):
        est_drag.append(env.new_drags[-1])
        est_lift.append(env.new_lifts[-1])
    else:
        est_drag.append(env.new_drags[0][-1])
    est_traj_vertices.append(len(env.flow_solver.mesh.coordinates()))
    state = next_state
    print("NUMBER OF VERTICES: {}, DONE: {}".format(
                    len(env.flow_solver.mesh.coordinates()), done))

    # Check if accuracy threshold has been reached
    if(done):
        break
    else:
        best_mesh = Mesh(env.flow_solver.mesh)

    # Run simulation if we removed a vertex
    if(complete_traj and (selected_action is not np.nan)):
        d, l, full_drags, full_lifts = run_sim(env)
        traj_drags.append(d)
        traj_lifts.append(l)
        traj_vertices.append(len(env.flow_solver.mesh.coordinates()))
        complete_drags.append(full_drags)
        if(not SURROGATE_MODEL):
            complete_lifts.append(full_lifts)

    tactions.append(selected_action)

    # Save things as we get them
    if(not SURROGATE_MODEL):
        est_data = np.vstack((est_traj_vertices, est_drag, est_lift)).T
    else:
        est_data = np.vstack((est_traj_vertices, est_drag)).T
    np.save("./{}/{}/{}/{}_interpolate_drag_trajectory.npy".format(
            results_dir, obj, 'deployed', obj), est_data)
    if(complete_traj):
        if(SURROGATE_MODEL):
            data = np.vstack((traj_drags, traj_vertices)).T
        else:
            data = np.vstack((traj_drags, traj_vertices, traj_lifts)).T
        np.save("./{}/{}/{}/{}_drag_trajectory.npy".format(results_dir, obj, 'deployed', obj), data)

print(est_traj_vertices, est_drag)
if(not SURROGATE_MODEL):
    est_data = np.vstack((est_traj_vertices, est_drag, est_lift)).T
else:
    est_data = np.vstack((est_traj_vertices, est_drag)).T
np.save("./{}/{}/{}/{}_interpolate_drag_trajectory.npy".format(results_dir, obj, 'deployed', obj), est_data)

print("PUTTING MESH BACK")
env.flow_solver.mesh = Mesh(best_mesh) # Set it back to last acceptable mesh
if(complete_traj):
    data = np.vstack((traj_drags, traj_vertices, traj_lifts)).T
    for d in data:
        print(d)
    np.save("./{}/{}/{}/{}_drag_trajectory.npy".format(results_dir, obj, 'deployed', obj), data)
    np.save("./{}/{}/{}/{}_complete_drags.npy".format(results_dir, obj, 'deployed', obj), complete_drags)
    np.save("./{}/{}/{}/{}_complete_lifts.npy".format(results_dir, obj, 'deployed', obj), complete_lifts)

def plot_mesh(mesh, name="mesh", title=None, actions=None, vertex_coords=None):
    coords = mesh.coordinates()
    edges = []
    for c in mesh.cells():
        edges.append([c[0], c[1]])
        edges.append([c[0], c[2]])
        edges.append([c[1], c[2]])

    fig, ax = plt.subplots(figsize=(8,4))
    ax.scatter(coords[:,0], coords[:,1], color='k', s=5, zorder=1)
    for e in edges:
        ax.plot([coords[e[0]][0], coords[e[1]][0]],
                [coords[e[0]][1], coords[e[1]][1]],
                color="#888888", lw=0.5, zorder=0)

    if(vertex_coords is not None):
        for v in vertex_coords:
            ax.scatter(v[0], v[1], color='r', s=10)

    if(title):
        ax.set_title(title, fontsize=14)

    plt.savefig("./{}.png".format(name), bbox_inches="tight")
    plt.close()

print("INITIAL NUMBER OF VERTICES: {}".format(len(original_mesh.coordinates())))
print("ACTIONS: ", tactions)
print("NUMER OF VERTICES REMOVED: {}".format(len(tactions)))
print("FINAL NUMBER OF VERTICES: {}".format(len(env.flow_solver.mesh.coordinates())))

# New mesh
drags = []
start = time.time()
env.flow_solver.remesh(Mesh(best_mesh))
with Timer() as t:
    for i in tqdm(range(env.solver_steps)):
        u, p, drag, lift = env.flow_solver.evolve()
        drags.append(drag)
print(t.elapsed())
new_drag = np.mean(drags[-1])
new_time = time.time() - start

###
# TODO: There is something wrong in going from vertex selection -> saving action
###
print("INITIAL NUMBER OF VERTICES: {}".format(len(original_mesh.coordinates())))
print("NUMER OF VERTICES REMOVED: {}".format(len(np.unique(tactions))))
print("FINAL NUMBER OF VERTICES: {}".format(len(env.flow_solver.mesh.coordinates())))
print(gt_drag)
print(gt_time)
print("GROUND TRUTH DRAG:\t{0:.6f}\tGROUND TRUTH TIME:\t{1:.6f}".format(gt_drag[-1], gt_time[-1]))
print("NEW DRAG:\t\t{0:.6f}\tNEW TIME:\t\t{1:.6f}".format(new_drag, new_time))

