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
MULTI_SNAPSHOT = True

# TODO: These are misnomers
from parallel_airfoilgcnn import AirfoilGCNN, NodeRemovalNet
#if(not SURROGATE_MODEL):
from ParallelMultiSnapshotEnv2DAirfoil import ParallelMultiSnapshotEnv2DAirfoil as Env2DAirfoil
#elif(MULTI_SNAPSHOT):
#    from OnlineInterpolationEnv2DAirfoil import OnlineInterpolationEnv2DAirfoil as Env2DAirfoil
#else:
#    from Env2DAirfoil import Env2DAirfoil

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
device = torch.device('cpu')


#TODONE: Use last mesh BEFORE reaching accuracy threshold. Significantly better results.
#TODO: Compare interpolated value to actual value on a vertex-by-vertex basis

complete_traj = True
plot_traj = True
end_plots = True

# Try this next
use_best = False

#obj = 'cylinder_multi'
#obj = 'ys930_5000K'
#obj = 'ag11'
#obj = 'ys930_691K'
#obj = 'ys930_multi'
#obj = 'ys930_691K'
#obj = 'ys930_50K'
#obj = 'lwk80120k25_13'
#obj = 'lwk80120k25_ray_scheduler'


#obj = 's1020_ray_scheduler'
#obj = 'nlf415_ray_scheduler'

#obj = 'ys930_ray_scheduler'
#obj = 'ah93w145_ray_scheduler'

#obj = 'rg1495_ray_scheduler'
#obj = 'rg1495_mega_parallel'
#obj = 'rg1495_regular_ray_scheduler'

#obj = 'cylinder_mega_parallel'
obj = 'e654_mega_parallel'
#obj = 'cylinder_mega_paralleler'

RESTART = False
CONFIRM = False
#if(RESTART):
PREFIX = '{0}/{0}_'.format(obj)
#else:
#    PREFIX = '{0}/restart_{0}_'.format(obj)

# Set up environment
with open("./training_results/{}/config.yaml".format(obj), 'r') as stream:
    flow_config = yaml.safe_load(stream)


print(flow_config)
if(SURROGATE_MODEL):
    flow_config['agent_params']['save_steps'] = flow_config['agent_params']['solver_steps']

env = Env2DAirfoil(flow_config)
env.plot_state(title="{} Closest Vertices to Airfoil")
env.flow_solver.deploy()

# Hold on to ground truth values
flow_config['agent_params']['gt_drag'] = env.gt_drag
flow_config['agent_params']['gt_time'] = env.gt_time
flow_config['agent_params']['u'] = [u.copy(deepcopy=True) for u in env.original_u]
flow_config['agent_params']['p'] = [p.copy(deepcopy=True) for p in env.original_p]
n_actions = env.action_space.n

if(SURROGATE_MODEL):
    env.model = joblib.load("./training_results/pretrained_model.joblib")
    env.mean_x = np.load("./training_results/mean_x.npy")
    env.std_x = np.load("./training_results/std_x.npy")
    env.mean_y = np.load("./training_results/mean_y.npy")
    env.std_y = np.load("./training_results/std_y.npy")

# Make deployment directory in results
results_dir = 'training_results'
if(not os.path.exists("./{}/{}/deployed/".format(results_dir, obj))):
    os.makedirs("./{}/{}/deployed/".format(results_dir, obj))

# Confirm results
if(CONFIRM):
    print("JUST CONFIRMING RESULTS")
    if(not os.path.exists("./{}/{}/deployed/confirmed/".format(results_dir, obj))):
        os.makedirs("./{}/{}/deployed/confirmed/".format(results_dir, obj))

net_restarts, d_restarts = "", ""
if(RESTART):
    RESTART_NUM = 0
    for f in os.listdir("./{}/{}/".format(results_dir, obj)):
        RESTART_NUM += int("{}_policy_net_1.pt".format(obj) in f)
    RESTART_NUM -= 1
    print("\n\nRESTART NUM: {}\n\n".format(RESTART_NUM))
    net_restarts, d_restarts = "", ""
    for i in range(RESTART_NUM):
        net_restarts += "restart_"
        d_restarts += "RESTART_"

# Save models, losses, rewards at tiem of deployment
if(RESTART):
    split_PREFIX = PREFIX.split("/")
    print("PREFIX: {}".format(PREFIX))
    print("SPLIT PREFIX: {}".format(split_PREFIX))
    if(CONFIRM):
        shutil.copy("./{}/{}/{}/{}{}losses.npy".format(results_dir, split_PREFIX[0], 'deployed', split_PREFIX[1], d_restarts),
                    "./{}/{}/{}/confirmed/{}{}losses.npy".format(results_dir, obj, 'deployed', obj+"_", d_restarts))
        
        shutil.copy("./{}/{}/{}/{}{}actions.npy".format(results_dir, split_PREFIX[0], 'deployed', split_PREFIX[1], d_restarts),
                "./{}/{}/{}/confirmed/{}{}actions.npy".format(results_dir, obj, 'deployed', obj+"_", d_restarts))

        shutil.copy("./{}/{}/{}/{}{}rewards.npy".format(results_dir, split_PREFIX[0], 'deployed', split_PREFIX[1], d_restarts),
                    "./{}/{}/{}/confirmed/{}{}rewards.npy".format(results_dir, obj, 'deployed', obj+"_", d_restarts))
        
        shutil.copy("./{}/{}/{}/{}{}policy_net_1.pt".format(results_dir, split_PREFIX[0], 'deployed', net_restarts, split_PREFIX[1]),
                    "./{}/{}/{}/confirmed/{}{}policy_net_1.pt".format(results_dir, obj, 'deployed', net_restarts, obj+"_"))
        
        shutil.copy("./{}/{}/{}/{}{}policy_net_2.pt".format(results_dir, split_PREFIX[0], 'deployed', net_restarts, split_PREFIX[1]),
                    "./{}/{}/{}/confirmed/{}{}policy_net_2.pt".format(results_dir, obj, 'deployed', net_restarts, obj+"_"))
    else:
        shutil.copy("./{}/{}{}losses.npy".format(results_dir, PREFIX, d_restarts),
                    "./{}/{}/{}/{}{}losses.npy".format(results_dir, obj, 'deployed', obj+"_", d_restarts))
        
        shutil.copy("./{}/{}{}actions.npy".format(results_dir, PREFIX, d_restarts),
                    "./{}/{}/{}/{}{}actions.npy".format(results_dir, obj, 'deployed', obj+"_", d_restarts))
        
        shutil.copy("./{}/{}{}rewards.npy".format(results_dir, PREFIX, d_restarts),
                    "./{}/{}/{}/{}{}rewards.npy".format(results_dir, obj, 'deployed', obj+"_", d_restarts))
        
        shutil.copy("./{}/{}/{}{}policy_net_1.pt".format(results_dir, split_PREFIX[0], net_restarts, split_PREFIX[1]),
                    "./{}/{}/{}/{}{}policy_net_1.pt".format(results_dir, obj, 'deployed', net_restarts, obj+"_"))
        
        shutil.copy("./{}/{}/{}{}policy_net_2.pt".format(results_dir, split_PREFIX[0], net_restarts, split_PREFIX[1]),
                    "./{}/{}/{}/{}{}policy_net_2.pt".format(results_dir, obj, 'deployed', net_restarts, obj+"_"))
else:
    if(CONFIRM):
        shutil.copy("./{}/{}/{}/{}losses.npy".format(results_dir, obj, 'deployed', obj+"_"),
                    "./{}/{}/{}/confirmed/{}losses.npy".format(results_dir, obj, 'deployed', obj+"_"))
        
        shutil.copy("./{}/{}/{}/{}actions.npy".format(results_dir, obj, 'deployed', obj+"_"),
                    "./{}/{}/{}/confirmed/{}actions.npy".format(results_dir, obj, 'deployed', obj+"_"))
        
        shutil.copy("./{}/{}/{}/{}rewards.npy".format(results_dir, obj, 'deployed', obj+"_"),
                    "./{}/{}/{}/confirmed/{}rewards.npy".format(results_dir, obj, 'deployed', obj+"_"))
        
        shutil.copy("./{}/{}/{}/{}policy_net_1.pt".format(results_dir, obj, 'deployed', obj+"_"),
                    "./{}/{}/{}/confirmed/{}policy_net_1.pt".format(results_dir, obj, 'deployed', obj+"_"))
        
        shutil.copy("./{}/{}/{}/{}policy_net_2.pt".format(results_dir, obj, 'deployed', obj+"_"),
                    "./{}/{}/{}/confirmed/{}policy_net_2.pt".format(results_dir, obj, 'deployed', obj+"_"))
    else:
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
if(RESTART):
    if(CONFIRM):
        policy_net_1.load_state_dict(torch.load("./{}/{}/{}/confirmed/{}{}policy_net_1.pt".format(
                                results_dir, obj, 'deployed', net_restarts, obj+'_'), map_location=torch.device(device)))
    else:
        policy_net_1.load_state_dict(torch.load("./{}/{}/{}/{}{}policy_net_1.pt".format(
                             results_dir, obj, 'deployed', net_restarts, obj+'_'), map_location=torch.device(device)))
else:
    if(CONFIRM):
        policy_net_1.load_state_dict(torch.load("./{}/{}/{}/confirmed/{}policy_net_1.pt".format(
                                results_dir, obj, 'deployed', obj+'_'), map_location=torch.device(device)))
    else:
        policy_net_1.load_state_dict(torch.load("./{}/{}/{}/{}policy_net_1.pt".format(
                                results_dir, obj, 'deployed', obj+'_'), map_location=torch.device(device)))

#raise
#env.model = joblib.load("./{}/{}/{}/{}surrogate_model.joblib".format(
#                        results_dir, obj, 'deployed', obj+"_"))

if(use_best):
    print("\nFOLLOWING BEST TRAJECTORY\n")
    attempt = 0
    while(True):
        try:
            actions = np.load("./{}/{}actions.npy".format(results_dir, PREFIX), allow_pickle=True)
            rewards = np.load("./{}/{}rewards.npy".format(results_dir, PREFIX), allow_pickle=True)
            break
        except OSError:
            attempt += 1
            print("FAILED ATTEMPT: {}".format(attempt))
            pass
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
        if((i+1)%env.save_steps == 0):
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

    print("NAME: {}".format(name))
    plt.savefig("./{}_zoriginal.png".format(name), bbox_inches="tight")
    if(vertex_coord is not None):
        ax.scatter(vertex_coord[0], vertex_coord[1], color='r', s=10)

    plt.savefig("./{}_selected.png".format(name), bbox_inches="tight")
    plt.close()
    

state = env.get_state()
_ = env.calculate_reward()
tactions, vertex_coords = [], []
#traj_vertices, traj_drags, traj_lifts = [len(original_mesh.coordinates())], [gt_drag], [gt_lift]
if(MULTI_SNAPSHOT):
    traj_vertices, traj_drags, traj_lifts = [len(original_mesh.coordinates())], [gt_drag], [gt_lift]

est_traj_vertices = [len(original_mesh.coordinates())]
print(env.new_drags)
#print(env.new_lifts)
if(SURROGATE_MODEL):
    est_drag = [env.new_drags[0][-1]]
else:
    est_drag = [env.new_drags]
    est_lift = [env.new_lifts]

complete_drags = [env.gt_drag]
complete_lifts = [env.gt_lift]

num_steps = flow_config['agent_params']['timesteps']
for t in range(num_steps):
    action = select_action(state)
    if(use_best):
        print("\nFOLLOWING BEST TRAJECTORY\n")
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
        if(end_plots and (t==0)):
            vertex_plot(env.flow_solver.mesh,
            #"./{}/{}/{}/confirmed/{}_{}_mesh_selected".format(results_dir, obj, 'deployed', plt_str, obj),
            "./{}/{}/{}/{}_{}_mesh_selected".format(results_dir, obj, 'deployed', plt_str, obj),
            "{} Mesh".format(obj.split("_")[0].upper()), vertex_coords[-1])
        if(plot_traj):
            if(CONFIRM):
                vertex_plot(env.flow_solver.mesh,
                "./{}/{}/{}/confirmed/{}_{}_mesh_selected".format(results_dir, obj, 'deployed', plt_str, obj),
                "{} Mesh".format(obj.split("_")[0].upper()), vertex_coords[-1])
            else:
                vertex_plot(env.flow_solver.mesh,
                "./{}/{}/{}/{}_{}_mesh_selected".format(results_dir, obj, 'deployed', plt_str, obj),
                "{} Mesh".format(obj), vertex_coords[-1])

    except KeyError:
        print("\nNO REMOVAL\n")
        selected_action = np.nan
        pass

    try:
        next_state, reward, done, _ = env.step(action.item())
    except RuntimeError:
        break

    if(not SURROGATE_MODEL):
        est_drag.append(env.new_drags)
        est_lift.append(env.new_lifts)
    else:
        est_drag.append(env.new_drags[0][-1])
    est_traj_vertices.append(len(env.flow_solver.mesh.coordinates()))
    state = next_state
    print("NUMBER OF VERTICES: {}, DONE: {}".format(
                    len(env.flow_solver.mesh.coordinates()), done))


    # Run simulation if we removed a vertex
    if(complete_traj and (selected_action is not np.nan)):
        d, l, full_drags, full_lifts = run_sim(env)
        if(MULTI_SNAPSHOT):
            traj_drags.append(full_drags)
            traj_lifts.append(full_lifts)
        else:
            traj_drags.append(d)
            traj_lifts.append(l)
        traj_vertices.append(len(env.flow_solver.mesh.coordinates()))
        complete_drags.append(full_drags)
        if(not SURROGATE_MODEL):
            complete_lifts.append(full_lifts)

    tactions.append(selected_action)

    # Save things as we get them
    if(not SURROGATE_MODEL):
        etv = np.array(est_traj_vertices)
        ed = np.array(est_drag)
        el = np.array(est_lift)
        est_data = np.hstack((np.array(est_traj_vertices)[:,np.newaxis], np.array(est_drag), np.array(est_lift)))
    else:
        est_data = np.vstack((est_traj_vertices, est_drag)).T
    if(CONFIRM):
        np.save("./{}/{}/{}/confirmed/{}{}_interpolate_drag_trajectory.npy".format(results_dir, obj, 'deployed', net_restarts, obj), est_data)
    else:
        np.save("./{}/{}/{}/{}{}_interpolate_drag_trajectory.npy".format(results_dir, obj, 'deployed', net_restarts, obj), est_data)


    if(complete_traj):
        if(SURROGATE_MODEL):
            #data = np.vstack((traj_drags, traj_vertices)).T
            data = np.vstack((np.array(traj_vertices), np.array(traj_drags), np.array(traj_lifts))).T
        else:
            tv = np.array(traj_vertices)
            td = np.array(traj_drags)
            tl = np.array(traj_lifts)
            data = np.hstack((tv[:,np.newaxis], td, tl))
        if(CONFIRM):
            np.save("./{}/{}/{}/confirmed/{}{}_drag_trajectory.npy".format(results_dir, obj, 'deployed', net_restarts, obj), data)
        else:
            np.save("./{}/{}/{}/{}{}_drag_trajectory.npy".format(results_dir, obj, 'deployed', net_restarts, obj), data)

    # Check if accuracy threshold has been reached
    best_mesh = Mesh(env.flow_solver.mesh)
    if(done):
        break
    else:
        best_mesh = Mesh(env.flow_solver.mesh)

if(end_plots):
    plt_str = str(len(env.flow_solver.mesh.coordinates()))
    while(len(plt_str) < 8):
        plt_str = '0' + plt_str
    if(len(vertex_coords) > 0):
        vertex_plot(env.flow_solver.mesh,
        "./{}/{}/{}/{}_{}_mesh_selected".format(results_dir, obj, 'deployed', plt_str, obj),
        "{} Mesh".format(obj.split("_")[0].upper()), vertex_coords[-1])
print(est_traj_vertices, est_drag)
if(not SURROGATE_MODEL):
    #est_data = np.vstack((est_traj_vertices, est_drag, est_lift)).T
    est_data = np.hstack((np.array(est_traj_vertices)[:,np.newaxis], np.array(est_drag), np.array(est_lift)))
else:
    est_data = np.vstack((est_traj_vertices, est_drag)).T

if(CONFIRM):
    np.save("./{}/{}/{}/confirmed/{}_interpolate_drag_trajectory.npy".format(results_dir, obj, 'deployed', obj), est_data)
else:
    np.save("./{}/{}/{}/{}_interpolate_drag_trajectory.npy".format(results_dir, obj, 'deployed', obj), est_data)

print("PUTTING MESH BACK")
env.flow_solver.mesh = Mesh(best_mesh) # Set it back to last acceptable mesh
if(complete_traj):
    #data = np.vstack((traj_drags, traj_vertices, traj_lifts)).T
    tv = np.array(traj_vertices)
    td = np.array(traj_drags)
    tl = np.array(traj_lifts)
    data = np.hstack((tv[:,np.newaxis], td, tl))
    for d in data:
        print(d)
    if(CONFIRM):
        np.save("./{}/{}/{}/confirmed/{}_drag_trajectory.npy".format(results_dir, obj, 'deployed', obj), data)
        np.save("./{}/{}/{}/confirmed/{}_complete_drags.npy".format(results_dir, obj, 'deployed', obj), complete_drags)
        np.save("./{}/{}/{}/confirmed/{}_complete_lifts.npy".format(results_dir, obj, 'deployed', obj), complete_lifts)
    else:
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
print("DRAG ERROR:\t{0:.5f}%".format(100*np.abs(new_drag - gt_drag[-1])/np.abs(gt_drag[-1])))

