import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.lines import Line2D
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
import os

SHOW_INTERPOLATION = True

# New ys930 drag trajectory plots
if(True):
    save_dir = "ys930_1386_long_interp"
    data = pd.read_csv("./benchmark_results/smooth_ys930_1.0_0.001_smooth_benchmark.csv")
    
    # Screen coarse meshes since they have inconsistent results
    big_data = data[data['NUM_COORDS'] > 1500]
    data = data[data['NUM_COORDS'] < 1200]
    #data = data[data['NUM_COORDS'] < 3000]
    
    model = LinearRegression()
    model.fit(big_data['NUM_COORDS'].values[:,np.newaxis], big_data['DRAG'].values)
    
    xs = np.linspace(150, 2000)

    # Mask training airfoil
    target_idx = 3
    mask = [True]*len(data["NUM_COORDS"])
    mask[target_idx] = False
    fig, ax = plt.subplots(figsize=(10,8))

    # Plot all values
    ax.scatter(data["NUM_COORDS"][mask], data["DRAG"][mask].abs(), marker='s',
               edgecolor='k', lw=3,
               s=100, label="Computed Airfoils", color='steelblue')

    # Plot training airfoil
    ax.axhline(np.abs(big_data['DRAG'].values[0]), color='#888888', lw=2, linestyle='--',
            label="Converged Value")

    # Load drag trajectory and plot
    drag_traj = np.load("./{}/deployed/{}_drag_trajectory.npy".format(save_dir, save_dir), allow_pickle=True)
    est_drag_traj = np.load("./{}/deployed/{}_interpolate_drag_trajectory.npy".format(save_dir, save_dir), allow_pickle=True)

    # Plot drag draj
    ax.plot(drag_traj[:,1], np.abs(drag_traj[:,0]), zorder=-1, label="Refinement Path", color='r', lw=1.5)
    if(SHOW_INTERPOLATION):
        ax.plot(est_drag_traj[:,0], np.abs(est_drag_traj[:,1]), zorder=-1, label="Interpolated Refinement Path", color='g', lw=1.5)

    # Last value is special
    ax.scatter(drag_traj[:,1][-2], np.abs(drag_traj[:,0][-2]), marker='*', s=200,
               color='goldenrod', edgecolor='k', lw=1.5, label="Refined Airfoil")

    # Original value
    ax.scatter(drag_traj[0][1],
               np.abs(drag_traj[0][0]), marker='p', lw=3,
               s=200, label="Original Airfoil", edgecolor='k', color='magenta')

    # Add zoomed section
    axins = zoomed_inset_axes(ax, zoom=5, loc='upper right', bbox_to_anchor=(1125,650))
    for axis in ['top', 'bottom', 'left', 'right']:
        axins.spines[axis].set_linewidth(2)

    axins.scatter(data["NUM_COORDS"][mask], data["DRAG"][mask].abs(), marker='s', edgecolor='k', lw=3,
               s=100, label="Computed Airfoils")
    axins.scatter(drag_traj[0][1],
               np.abs(drag_traj[0][0]), marker='p', lw=3,
               s=200, label="Original Airfoil", edgecolor='k', color='magenta')
    axins.plot(est_drag_traj[:,0],
               np.abs(est_drag_traj[:,1]), zorder=-1, color='g')
    axins.axhline(np.abs(big_data['DRAG'].values[0]), color='#888888', lw=2, linestyle='--')

    if(SHOW_INTERPOLATION):
        axins.axhline(1.001*np.abs(data['DRAG'].values[target_idx]), color='#aaaaaa', lw=2, linestyle='--')
        axins.axhline(0.999*np.abs(data['DRAG'].values[target_idx]), color='#aaaaaa', lw=2, linestyle='--')

    axins.axhline(np.abs(drag_traj[0][0]), color='orchid', lw=2, linestyle='--', label="Original Value", zorder=-1)
    axins.plot(drag_traj[:,1], np.abs(drag_traj[:,0]), zorder=-1, label="Refinement Path", color='r', lw=1.5)
    axins.plot(est_drag_traj[:,0], np.abs(est_drag_traj[:,1]), zorder=-1, label="Refinement Path", color='g', lw=1.5)
    axins.scatter(drag_traj[:,1][-2], np.abs(drag_traj[:,0][-2]), marker='*', s=200,
               color='goldenrod', edgecolor='k', lw=1.5, label="Refined Airfoil")

    axins.set_xticks([i for i in np.arange(730, 890, 50)])
    axins.set_yticks([i for i in np.arange(0.1127, 0.1135, 0.0004)])
    axins.set_yticklabels(["{0:.4f}".format(i) for i in np.arange(0.1127, 0.1135, 0.0004)], rotation=30)

    x1, x2, y1, y2 = 730, 890, 0.1126, 0.1135
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    axins.yaxis.get_major_locator().set_params(nbins=7)
    axins.xaxis.get_major_locator().set_params(nbins=7)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.2", lw=2)

    # Labels
    ax.set_xlabel("Number of Vertices", fontsize=20)
    ax.set_ylabel("Drag", fontsize=20)
    ax.set_title("Drag Improvement After MeshDQN Training", fontsize=26, y=1.01)

    # Tick labels
    ax.set_xticks([i for i in np.arange(100, 1201, 100)])
    ax.set_xticklabels([str(i) for i in np.arange(100, 1201, 100)], fontsize=14)

    custom_lines = [
            Line2D([0], [0], color='steelblue', marker='s', markeredgecolor='k', markeredgewidth=3, markersize=10, lw=0),
            Line2D([0], [0], color='magenta', marker='p', markeredgecolor='k', markeredgewidth=3, markersize=15, lw=0),
            Line2D([0], [0], color='goldenrod', marker='*', markeredgecolor='k', markeredgewidth=2, markersize=20, lw=0),
            Line2D([0], [0], color='#888888', lw=2, linestyle='--'),
            Line2D([0], [0], color='orchid', lw=2, linestyle='--'),
            Line2D([0], [0], color='red', lw=2),
            Line2D([0], [0], color='#aaaaaa', lw=2, linestyle='--'),
            Line2D([0], [0], color='g', lw=2)
    ]
    if(not SHOW_INTERPOLATION):
        labels = ["Computed Airfoils", "Original Airfoil", "Refined Airfoil", "Converged Value",
                "Original Value", "Refinement Path"]
        lgd = ax.legend(custom_lines, labels, fontsize=16, ncol=3, bbox_to_anchor=(1.06, -0.1))
    else:
        labels = ["Computed Airfoils", "Original Airfoil", "Refined Airfoil", "Converged Value",
                  "Original Value", "Refinement Path", "Error Threshold", "Interpolation Path"]
        lgd = ax.legend(custom_lines, labels, fontsize=16, ncol=4, bbox_to_anchor=(1.5, -0.1))
    plt.savefig("./{}/deployed/{}_drag_improvement.png".format(save_dir, save_dir),
                bbox_extra_artists=(lgd, axins), bbox_inches='tight')

    print()
    print("INITIAL DRAG:\t{0:.8f}".format(drag_traj[0][0]))
    print("FINAL DRAG:\t{0:.8f}".format(drag_traj[-1][0]))
    print("DRAG ERROR:\t{0:.5f}%".format(100*np.abs(drag_traj[0][0] - drag_traj[-2][0])/np.abs(drag_traj[0][0])))
    print()
    print("INITIAL LIFT:\t{0:.8f}".format(drag_traj[0][2]))
    print("FINAL LIFT:\t{0:.8f}".format(drag_traj[-1][2]))
    print("LIFT ERROR:\t{0:.5f}%".format(100*np.abs(drag_traj[0][2] - drag_traj[-2][2])/np.abs(drag_traj[0][2])))
    print()
    print("INITIAL VERTICES:\t{0:.5f}".format(drag_traj[0][1]))
    print("FINAL VERTICES:\t{0:.5f}".format(drag_traj[-1][1]))
    print("VERTICES PERCENT: {0:.3f}%".format(100*(1-drag_traj[-2][1]/drag_traj[0][1])))
    #plt.show()


# New ys930 lift trajectory plots
if(True):
    save_dir = "ys930_1386_long_interp"
    data = pd.read_csv("./benchmark_results/smooth_ys930_1.0_0.001_smooth_benchmark.csv")
    
    # Screen coarse meshes since they have inconsistent results
    big_data = data[data['NUM_COORDS'] > 1500]
    data = data[data['NUM_COORDS'] < 1200]
    #data = data[data['NUM_COORDS'] < 3000]
    
    model = LinearRegression()
    model.fit(big_data['NUM_COORDS'].values[:,np.newaxis], big_data['DRAG'].values)
    
    xs = np.linspace(150, 2000)

    # Mask training airfoil
    target_idx = 3
    mask = [True]*len(data["NUM_COORDS"])
    mask[target_idx] = False
    fig, ax = plt.subplots(figsize=(10,8))

    # Plot all values
    ax.scatter(data["NUM_COORDS"][mask], data["LIFT"][mask].abs(), marker='s',
               edgecolor='k', lw=3,
               s=100, label="Computed Airfoils", color='steelblue')

    # Plot training airfoil
    ax.axhline(np.abs(big_data['LIFT'].values[0]), color='#888888', lw=2, linestyle='--',
            label="Converged Value")

    # Load drag trajectory and plot
    drag_traj = np.load("./{}/deployed/{}_drag_trajectory.npy".format(save_dir, save_dir), allow_pickle=True)
    est_drag_traj = np.load("./{}/deployed/{}_interpolate_drag_trajectory.npy".format(save_dir, save_dir), allow_pickle=True)

    # Plot drag draj
    ax.plot(drag_traj[:,1], np.abs(drag_traj[:,2]), zorder=-1, label="Refinement Path", color='r', lw=1.5)
    if(SHOW_INTERPOLATION):
        ax.plot(est_drag_traj[:,0], np.abs(est_drag_traj[:,2]), zorder=-1, label="Interpolated Refinement Path", color='g', lw=1.5)

    # Last value is special
    ax.scatter(drag_traj[:,1][-2], np.abs(drag_traj[:,2][-2]), marker='*', s=200,
               color='goldenrod', edgecolor='k', lw=1.5, label="Refined Airfoil")

    # Original value
    ax.scatter(drag_traj[0][1],
               np.abs(drag_traj[0][2]), marker='p', lw=3,
               s=200, label="Original Airfoil", edgecolor='k', color='magenta')

    # Add zoomed section
    #axins = zoomed_inset_axes(ax, zoom=5, loc='upper right', bbox_to_anchor=(1125,650))
    axins = zoomed_inset_axes(ax, zoom=3, loc='upper right', bbox_to_anchor=(1050,375))
    for axis in ['top', 'bottom', 'left', 'right']:
        axins.spines[axis].set_linewidth(2)

    axins.scatter(data["NUM_COORDS"][mask], data["LIFT"][mask].abs(), marker='s', edgecolor='k', lw=3,
               s=100, label="Computed Airfoils")
    axins.scatter(drag_traj[0][1],
               np.abs(drag_traj[0][2]), marker='p', lw=3,
               s=200, label="Original Airfoil", edgecolor='k', color='magenta')
    axins.plot(est_drag_traj[:,0],
               np.abs(est_drag_traj[:,2]), zorder=-1, color='g')
    axins.axhline(np.abs(big_data['DRAG'].values[0]), color='#888888', lw=2, linestyle='--')

    if(SHOW_INTERPOLATION):
        axins.axhline(1.001*np.abs(data['DRAG'].values[target_idx]), color='#aaaaaa', lw=2, linestyle='--')
        axins.axhline(0.999*np.abs(data['DRAG'].values[target_idx]), color='#aaaaaa', lw=2, linestyle='--')

    axins.axhline(np.abs(drag_traj[0][2]), color='orchid', lw=2, linestyle='--', label="Original Value", zorder=-1)
    axins.plot(drag_traj[:,1], np.abs(drag_traj[:,2]), zorder=-1, label="Refinement Path", color='r', lw=1.5)
    axins.plot(est_drag_traj[:,0], np.abs(est_drag_traj[:,2]), zorder=-1, label="Refinement Path", color='g', lw=1.5)
    axins.scatter(drag_traj[:,1][-2], np.abs(drag_traj[:,2][-2]), marker='*', s=200,
               color='goldenrod', edgecolor='k', lw=1.5, label="Refined Airfoil")

    axins.set_xticks([i for i in np.arange(730, 890, 50)])
    axins.set_yticks([i for i in np.arange(0.1127, 0.1135, 0.0004)])
    axins.set_yticklabels(["{0:.4f}".format(i) for i in np.arange(0.1127, 0.1135, 0.0004)], rotation=30)

    x1, x2, y1, y2 = 730, 890, 0.0455, 0.0477
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    axins.yaxis.get_major_locator().set_params(nbins=7)
    axins.xaxis.get_major_locator().set_params(nbins=7)
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.2", lw=2)

    # Labels
    ax.set_xlabel("Number of Vertices", fontsize=20)
    ax.set_ylabel("Drag", fontsize=20)
    ax.set_title("Drag Improvement After MeshDQN Training", fontsize=26, y=1.01)

    # Tick labels
    axins.set_xticks([i for i in np.arange(730, 890, 50)])
    axins.set_yticks([i for i in np.arange(0.0455, 0.0475, 0.001)])
    axins.set_yticklabels(["{0:.4f}".format(i) for i in np.arange(0.0455, 0.0475, 0.001)], rotation=30)

    custom_lines = [
            Line2D([0], [0], color='steelblue', marker='s', markeredgecolor='k', markeredgewidth=3, markersize=10, lw=0),
            Line2D([0], [0], color='magenta', marker='p', markeredgecolor='k', markeredgewidth=3, markersize=15, lw=0),
            Line2D([0], [0], color='goldenrod', marker='*', markeredgecolor='k', markeredgewidth=2, markersize=20, lw=0),
            Line2D([0], [0], color='#888888', lw=2, linestyle='--'),
            Line2D([0], [0], color='orchid', lw=2, linestyle='--'),
            Line2D([0], [0], color='red', lw=2),
            Line2D([0], [0], color='#aaaaaa', lw=2, linestyle='--'),
            Line2D([0], [0], color='g', lw=2)
    ]
    if(not SHOW_INTERPOLATION):
        labels = ["Computed Airfoils", "Original Airfoil", "Refined Airfoil", "Converged Value",
                "Original Value", "Refinement Path"]
        lgd = ax.legend(custom_lines, labels, fontsize=16, ncol=3, bbox_to_anchor=(1.06, -0.1))
    else:
        labels = ["Computed Airfoils", "Original Airfoil", "Refined Airfoil", "Converged Value",
                  "Original Value", "Refinement Path", "Error Threshold", "Interpolation Path"]
        lgd = ax.legend(custom_lines, labels, fontsize=16, ncol=4, bbox_to_anchor=(1.5, -0.1))
    plt.savefig("./{}/deployed/{}_lift_improvement.png".format(save_dir, save_dir),
                bbox_extra_artists=(lgd, axins), bbox_inches='tight')

    print()
    print("INITIAL DRAG:\t{0:.8f}".format(drag_traj[0][0]))
    print("FINAL DRAG:\t{0:.8f}".format(drag_traj[-1][0]))
    print("DRAG ERROR:\t{0:.5f}%".format(100*np.abs(drag_traj[0][0] - drag_traj[-2][0])/np.abs(drag_traj[0][0])))
    print()
    print("INITIAL LIFT:\t{0:.8f}".format(drag_traj[0][2]))
    print("FINAL LIFT:\t{0:.8f}".format(drag_traj[-1][2]))
    print("LIFT ERROR:\t{0:.5f}%".format(100*np.abs(drag_traj[0][2] - drag_traj[-2][2])/np.abs(drag_traj[0][2])))
    print()
    print("INITIAL VERTICES:\t{0:.5f}".format(drag_traj[0][1]))
    print("FINAL VERTICES:\t{0:.5f}".format(drag_traj[-1][1]))
    print("VERTICES PERCENT: {0:.3f}%".format(100*(1-drag_traj[-2][1]/drag_traj[0][1])))
    #plt.show()


# Drag reward one sided
if(False):
    fig, ax = plt.subplots(figsize=(10,8))
    xs = np.linspace(100., 100.2, 1001)

    f = lambda x: 2*np.exp(-50*np.abs(100 - x)/np.abs(100.)) - 1
    ax.plot(xs, f(xs), lw=2, color='sienna', label=r"$K=50$")
            #label=r"$2\exp\left(\frac{-50\left|d_{gt} - d_{n} \right|}{\left|d_{gt} \right|}\right) - 1$")

    f = lambda x: 2*np.exp(-1386/2*np.abs(100 - x)/np.abs(100.)) - 1
    ax.plot(xs, f(xs), lw=2, color='k', label=r"$K=691$")
            #label=r"$2\exp\left(\frac{-691\left|d_{gt} - d_{n} \right|}{\left|d_{gt} \right|}\right) - 1$")
    f = lambda x: 2*np.exp(-1386*np.abs(100 - x)/np.abs(100.)) - 1
    ax.plot(xs, f(xs), lw=2, color='b', label=r"$K=1386$")
            #label=r"$2\exp\left(\frac{-1386\left|d_{gt} - d_{n} \right|}{\left|d_{gt} \right|}\right) - 1$")

    f = lambda x: 2*np.exp(-5000*np.abs(100 - x)/np.abs(100.)) - 1
    ax.plot(xs, f(xs), lw=2, color='g', label=r"$K=5000$")
            #label=r"$2\exp\left(\frac{-5000\left|d_{gt} - d_{n} \right|}{\left|d_{gt} \right|}\right) - 1$")

    #f = lambda x: 2*np.exp(-50*np.abs(100 - x)/np.abs(100.)) - 1
    #ax.plot(xs, f(xs), lw=2, color='k')

    ax.axvline(100.05, color='lightcoral', lw=2, label="Half Accuuracy Threshold")
    #ax.axvline(99.95, color='lightcoral', lw=2, label="Half Accuracy Threshold")
    ax.axvline(100.1, color='red', label="Accuracy Threshold", lw=2)
    #ax.axvline(99.9, color='red', lw=2)
    ax.axhline(0, color='#888888', lw=2, linestyle='--')#, label="0 Drag Reward")

    #ax.set_xticks([i for i in np.arange(99.8, 100.2, 0.05)])
    ax.set_xticks([i for i in np.arange(100, 100.2, 0.05)])
    #ax.set_xticklabels(["{0:.2f}".format(i) for i in np.arange(99.8, 100.2, 0.05)], fontsize=14)
    ax.set_xticklabels(["{0:.2f}".format(i) for i in np.arange(0, 0.21, 0.05)], fontsize=14)
    
    ax.set_yticks([i for i in np.arange(-1, 1.1, 0.25)])
    ax.set_yticklabels(["{0:.2f}".format(i) for i in np.arange(-1, 1.1, 0.25)], fontsize=14)

    ax.set_xlabel("% Error", fontsize=20)
    ax.set_ylabel("Drag Reward Component", fontsize=20)
    ax.set_title("Drag Component of Reward Function", fontsize=26)
    #ax.legend(bbox_to_anchor=(1.2, 1.0), fontsize=14)
    ax.legend(loc='center right', fontsize=16, framealpha=1, bbox_to_anchor=(0.99, 0.7))
    plt.tight_layout()
    plt.savefig("./drag_reward.png")
    #plt.show()

