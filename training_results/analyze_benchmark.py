import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.lines import Line2D
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
import os
from matplotlib.ticker import FormatStrFormatter

SHOW_INTERPOLATION = True
save_dir = "ys930_results"
#save_dir = "ah93w145_results"

ys930 = "ys930" in save_dir
s1020 = "e654" in save_dir
#s1020 = "ah93w145" in save_dir
FINAL_IDX = -1
RESTART = True
FINAL = True

# New ys930 drag trajectory plots
if(True):
    if(ys930):
        data = pd.read_csv("./benchmark_results/smooth_ys930_1.0_0.001_smooth_benchmark.csv")
    else:
        if(s1020):
            #data = pd.read_csv("./benchmark_results/smooth_s1020_1.0_0.001_smooth_benchmark.csv")
            #data = pd.read_csv("./benchmark_results/smooth_nlf415_1.0_0.001_smooth_benchmark.csv")
            #data = pd.read_csv("./benchmark_results/smooth_rg1495_1.0_0.001_smooth_benchmark.csv")
            data = pd.read_csv("./benchmark_results/smooth_e654_1.0_0.001_smooth_benchmark.csv")
        else:
            data = pd.read_csv("./benchmark_results/smooth_ah93w145_1.0_0.001_smooth_benchmark.csv")

        median = data['DRAG'].median()
        std = data['DRAG'].std()
        #data = data[np.abs(data['DRAG']) < 2.5*std + np.abs(median)]
        data = data[np.abs(data['DRAG']) < 1.5*std + np.abs(median)]

    
    # Screen coarse meshes since they have inconsistent results
    big_data = data[data['NUM_COORDS'] > 1500]
    if(s1020):
        data = data[data['NUM_COORDS'] < 1200]
    else:
        data = data[data['NUM_COORDS'] < 1200]
    #data = data[data['NUM_COORDS'] < 3000]
    
    model = LinearRegression()
    model.fit(big_data['NUM_COORDS'].values[:,np.newaxis], big_data['DRAG'].values)
    
    xs = np.linspace(150, 2000)

    # Mask training airfoil
    if(ys930):
        target_idx = 3
    else:
        if(s1020):
            target_idx = 7
        else:
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
    if(RESTART):
        if(FINAL):
            drag_traj = np.load("./{}/deployed/confirmed/restart_{}_drag_trajectory.npy".format(save_dir, save_dir), allow_pickle=True)
            est_drag_traj = np.load("./{}/deployed/confirmed/restart_{}_interpolate_drag_trajectory.npy".format(save_dir, save_dir),
                                allow_pickle=True)
            #drag_traj = np.load("./FINAL_RESULT_{}/deployed/confirmed/restart_{}_drag_trajectory.npy".format(save_dir, save_dir), allow_pickle=True)
            #est_drag_traj = np.load("./FINAL_RESULT_{}/deployed/confirmed/restart_{}_interpolate_drag_trajectory.npy".format(save_dir, save_dir),
            #                    allow_pickle=True)
        else:
            drag_traj = np.load("./{}/deployed/restart_{}_drag_trajectory.npy".format(save_dir, save_dir), allow_pickle=True)
            est_drag_traj = np.load("./{}/deployed/restart_{}_interpolate_drag_trajectory.npy".format(save_dir, save_dir),
                                allow_pickle=True)
    else:
        drag_traj = np.load("./{}/deployed/{}_drag_trajectory.npy".format(save_dir, save_dir), allow_pickle=True)
        est_drag_traj = np.load("./{}/deployed/{}_interpolate_drag_trajectory.npy".format(save_dir, save_dir), allow_pickle=True)
        #drag_traj = np.load("./{}/ys930_to_ah93w145_drag_trajectory.npy".format(save_dir, save_dir), allow_pickle=True)
        #est_drag_traj = np.load("./{}/ys930_to_ah93w145_interpolate_drag_trajectory.npy".format(save_dir, save_dir), allow_pickle=True)
    d_idx = drag_traj.shape[1]//2
    l_idx = drag_traj.shape[1]-1

    # Plot drag draj
    ax.plot(drag_traj[:,0], np.abs(drag_traj[:,d_idx]), zorder=-1, label="Refinement Path", color='r', lw=1.5)
    if(SHOW_INTERPOLATION):
        ax.plot(est_drag_traj[:,0], np.abs(est_drag_traj[:,d_idx]), zorder=-1, label="Interpolated Refinement Path", color='g', lw=1.5)

    # Last value is special
    ax.scatter(drag_traj[:,0][FINAL_IDX], np.abs(drag_traj[:,d_idx][FINAL_IDX]), marker='*', s=200,
               color='goldenrod', edgecolor='k', lw=1.5, label="Refined Airfoil")

    # Original value
    ax.scatter(drag_traj[0][0],
               np.abs(drag_traj[0][d_idx]), marker='p', lw=3,
               s=200, label="Original Airfoil", edgecolor='k', color='magenta')

    # Add zoomed section
    if(ys930):
        axins = zoomed_inset_axes(ax, zoom=6, loc='upper right', bbox_to_anchor=(825,450))
    else:
        if(s1020):
            axins = zoomed_inset_axes(ax, zoom=6, loc='upper right', bbox_to_anchor=(720,480))
        else:
            axins = zoomed_inset_axes(ax, zoom=8, loc='upper right', bbox_to_anchor=(825,430))
    for axis in ['top', 'bottom', 'left', 'right']:
        axins.spines[axis].set_linewidth(2)

    axins.scatter(data["NUM_COORDS"][mask], data["DRAG"][mask].abs(), marker='s', edgecolor='k', lw=3,
               s=100, label="Computed Airfoils")
    axins.scatter(drag_traj[0][0],
               np.abs(drag_traj[0][d_idx]), marker='p', lw=3,
               s=200, label="Original Airfoil", edgecolor='k', color='magenta')
    axins.axhline(np.abs(big_data['DRAG'].values[0]), color='#888888', lw=2, linestyle='--')

    if(SHOW_INTERPOLATION):
        axins.axhline(1.001*np.abs(data['DRAG'].values[target_idx]), color='#aaaaaa', lw=2, linestyle='--')
        axins.axhline(0.999*np.abs(data['DRAG'].values[target_idx]), color='#aaaaaa', lw=2, linestyle='--')

    axins.axhline(np.abs(drag_traj[0][d_idx]), color='orchid', lw=2, linestyle='--', label="Original Value", zorder=-1)
    axins.plot(drag_traj[:,0], np.abs(drag_traj[:,d_idx]), zorder=-1, label="Refinement Path", color='r', lw=1.5)
    if(SHOW_INTERPOLATION):
        axins.plot(est_drag_traj[:,0], np.abs(est_drag_traj[:,d_idx]), zorder=-1, label="Refinement Path", color='g', lw=1.5)
    axins.scatter(drag_traj[:,0][FINAL_IDX], np.abs(drag_traj[:,d_idx][FINAL_IDX]), marker='*', s=200,
               color='goldenrod', edgecolor='k', lw=1.5, label="Refined Airfoil")

    if(ys930):
        x1, x2, y1, y2 = 810, 891, 0.1129, 0.11331
        axins.set_xticks([i for i in np.arange(x1, x2, 40)])
        axins.set_yticks([i for i in np.arange(y1, y2, 0.0004)])
        axins.set_xticklabels([i for i in np.arange(x1, x2, 40)], fontsize=12)
        axins.set_yticklabels(["{0:.4f}".format(i) for i in np.arange(y1, y2, 0.0004)], rotation=30, fontsize=12)
    else:
        if(s1020):
            #x1, x2, y1, y2 = 350, 610, 0.115, 0.1158
            x1, x2, y1, y2 = 650, 700, 0.1495, 0.1505
            axins.set_xticks([i for i in np.arange(x1, x2+1, 25)])
            axins.set_xticklabels([i for i in np.arange(x1, x2+1, 25)], fontsize=12)
            axins.set_yticks([i for i in np.arange(y1, y2, 0.0005)])
            axins.set_yticklabels(["{0:.4f}".format(i) for i in np.arange(y1, y2, 0.0005)], rotation=30, fontsize=12)
        else:
            x1, x2, y1, y2 = 740, 811, 0.13, 0.1305
            axins.set_xticks([i for i in np.arange(x1, x2, 30)])
            axins.set_xticklabels([i for i in np.arange(x1, x2, 30)], fontsize=12)
            axins.set_yticks([i for i in np.arange(y1, y2, 0.0005)])
            axins.set_yticklabels(["{0:.4f}".format(i) for i in np.arange(y1, y2, 0.0005)], rotation=30, fontsize=12)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    axins.yaxis.get_major_locator().set_params(nbins=7)
    axins.xaxis.get_major_locator().set_params(nbins=7)
    if(s1020):
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.2", lw=2)
    else:
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.2", lw=2)

    # Labels
    ax.set_xlabel("Number of Vertices", fontsize=20)
    ax.set_ylabel("Drag", fontsize=20)
    ax.set_title("{} Mesh Improvement (Drag)".format(save_dir.split("_")[0].upper()), fontsize=26, y=1.01)

    # Tick labels
    if(s1020):
        ax.set_xticks([i for i in np.arange(100, 1201, 100)])
        ax.set_xticklabels([str(i) for i in np.arange(100, 1201, 100)], fontsize=14)
    elif(ys930):
        ax.set_xticks([i for i in np.arange(100, 1201, 200)])
        ax.set_xticklabels([str(i) for i in np.arange(100, 1201, 200)], fontsize=14)
        ax.set_yticks([i for i in np.arange(0.113, 0.121, 0.001)])
        ax.set_yticklabels(["{0:.3f}".format(i) for i in np.arange(0.113, 0.121, 0.001)], fontsize=14, rotation=30)
    else:
        ax.set_xticks([i for i in np.arange(100, 1201, 200)])
        ax.set_xticklabels([str(i) for i in np.arange(100, 1201, 200)], fontsize=14)
        ax.set_yticks([i for i in np.arange(0.13, 0.1405, 0.002)])
        ax.set_yticklabels(["{0:.3f}".format(i) for i in np.arange(0.13, 0.1405, 0.002)], fontsize=14, rotation=30)

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
    #plt.savefig("./FINAL_RESULT_{}/deployed/{}_drag_improvement.png".format(save_dir, save_dir),
    #plt.savefig("./FINAL_{}/deployed/{}_drag_improvement.png".format(save_dir, save_dir),
    #plt.savefig("./{}/deployed/{}_drag_improvement.png".format(save_dir, save_dir),
    plt.savefig("./{}/deployed/{}_drag_improvement.svg".format(save_dir, save_dir),
                bbox_extra_artists=(lgd, axins), bbox_inches='tight', format='svg')
    #plt.savefig("./deployed_tl/{}_drag_improvement.png".format(save_dir, save_dir),

    print()
    print("INITIAL DRAG:\t{0:.8f}".format(drag_traj[0][d_idx]))
    print("FINAL DRAG:\t{0:.8f}".format(drag_traj[FINAL_IDX][d_idx]))
    print("DRAG ERROR:\t{0:.5f}%".format(100*np.abs(drag_traj[0][d_idx] - drag_traj[FINAL_IDX][d_idx])/np.abs(drag_traj[0][d_idx])))
    print()
    print("INITIAL LIFT:\t{0:.8f}".format(drag_traj[0][l_idx]))
    print("FINAL LIFT:\t{0:.8f}".format(drag_traj[FINAL_IDX][l_idx]))
    print("LIFT ERROR:\t{0:.5f}%".format(100*np.abs(drag_traj[0][l_idx] - drag_traj[FINAL_IDX][l_idx])/np.abs(drag_traj[0][l_idx])))
    print()
    print("INITIAL VERTICES:\t{0:.5f}".format(drag_traj[0][0]))
    print("FINAL VERTICES:\t\t{0:.5f}".format(drag_traj[FINAL_IDX][0]))
    print("VERTICES REMOVED:\t{0:.5f}".format(drag_traj[0][0] - drag_traj[FINAL_IDX][0]))
    print("VERTICES PERCENT:\t{0:.3f}%".format(100*(1-drag_traj[FINAL_IDX][0]/drag_traj[0][0])))


# New ys930 lift trajectory plots
if(True):
    if(ys930):
        data = pd.read_csv("./benchmark_results/smooth_ys930_1.0_0.001_smooth_benchmark.csv")
    else:
        if(s1020):
            #data = pd.read_csv("./benchmark_results/smooth_s1020_1.0_0.001_smooth_benchmark.csv")
            #data = pd.read_csv("./benchmark_results/smooth_nlf415_1.0_0.001_smooth_benchmark.csv")
            #data = pd.read_csv("./benchmark_results/smooth_rg1495_1.0_0.001_smooth_benchmark.csv")
            data = pd.read_csv("./benchmark_results/smooth_e654_1.0_0.001_smooth_benchmark.csv")
        else:
            data = pd.read_csv("./benchmark_results/smooth_ah93w145_1.0_0.001_smooth_benchmark.csv")

        median = data['DRAG'].median()
        std = data['DRAG'].std()
        data = data[np.abs(data['DRAG']) < 1.5*std + np.abs(median)]
    
    # Screen coarse meshes since they have inconsistent results
    big_data = data[data['NUM_COORDS'] > 1500]
    if(s1020):
        data = data[data['NUM_COORDS'] < 1200]
    else:
        data = data[data['NUM_COORDS'] < 1200]
    #data = data[data['NUM_COORDS'] < 3000]
    
    model = LinearRegression()
    model.fit(big_data['NUM_COORDS'].values[:,np.newaxis], big_data['DRAG'].values)
    
    xs = np.linspace(150, 2000)

    # Mask training airfoil
    if(ys930):
        target_idx = 3
    else:
        if(s1020):
            target_idx = 7
        else:
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
    if(RESTART):
        if(FINAL):
            drag_traj = np.load("./{}/deployed/confirmed/restart_{}_drag_trajectory.npy".format(save_dir, save_dir), allow_pickle=True)
            est_drag_traj = np.load("./{}/deployed/confirmed/restart_{}_interpolate_drag_trajectory.npy".format(save_dir, save_dir),
                                allow_pickle=True)
            #drag_traj = np.load("./FINAL_RESULT_{}/deployed/confirmed/restart_{}_drag_trajectory.npy".format(save_dir, save_dir), allow_pickle=True)
            #est_drag_traj = np.load("./FINAL_RESULT_{}/deployed/confirmed/restart_{}_interpolate_drag_trajectory.npy".format(save_dir, save_dir),
            #                    allow_pickle=True)
        else:
            drag_traj = np.load("./{}/deployed/restart_{}_drag_trajectory.npy".format(save_dir, save_dir), allow_pickle=True)
            est_drag_traj = np.load("./{}/deployed/restart_{}_interpolate_drag_trajectory.npy".format(save_dir, save_dir),
                                allow_pickle=True)
    #else:
    #    drag_traj = np.load("./{}/deployed/{}_drag_trajectory.npy".format(save_dir, save_dir), allow_pickle=True)
    else:
        drag_traj = np.load("./{}/deployed/{}_drag_trajectory.npy".format(save_dir, save_dir), allow_pickle=True)
        est_drag_traj = np.load("./{}/deployed/{}_interpolate_drag_trajectory.npy".format(save_dir, save_dir), allow_pickle=True)
    l_idx = drag_traj.shape[1]-1

    # Plot drag draj
    ax.plot(drag_traj[:,0], np.abs(drag_traj[:,l_idx]), zorder=-1, label="Refinement Path", color='r', lw=1.5)
    if(SHOW_INTERPOLATION):
        ax.plot(est_drag_traj[:,0], np.abs(est_drag_traj[:,l_idx]), zorder=-1, label="Interpolated Refinement Path", color='g', lw=1.5)

    # Last value is special
    ax.scatter(drag_traj[:,0][FINAL_IDX], np.abs(drag_traj[:,l_idx][FINAL_IDX]), marker='*', s=200,
               color='goldenrod', edgecolor='k', lw=1.5, label="Refined Airfoil")

    # Original value
    ax.scatter(drag_traj[0][0],
               np.abs(drag_traj[0][l_idx]), marker='p', lw=3,
               s=200, label="Original Airfoil", edgecolor='k', color='magenta')

    # Add zoomed section
    if(ys930):
        axins = zoomed_inset_axes(ax, zoom=3, loc='upper right', bbox_to_anchor=(750,265))
    else:
        if(s1020):
            axins = zoomed_inset_axes(ax, zoom=5, loc='upper right', bbox_to_anchor=(700,250))
        else:
            axins = zoomed_inset_axes(ax, zoom=5, loc='upper right', bbox_to_anchor=(750,475))
    for axis in ['top', 'bottom', 'left', 'right']:
        axins.spines[axis].set_linewidth(2)

    axins.scatter(data["NUM_COORDS"][mask], data["LIFT"][mask].abs(), marker='s', edgecolor='k', lw=3,
               s=100, label="Computed Airfoils")
    axins.scatter(drag_traj[0][0],
               np.abs(drag_traj[0][l_idx]), marker='p', lw=3,
               s=200, label="Original Airfoil", edgecolor='k', color='magenta')
    axins.axhline(np.abs(big_data['LIFT'].values[0]), color='#888888', lw=2, linestyle='--')

    if(SHOW_INTERPOLATION):
        axins.axhline(1.001*np.abs(data['LIFT'].values[target_idx]), color='#aaaaaa', lw=2, linestyle='--')
        axins.axhline(0.999*np.abs(data['LIFT'].values[target_idx]), color='#aaaaaa', lw=2, linestyle='--')
        axins.plot(est_drag_traj[:,0], np.abs(est_drag_traj[:,l_idx]), zorder=-1, label="Refinement Path", color='g', lw=1.5)

    axins.axhline(np.abs(drag_traj[0][l_idx]), color='orchid', lw=2, linestyle='--', label="Original Value", zorder=-1)
    axins.plot(drag_traj[:,0], np.abs(drag_traj[:,l_idx]), zorder=-1, label="Refinement Path", color='r', lw=1.5)
    axins.scatter(drag_traj[:,0][FINAL_IDX], np.abs(drag_traj[:,l_idx][FINAL_IDX]), marker='*', s=200,
               color='goldenrod', edgecolor='k', lw=1.5, label="Refined Airfoil")

    if(ys930):
        x1, x2, y1, y2 = 810, 891, 0.0457, 0.0473
        axins.set_xticks([i for i in np.arange(x1, x2, 40)])
        axins.set_yticks([i for i in np.arange(y1, y2, 0.0007)])
        axins.set_xticklabels(["{0}".format(i) for i in np.arange(x1, x2, 40)], fontsize=12)
        axins.set_yticklabels(["{0:.4f}".format(i) for i in np.arange(y1, y2, 0.0007)], rotation=30, fontsize=12)
    else:
        if(s1020):
            x1, x2, y1, y2 = 650, 700, 0.013, 0.0151
            axins.set_xticks([i for i in np.arange(x1, x2+1, 25)])
            axins.set_xticklabels([i for i in np.arange(x1, x2+1, 25)])
            axins.set_yticks([i for i in np.arange(y1, y2, 0.001)])
            axins.set_yticklabels(["{0:.3f}".format(i) for i in np.arange(y1, y2, 0.001)], rotation=30)
        else:
            x1, x2, y1, y2 = 740, 811, 0.049, 0.0525
            axins.set_xticks([i for i in np.arange(x1, x2, 30)])
            axins.set_xticklabels([i for i in np.arange(x1, x2, 30)], fontsize=12)
            axins.set_yticks([i for i in np.arange(y1, y2, 0.0015)])
            axins.set_yticklabels(["{0:.3f}".format(i) for i in np.arange(y1, y2, 0.0015)], rotation=30, fontsize=12)

    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    axins.yaxis.get_major_locator().set_params(nbins=7)
    axins.xaxis.get_major_locator().set_params(nbins=7)
    if(ys930):
        mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.2", lw=2)
    else:
        if(s1020):
            mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.2", lw=2)
        else:
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.2", lw=2)

    # Labels
    ax.set_xlabel("Number of Vertices", fontsize=20)
    ax.set_ylabel("Lift", fontsize=20)
    #ax.set_title("{} Lift Improvement After MeshDQN Training".format(save_dir.split("_")[0]), fontsize=26, y=1.01)
    ax.set_title("{} Mesh Improvement (Lift)".format(save_dir.split("_")[0].upper()), fontsize=26, y=1.01)

    # Tick labels
    if(s1020):
        ax.set_xticks([i for i in np.arange(100, 1201, 100)])
        ax.set_xticklabels([str(i) for i in np.arange(100, 1201, 100)], fontsize=14)
    elif(ys930):
        ax.set_xticks([i for i in np.arange(100, 1201, 200)])
        ax.set_xticklabels([str(i) for i in np.arange(100, 1201, 200)], fontsize=14)
        ax.set_yticks([i for i in np.arange(0.038, 0.054, 0.002)])
        ax.set_yticklabels(["{0:.3f}".format(i) for i in np.arange(0.038, 0.054, 0.002)], fontsize=14, rotation=30)
    else:
        ax.set_xticks([i for i in np.arange(100, 1201, 200)])
        ax.set_xticklabels([str(i) for i in np.arange(100, 1201, 200)], fontsize=14)
        ax.set_yticks([i for i in np.arange(0.03, 0.08, 0.01)])
        ax.set_yticklabels(["{0:.3f}".format(i) for i in np.arange(0.03, 0.08, 0.01)], fontsize=14, rotation=30)

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
    #plt.savefig("./FINAL_RESULT_{}/deployed/{}_lift_improvement.png".format(save_dir, save_dir),
    #plt.savefig("./FINAL_{}/deployed/{}_lift_improvement.png".format(save_dir, save_dir),
    plt.savefig("./{}/deployed/{}_lift_improvement.svg".format(save_dir, save_dir),
                bbox_extra_artists=(lgd, axins), bbox_inches='tight', format='svg')

    print()
    print("INITIAL DRAG:\t{0:.8f}".format(drag_traj[0][d_idx]))
    print("FINAL DRAG:\t{0:.8f}".format(drag_traj[FINAL_IDX][d_idx]))
    print("DRAG ERROR:\t{0:.5f}%".format(100*np.abs(drag_traj[0][d_idx] - drag_traj[FINAL_IDX][d_idx])/np.abs(drag_traj[0][d_idx])))
    print()
    print("INITIAL LIFT:\t{0:.8f}".format(drag_traj[0][l_idx]))
    print("FINAL LIFT:\t{0:.8f}".format(drag_traj[FINAL_IDX][l_idx]))
    print("LIFT ERROR:\t{0:.5f}%".format(100*np.abs(drag_traj[0][l_idx] - drag_traj[FINAL_IDX][l_idx])/np.abs(drag_traj[0][l_idx])))
    print()
    print("INITIAL VERTICES:\t{0:.5f}".format(drag_traj[0][0]))
    print("FINAL VERTICES:\t\t{0:.5f}".format(drag_traj[FINAL_IDX][0]))
    print("VERTICES REMOVED:\t{0:.5f}".format(drag_traj[0][0] - drag_traj[FINAL_IDX][0]))
    print("VERTICES PERCENT:\t{0:.3f}%".format(100*(1-drag_traj[FINAL_IDX][0]/drag_traj[0][0])))
    #plt.show()


# Check interpolation at each timestep
if(True):
    vertical = False
    if(vertical):
        fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(7,20))
    else:
        fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(20,7))

    # Load drag trajectory and plot
    if(RESTART):
        drag_traj = np.load("./{}/deployed/confirmed/restart_{}_drag_trajectory.npy".format(save_dir, save_dir), allow_pickle=True)
        est_drag_traj = np.load("./{}/deployed/confirmed/restart_{}_interpolate_drag_trajectory.npy".format(save_dir, save_dir),
                                allow_pickle=True)
    else:
        drag_traj = np.load("./{}/deployed/{}_drag_trajectory.npy".format(save_dir, save_dir), allow_pickle=True)
        est_drag_traj = np.load("./{}/deployed/{}_interpolate_drag_trajectory.npy".format(save_dir, save_dir), allow_pickle=True)
    d_idx = drag_traj.shape[1]//2
    l_idx = drag_traj.shape[1]-1

    # Plot drag draj
    for i in range(d_idx):

        if(vertical):
            c1, c2, c3, c4 = i, i, 0, 1
        else:
            c1, c2, c3, c4 = 0, 1, i, i
        ax[c1][c3].plot(drag_traj[:,0], np.abs(drag_traj[:,i+1]), zorder=-1, label="Refinement Path", color='r', lw=1.5)
        ax[c2][c4].plot(drag_traj[:,0], np.abs(drag_traj[:,i+d_idx+1]), zorder=-1, label="Refinement Path", color='r', lw=1.5)

        ax[c1][c3].axhline(np.abs(drag_traj[0][i+1]), color='#888888', lw=2, linestyle='--')
        ax[c1][c3].axhline(1.001*np.abs(drag_traj[0][i+1]), color='#aaaaaa', lw=2, linestyle='--')
        ax[c1][c3].axhline(0.999*np.abs(drag_traj[0][i+1]), color='#aaaaaa', lw=2, linestyle='--')

        ax[c2][c4].axhline(np.abs(drag_traj[0][i+d_idx+1]), color='#888888', lw=2, linestyle='--')
        ax[c2][c4].axhline(1.001*np.abs(drag_traj[0][i+d_idx+1]), color='#aaaaaa', lw=2, linestyle='--')
        ax[c2][c4].axhline(0.999*np.abs(drag_traj[0][i+d_idx+1]), color='#aaaaaa', lw=2, linestyle='--')
        #if(SHOW_INTERPOLATION):
        ax[c1][c3].plot(est_drag_traj[:,0], np.abs(est_drag_traj[:,i+1]), zorder=-1,
                        label="Interpolated Refinement Path", color='g', lw=1.5)
        ax[c2][c4].plot(est_drag_traj[:,0], np.abs(est_drag_traj[:,i+d_idx+1]), zorder=-1,
                        label="Interpolated Refinement Path", color='g', lw=1.5)

        if(vertical):
            ax[c1][c3].set_ylabel("Snapshot: {}".format(i+1), fontsize=14)
            #ax[c2][c4].set_ylabel("Vertices".format(i+1), fontsize=14)
        else:
            ax[c1][c3].set_title("Snapshot: {}".format(i+1), fontsize=20)
            ax[c2][c4].set_xlabel("Vertices".format(i+1), fontsize=20)

        ax[c1][c3].set_yticks([], [])
        ax[c2][c4].set_yticks([], [])
        if(vertical):# and (c2!=(d_idx-1))):
            if(c2 != (d_idx-1)):
                ax[c2][c4].set_xticks([], [])
                ax[c1][c3].set_xticks([], [])
        else:
            ax[c1][c3].set_xticks([], [])


    if(vertical):
        ax[0][0].set_title("Drag", fontsize=18)
        ax[0][1].set_title("Lift", fontsize=14)
        ax[-1][0].set_xlabel("Vertices", fontsize=14)
        ax[-1][1].set_xlabel("Vertices", fontsize=14)
    else:
        ax[0][0].set_ylabel("Drag", fontsize=20)
        ax[1][0].set_ylabel("Lift", fontsize=20)

    if(ys930 and vertical):
        ax[-1][0].set_xticks([i for i in np.arange(830, 871, 10)])
        ax[-1][0].set_xticklabels([i for i in np.arange(830, 871, 10)], fontsize=12)
        ax[-1][1].set_xticks([i for i in np.arange(830, 871, 10)])
        ax[-1][1].set_xticklabels([i for i in np.arange(830, 871, 10)], fontsize=12)
    elif(ys930):
        ax[-1][0].set_xticks([i for i in np.arange(830, 881, 25)])
        ax[-1][0].set_xticklabels([i for i in np.arange(830, 881, 25)], fontsize=16)
        ax[-1][1].set_xticks([i for i in np.arange(830, 881, 25)])
        ax[-1][1].set_xticklabels([i for i in np.arange(830, 881, 25)], fontsize=16)
        ax[-1][2].set_xticks([i for i in np.arange(830, 881, 25)])
        ax[-1][2].set_xticklabels([i for i in np.arange(830, 881, 25)], fontsize=16)
        ax[-1][3].set_xticks([i for i in np.arange(830, 881, 25)])
        ax[-1][3].set_xticklabels([i for i in np.arange(830, 881, 25)], fontsize=16)
        ax[-1][4].set_xticks([i for i in np.arange(830, 881, 25)])
        ax[-1][4].set_xticklabels([i for i in np.arange(830, 881, 25)], fontsize=16)


    #plt.axis('off')
    custom_lines = [
            Line2D([0], [0], color='red', lw=2),
            Line2D([0], [0], color='g', lw=2),
            Line2D([0], [0], color='#aaaaaa', lw=2, linestyle='--'),
            Line2D([0], [0], color='#888888', lw=2, linestyle='--'),
    ]
    labels = ["Calculated Path", "Interpolation Path", "Original Value", "Error Bounds"]

    if(vertical):
        lgd = fig.legend(custom_lines, labels, fontsize=14, ncol=2, bbox_to_anchor=(0.84, 0.08))
    else:
        lgd = fig.legend(custom_lines, labels, fontsize=14, ncol=4, bbox_to_anchor=(0.75, 0.02))

    t = fig.suptitle("{} Interpolation Comparison".format(save_dir.split("_")[0].upper()), fontsize=28, y=0.97)
    #ax[0][0].set_title("{}".format(save_dir.split("_")[0].upper()), fontsize=36, y=0)
    #plt.savefig("./FINAL_RESULT_{}/deployed/{}_comparison.png".format(save_dir, save_dir),
    #plt.savefig("./FINAL_{}/deployed/{}_comparison.png".format(save_dir, save_dir),
    #plt.savefig("./{}/deployed/{}_comparison.png".format(save_dir, save_dir),
    plt.savefig("./{}/deployed/{}_comparison.svg".format(save_dir, save_dir),
                bbox_extra_artists=(lgd,t,), bbox_inches='tight', format="svg")
    

if(False):
    vertical = True
    if(vertical):
        fig, ax = plt.subplots(nrows=10, ncols=2, figsize=(7,20))
    else:
        fig, ax = plt.subplots(nrows=2, ncols=10, figsize=(20,7))

    # Load drag trajectory and plot
    if(RESTART):
        drag_traj = np.load("./{}/deployed/confirmed/restart_{}_drag_trajectory.npy".format(save_dir, save_dir), allow_pickle=True)
        est_drag_traj = np.load("./{}/deployed/confirmed/restart_{}_interpolate_drag_trajectory.npy".format(save_dir, save_dir),
                                allow_pickle=True)
    else:
        drag_traj = np.load("./{}/deployed/{}_drag_trajectory.npy".format(save_dir, save_dir), allow_pickle=True)
        est_drag_traj = np.load("./{}/deployed/{}_interpolate_drag_trajectory.npy".format(save_dir, save_dir), allow_pickle=True)
    d_idx = drag_traj.shape[1]//2
    l_idx = drag_traj.shape[1]-1

    # Plot drag draj
    for i in range(d_idx):

        if(vertical):
            c1, c2, c3, c4 = i, i, 0, 1
        else:
            c1, c2, c3, c4 = 0, 1, i, i
        ax[c1][c3].plot(drag_traj[:,0], np.abs(drag_traj[:,i+1]), zorder=-1, label="Refinement Path", color='r', lw=1.5)
        ax[c2][c4].plot(drag_traj[:,0], np.abs(drag_traj[:,i+d_idx+1]), zorder=-1, label="Refinement Path", color='r', lw=1.5)

        ax[c1][c3].axhline(np.abs(drag_traj[0][i+1]), color='#888888', lw=2, linestyle='--')
        ax[c1][c3].axhline(1.001*np.abs(drag_traj[0][i+1]), color='#aaaaaa', lw=2, linestyle='--')
        ax[c1][c3].axhline(0.999*np.abs(drag_traj[0][i+1]), color='#aaaaaa', lw=2, linestyle='--')

        ax[c2][c4].axhline(np.abs(drag_traj[0][i+d_idx+1]), color='#888888', lw=2, linestyle='--')
        ax[c2][c4].axhline(1.001*np.abs(drag_traj[0][i+d_idx+1]), color='#aaaaaa', lw=2, linestyle='--')
        ax[c2][c4].axhline(0.999*np.abs(drag_traj[0][i+d_idx+1]), color='#aaaaaa', lw=2, linestyle='--')
        #if(SHOW_INTERPOLATION):
        ax[c1][c3].plot(est_drag_traj[:,0], np.abs(est_drag_traj[:,i+1]), zorder=-1,
                        label="Interpolated Refinement Path", color='g', lw=1.5)
        ax[c2][c4].plot(est_drag_traj[:,0], np.abs(est_drag_traj[:,i+d_idx+1]), zorder=-1,
                        label="Interpolated Refinement Path", color='g', lw=1.5)

        if(vertical):
            ax[c1][c3].set_ylabel("Snapshot: {}".format(i+1), fontsize=14)
            #ax[c2][c4].set_ylabel("Vertices".format(i+1), fontsize=14)
        else:
            ax[c1][c3].set_title("Snapshot: {}".format(i+1), fontsize=14)
            ax[c2][c4].set_xlabel("Vertices".format(i+1), fontsize=14)

        ax[c1][c3].set_yticks([], [])
        ax[c2][c4].set_yticks([], [])
        if(vertical):# and (c2!=(d_idx-1))):
            if(c2 != (d_idx-1)):
                ax[c2][c4].set_xticks([], [])
                ax[c1][c3].set_xticks([], [])
        else:
            ax[c1][c3].set_xticks([], [])


    if(vertical):
        ax[0][0].set_title("Drag", fontsize=14)
        ax[0][1].set_title("Lift", fontsize=14)
        ax[-1][0].set_xlabel("Vertices", fontsize=14)
        ax[-1][1].set_xlabel("Vertices", fontsize=14)
    else:
        ax[0][0].set_ylabel("Drag", fontsize=14)
        ax[1][0].set_ylabel("Lift", fontsize=14)

    #plt.axis('off')
    custom_lines = [
            Line2D([0], [0], color='red', lw=2),
            Line2D([0], [0], color='g', lw=2),
            Line2D([0], [0], color='#aaaaaa', lw=2, linestyle='--'),
            Line2D([0], [0], color='#888888', lw=2, linestyle='--'),
    ]
    labels = ["Calculated Path", "Interpolation Path", "Original Value", "Error Bounds"]

    if(vertical):
        lgd = fig.legend(custom_lines, labels, fontsize=14, ncol=2, bbox_to_anchor=(0.84, 0.08))
    else:
        lgd = fig.legend(custom_lines, labels, fontsize=14, ncol=4, bbox_to_anchor=(0.75, 0.04))

    t = fig.suptitle("{} Interpolation Comparison".format(save_dir.split("_")[0].upper()), fontsize=22, y=0.96)
    #ax[0][0].set_title("{}".format(save_dir.split("_")[0].upper()), fontsize=36, y=0)
    #plt.savefig("./FINAL_RESULT_{}/deployed/{}_comparison.png".format(save_dir, save_dir),
    #plt.savefig("./FINAL_{}/deployed/{}_comparison.png".format(save_dir, save_dir),
    plt.savefig("./{}/deployed/{}_comparison.png".format(save_dir, save_dir),
                bbox_extra_artists=(lgd,t,), bbox_inches='tight')

    print()
    print("INITIAL DRAG:\t{0:.8f}".format(drag_traj[0][d_idx]))
    print("FINAL DRAG:\t{0:.8f}".format(drag_traj[FINAL_IDX][d_idx]))
    print("DRAG ERROR:\t{0:.5f}%".format(100*np.abs(drag_traj[0][d_idx] - drag_traj[FINAL_IDX][d_idx])/np.abs(drag_traj[0][d_idx])))
    print()
    print("INITIAL LIFT:\t{0:.8f}".format(drag_traj[0][l_idx]))
    print("FINAL LIFT:\t{0:.8f}".format(drag_traj[FINAL_IDX][l_idx]))
    print("LIFT ERROR:\t{0:.5f}%".format(100*np.abs(drag_traj[0][l_idx] - drag_traj[FINAL_IDX][l_idx])/np.abs(drag_traj[0][l_idx])))
    print()
    print("INITIAL VERTICES:\t{0:.5f}".format(drag_traj[0][0]))
    print("FINAL VERTICES:\t\t{0:.5f}".format(drag_traj[FINAL_IDX][0]))
    print("VERTICES REMOVED:\t{0:.5f}".format(drag_traj[0][0] - drag_traj[FINAL_IDX][0]))
    print("VERTICES PERCENT:\t{0:.3f}%".format(100*(1-drag_traj[FINAL_IDX][0]/drag_traj[0][0])))
    #plt.show()


# Drag reward one sided
if(False):
    fig, ax = plt.subplots(figsize=(10,8))
    xs = np.linspace(100., 100.2, 1001)

    #f = lambda x: 2*np.exp(-50*np.abs(100 - x)/np.abs(100.)) - 1
    #ax.plot(xs, f(xs), lw=2, color='sienna', label=r"$K=50$")
            #label=r"$2\exp\left(\frac{-50\left|d_{gt} - d_{n} \right|}{\left|d_{gt} \right|}\right) - 1$")

    f = lambda x: 2*np.exp(-693.15*np.abs(100 - x)/np.abs(100.)) - 1
    ax.plot(xs, f(xs), lw=2, color='k', label=r"$K=693.15$")
            #label=r"$2\exp\left(\frac{-691\left|d_{gt} - d_{n} \right|}{\left|d_{gt} \right|}\right) - 1$")
            #label=r"$2\exp\left(\frac{-1386\left|d_{gt} - d_{n} \right|}{\left|d_{gt} \right|}\right) - 1$")

    f = lambda x: 2*np.exp(-1386.29*np.abs(100 - x)/np.abs(100.)) - 1
    ax.plot(xs, f(xs), lw=2, color='g', label=r"$K=1386.29$")

    f = lambda x: 2*np.exp(-2772.59*np.abs(100 - x)/np.abs(100.)) - 1
    ax.plot(xs, f(xs), lw=2, color='b', label=r"$K=2772.59$")
    #f = lambda x: 2*np.exp(-5000*np.abs(100 - x)/np.abs(100.)) - 1
    #ax.plot(xs, f(xs), lw=2, color='g', label=r"$K=5000$")
            #label=r"$2\exp\left(\frac{-5000\left|d_{gt} - d_{n} \right|}{\left|d_{gt} \right|}\right) - 1$")

    #f = lambda x: 2*np.exp(-50*np.abs(100 - x)/np.abs(100.)) - 1
    #ax.plot(xs, f(xs), lw=2, color='k')

    ax.axvline(100.025, color='lightcoral', lw=2, label="Quarter Accuracy Threshold", linestyle=':')
    ax.axvline(100.05, color='tomato', lw=2, label="Half Accuracy Threshold", linestyle='-.')
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

