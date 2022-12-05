import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

RESTART = False
VERBOSE = 1

def plot(PREFIX):
    if(RESTART):
        actions = np.load("./{}/{}_RESTART_actions.npy".format(PREFIX, PREFIX), allow_pickle=True)
        rewards = np.load("./{}/{}_RESTART_rewards.npy".format(PREFIX, PREFIX), allow_pickle=True)
        losses = np.load("./{}/{}_RESTART_losses.npy".format(PREFIX, PREFIX), allow_pickle=True)
        losses = losses[losses != np.array(None)]
        epss = np.load("./{}/{}_RESTART_eps.npy".format(PREFIX, PREFIX), allow_pickle=True)
    else:
        while(True):
            try:
                actions = np.load("./{}/{}_actions.npy".format(PREFIX, PREFIX), allow_pickle=True)
                break
            except OSError:
                pass
        rewards = np.load("./{}/{}_rewards.npy".format(PREFIX, PREFIX), allow_pickle=True)
        while(True):
            try:
                rewards = np.load("./{}/{}_rewards.npy".format(PREFIX, PREFIX), allow_pickle=True)
                break
            except OSError:
                pass
        while(True):
            try:
                losses = np.load("./{}/{}_losses.npy".format(PREFIX, PREFIX), allow_pickle=True)
                break
            except OSError:
                pass
        while(True):
            try:
                losses = losses[losses != np.array(None)]
                break
            except OSError:
                pass
        while(True):
            try:
                epss = np.load("./{}/{}_eps.npy".format(PREFIX, PREFIX), allow_pickle=True)
                break
            except OSError:
                pass

    if(VERBOSE >= 1):
        print("CURRENT EPSILON:\t\t{0:.5f} AT STEP: {1}".format(epss[-1], len(losses)))
        print("OPTIMIZER STEPS: {}".format(len(losses)))
        print(len(actions), np.hstack(actions).shape)
    
    #actions = np.load("./100k_new_reward_small_lr_batch_32_drag_and_time_ag11_actions.npy", allow_pickle=True)
    #rewards = np.load("./100k_new_reward_small_lr_batch_32_drag_and_time_ag11_rewards.npy", allow_pickle=True)
    
    
    ep_rews = np.empty(len(rewards))
    longest_ep, longest_idx = 0, -1
    num_max = 0
    for idx, r in enumerate(rewards):
        ep_rews[idx] = np.sum(r)
        if(len(r) > longest_ep):
            longest_ep = len(r)
            longest_idx = idx
        if(len(r) == 44):
        #if(len(r) == 30):
            num_max += 1
    
    worst = np.argmin(ep_rews)
    best = np.argmax(ep_rews)
    if(VERBOSE >= 1):
        print("WORST: {0}\t{1:.5f}\t{2}".format(worst, ep_rews[worst], len(rewards[worst])))
    if(VERBOSE >= 2):
        print(actions[worst])
        print(rewards[worst])
    
    if(VERBOSE >= 1):
        print("\nBEST: {0}\t{1:.5f}\t{2}".format(best, ep_rews[best], len(rewards[best])))
    if(VERBOSE >= 2):
        print(actions[best])
        print(rewards[best])
        print()

    if(VERBOSE >= 3):
        for i in range(1, 4):
            print(actions[-i])
            print(rewards[-i])
            print()

    if(VERBOSE >= 1):
        print("\nLONGEST EPISODE IS {} WITH {} STEPS".format(longest_idx, longest_ep))
        print("NUMBER OF MAX LENGTH EPISODES: {}\n".format(num_max))
    
    num_steps = len(ep_rews)
    #for i in range(num_steps - 10, num_steps):
    for i in range(1, 11)[::-1]:
        num_removals = sum(np.array(actions[-i]) != 110)
        if(VERBOSE >= 2):
            print(num_steps - i, num_removals, len(rewards[-i]), ep_rews[-i])
    
    #print(actions.shape)
    #print("CURRENT EPSILON:\t\t{0:.5f}".format(epss[-1]))
    #print("CURRENT EPSILON:\t\t{0:.5f} AT STEP: {1}".format(epss[-1], len(losses)))
    
    def _movingaverage(values, window):
        weights = np.repeat(1.0, window)/window
        sma = np.convolve(values, weights, 'valid')
        return sma
    
    #fig, ax = plt.subplots(figsize=(8,6))
    fig, ax = plt.subplots()
    #ax.plot(losses, label="Loss")
    print("OPTIMIZER STEPS: {}".format(len(losses)))
    try:
        ax.plot(range(len(losses))[199:], _movingaverage(losses, 200), label="200 Step Window")
    except ValueError:
        ax.plot(losses)
        #pass
    try:
        ax.plot(range(len(losses))[499:], _movingaverage(losses, 500), label="500 Step Window")
    except ValueError:
        #ax.plot(losses)
        pass
    try:
        ax.plot(range(len(losses))[999:], _movingaverage(losses, 1000), label="1000 Step Moving Average")
    except ValueError:
        pass
    try:
        ax.plot(range(len(losses))[4999:], _movingaverage(losses, 5000), label="5000 Step Window")
    except ValueError:
        pass
    try:
        ax.plot(range(len(losses))[49999:], _movingaverage(losses, 50000), label="50000 Step Window")
    except ValueError:
        pass
    #try:
    #    ax.plot(range(len(losses))[499999:], _movingaverage(losses, 500000), label="500000 Step Window")
    #except ValueError:
    #    pass
    _, counts = np.unique(np.hstack(actions), return_counts=True)
    percents = counts / sum(counts)
    #print("200 STEP WINDOW LOSS: {} AT STEP: {}".format(_movingaverage(losses, 200)[-1], len(losses)))
    print("5000 STEP WINDOW LOSS: {} AT STEP: {}".format(_movingaverage(losses, 5000)[-1], len(losses)))
    print("DO NOTHING PERCENT: {}, MEDIAN: {}".format(100*percents[-1], np.median(100*percents)))
    print("DO NOTHING RATIO: {}".format(100*percents[-1]/np.median(100*percents)))
    print("DO NOTHING NUMBER: {}".format(counts[-1]))
    
    ax.set_title("Double DQN Loss Over Time", fontsize=14)
    ax.set_xlabel("Optimizer Steps", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.legend(loc='best')
    plt.savefig("./{}/{}_losses.png".format(PREFIX, PREFIX))
    #plt.show()
    
    #print(np.hstack(actions).shape)
    
    fig, ax = plt.subplots()
    #ax.hist(np.hstack(actions), bins=201, density=True)
    #for i in range(10):
    #    print(actions[i])
    
    print("NUMBER OF STEPS: {} WITH {} TOTAL ACTIONS".format(len(actions), np.hstack(actions).shape[0]))
    if('nlf' in PREFIX):
        ax.hist(np.hstack(actions), bins=251, density=True)
    if('rg' in PREFIX):
        ax.hist(np.hstack(actions), bins=111, density=True)
    if('cylinder' in PREFIX):
        ax.hist(np.hstack(actions), bins=201, density=True)
    else:
        ax.hist(np.hstack(actions), bins=181, density=True)
    ax.set_xlabel("Action", fontsize=12)
    ax.set_ylabel("Fraction of Selections", fontsize=12)
    ax.set_title("Double DQN Action Selection", fontsize=14)
    plt.savefig("./{}/{}_action_selection.png".format(PREFIX, PREFIX))
    #plt.show()
        
ps = [
    'ys930_results',
    #'ah93w145_results'
]
for p in ps:
    plot(p)
    print()
    print()

