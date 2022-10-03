import numpy as np
from matplotlib import pyplot as plt

def _movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 80000
f = lambda steps_done: EPS_END + (EPS_START - EPS_END) * np.exp(-steps_done/EPS_DECAY)

# Good success so far
#PREFIX = 'ys930_1386_long_interp'
#PREFIX = 'ys930_ray'
#PREFIX = 'ys930_ray_remote_dqn'
#PREFIX = 'ys930_ray_remote_dqn_tuning'
#PREFIX = 'ys930_ray_single_remote_dqn_tuning'
PREFIX = 'ys930_ray_parallel_training'

#TODO Intrpolate original mesh to current mesh each time instead of incremental interpolation?
ep_reward = np.load("./{}/{}_reward.npy".format(PREFIX, PREFIX))
for i in range(10)[::-1]:
    print(ep_reward[-i])

fig, ax = plt.subplots()
#ax.plot(ep_reward, label="Reward")
#ax.axhline(0, color='#888888')
try:
    ax.plot(list(range(len(ep_reward)))[199:], _movingaverage(ep_reward, 200), label="200 Episode Moving Average")
except ValueError:
    ax.plot(ep_reward, label="No Window")
    pass
#try:
#    ax.plot(list(range(len(ep_reward)))[999:], _movingaverage(ep_reward, 1000), label="1000 Episode Window")
#    pass
#except ValueError:
#    pass
#try:
#    ax.plot(list(range(len(ep_reward)))[4999:], _movingaverage(ep_reward, 5000), label="5000 Episode Window")
#    pass
#except ValueError:
#    pass
ax.set_xlabel("Episode", fontsize=12)
ax.set_ylabel("Reward", fontsize=12)
ax.set_title("Double DQN Moving Average Reward", fontsize=14)
ax.legend(loc='best')
#plt.savefig("./100k_new_reward_small_lr_batch_32_drag_and_time_ma_ag11_reward.png")
print("200 STEP WINDOW FINAL REWARD: {} AT STEP: {}".format(_movingaverage(ep_reward, 200)[-1], len(ep_reward)))
#print("200 STEP WINDOW FINAL REWARD: {} AT STEP: {}".format(_movingaverage(ep_reward, 200)[7100-200], 7100))
plt.show()

fig, ax = plt.subplots()
ax.plot(ep_reward, label="Reward")
try:
    ax.plot(list(range(len(ep_reward)))[199:], _movingaverage(ep_reward, 200), label="200 Step Window")
except ValueError:
    pass
#ax.plot(list(range(len(ep_reward)))[999:], _movingaverage(ep_reward, 1000), label="1000 Step Window")
ax.set_xlabel("Episode", fontsize=12)
ax.set_ylabel("Reward", fontsize=12)
ax.set_title("Double DQN Moving Average Reward", fontsize=14)
ax.legend(loc='best')
plt.savefig("./{}/{}_reward.png".format(PREFIX, PREFIX))

#fig, ax = plt.subplots()
#ax.plot(np.linspace(0, 200000, 500), f(np.linspace(0, 200000, 500)))
#plt.show()
