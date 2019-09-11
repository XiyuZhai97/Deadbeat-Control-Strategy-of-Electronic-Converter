"""
Plug a RL method to the framework, this method can be discrete or continuous.
This script is based on a continuous action RL. If you want to change to discrete RL like DQN,
please change the env.py and rl.py correspondingly.
"""
from env import InverterEnv
from rl import DDPG
import Signal_Generator
import numpy as np
import matplotlib.pyplot as plt

MAX_EPISODES = 5000
MAX_EP_STEPS = 1000
ON_TRAIN = False # True
ON_TRAIN = True

# set env
env = InverterEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)


def train():
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()
        env.harmonic_n = 2
        ep_r=0;
        for j in range(MAX_EP_STEPS+1):
            a = rl.choose_action(s)
            s_, r = env.step(a)
            # plt.savefig('image\Outputfig' + str(j) + '.jpg')
            rl.store_transition(s, a, r, s_)
            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()
            s = s_
            if j == MAX_EP_STEPS:
                print('Ep: %i | r: %.1f| steps: %i| kp1: %.4f| kr1: %.4f|  kp3: %.4f| kr3: %.4f|' % (i, ep_r, j, a[0], a[1],a[2],a[3]))
                break
        # print("done 1 episode")
        np.savetxt('out_sum_r_list'+str(i)+'.txt', env.out_sum_r_list)
        # plt.savefig ('image\Outputfig' + str(i) + '.jpg')
        if i % 10 == 0:
            print("输出模型")
            rl.save(i)

def eval(): # 测试
    rl.restore()
    # while True:
    s = env.reset()
    env.rL = 4.44425
    env.L = 1950e-6
    env.C = 15e-6
    env.X = 12.5
    # env.harmonic_n = 2
    a = rl.choose_action(s)
    s, r = env.step(a)
    plt.show()
        # plt.savefig('image\Outputfig' + str(j) + '.jpg')


if ON_TRAIN:
    train()
else:
    eval()



