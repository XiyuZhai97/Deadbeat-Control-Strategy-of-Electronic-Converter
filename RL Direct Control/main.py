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
# MAX_EP_STEPS = 200
ON_TRAIN = False
ON_TRAIN = True

# set env
env = InverterEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)
A1 = 50
A3 = 70
A5 = 50
A25 = 30
A27 = 30
A47 = 20
A49 = 20

def train():
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()
        # print('rest后的状态s',s)
        ep_r = 0.
        n = round(10*np.random.random())+1
        t = 0.03
        fs = 200000

        setpoint_list = Signal_Generator.mix_signal(A1,A3,A5,A25,A27,A47,A49,t,fs)  # 参考/给定输入
        n = np.arange(0, t, 1 / fs)
        MAX_EP_STEPS = len(n)  # L仿真步数
        print(MAX_EP_STEPS)
        for j in range(MAX_EP_STEPS+1):
            env.goal = setpoint_list[j]
            a = rl.choose_action(s)
            s_, r = env.step(a)
            rl.store_transition(s, a, r, s_)
            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()
            s = s_
            if j == MAX_EP_STEPS-1:
                print('Ep: %i | ep_r: %.1f | steps: %i' % (i, ep_r, j))
                break
        # plt.show
        # plt.ion()
        env.render()
        # if i % 200 == 0:
        plt.savefig('image_6_4\Outputfig' + str(i) + '.jpg')
        if i % 200 == 0:
            rl.save(i)


def eval(): # 测试
    rl.restore()
    # env.render()
    n = 10
    t = 0.12
    fs = 20000
    setpoint_list = Signal_Generator.mix_signal(A1, A3, A5, A25, A27, A47, A49, t, fs)  # 参考/给定输入
    n = np.arange(0, t, 1 / fs)
    MAX_EP_STEPS = len(n)  # L仿真步数
    # while True:
    s = env.reset()
    sum_error=0
    for _ in range(MAX_EP_STEPS):
        env.goal = setpoint_list[_]
        a = rl.choose_action(s)
        s, r = env.step(a)
        sum_error = np.abs(env.output-env.goal)+sum_error
    print("average_error",sum_error/MAX_EP_STEPS)
    # plt.show
    # plt.ion()
    env.render()
    # plt.show()


if ON_TRAIN:
    train()
else:
    eval()



