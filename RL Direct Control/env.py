import numpy as np
import pyglet
import PR_CON
import Plant
import Signal_Generator
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
class InverterEnv(object):
    viewer = None
    action_bound = [-1000, 1000]
    goal = 220
    state_dim = 7
    action_dim = 1
    def __init__(self, rl=0.1, l=200e-6, c=5e-6, t_sample=1/200000):
        self.firstflag = 0
        self.rL = rl
        self.L = l
        self.C = c
        self.t_sample_feature=t_sample
        self.plantb0 = pow(t_sample, 2) / (4 * l * c + 2 * c * rl * t_sample + pow(t_sample, 2))
        self.plantb1 = 2 * pow(t_sample, 2) / (4 * l * c + 2 * c * rl * t_sample + pow(t_sample, 2))
        self.plantb2 = pow(t_sample, 2) / (4 * l * c + 2 * c * rl * t_sample + pow(t_sample, 2))
        self.planta1 = (2 * pow(t_sample, 2) - 8 * l * c) / (4 * l * c + 2 * c * rl * t_sample + pow(t_sample, 2))
        self.planta2 = (pow(t_sample, 2) + 4 * l * c - 2 * c * rl * t_sample) / (
                4 * l * c + 2 * c * rl * t_sample + pow(t_sample, 2))
        self.reset()
        self.input = 0.0
        self.last1_input = 0.0
        self.last2_input = 0.0
        self.output = 0.0
        self.last1_output = 0.0
        self.last2_output = 0.0
        self.final_output_list = []
        self.goal_list = []
        self.action_list = []
        self.lastr = 0.
    def step(self, action):
        action = np.clip(action, *self.action_bound) # 将数组中的元素限制在a_min, a_max之间
        # print('action=%.1f' %( action ) )
        self.input = action
        self.action_list.append(action/10)
        self.output = (self.plantb0*self.input + self.plantb1*self.last1_input + self.plantb2*self.last2_input - self.planta1*self.last1_output - self.planta2*self.last2_output)
        self.last2_output = self.last1_output
        self.last1_output = self.output
        self.last2_input = self.last1_input
        self.last1_input = self.input
        self.final_output_list.append(self.output[0])
        self.goal_list.append(self.goal)
        # done and reward
        real_r = -np.abs((self.output - self.goal)[0]/10)
        r = real_r
        if (np.abs(self.output - self.goal)[0] < 5):
            r = 100.* (1- (np.abs(self.output - self.goal)[0])/5)
        if(np.abs(r) < np.abs(self.lastr)*0.9):
            r = r + 20
        # print(r)
        # state
        # s = np.concatenate((np.array([self.rL]), np.array([self.L]),np.array([self.C]), np.array([self.t_sample_feature]), np.array(self.output),np.array([self.goal])))
        if self.firstflag == 0:
            s = np.concatenate((np.array([self.last2_input]),np.array(self.last1_input), np.array([self.last2_output]),
                            np.array(self.last1_output), np.array(self.input), np.array(self.output),np.array([self.goal])))
        elif self.firstflag == 1:
            s = np.concatenate((np.array(self.last2_input), np.array(self.last1_input), np.array(self.last2_output),
                                np.array(self.last1_output), np.array(self.input), np.array(self.output),
                                np.array([self.goal])))
        self.lastr = real_r
        # print('r',r,'real r',real_r)
        self.firstflag = 1
        return s, r


    def reset(self):
        self.input = 0.0
        self.last1_input = 0.0
        self.last2_input = 0.0
        self.output = 0.0
        self.last1_output = 0.0
        self.last2_output = 0.0
        self.final_output_list = []
        self.goal_list = []
        self.action_list = []
        # s = np.concatenate((np.array([self.rL]), np.array([self.L]),np.array([self.C]), np.array([self.t_sample_feature]), np.array([self.output]),np.array([self.goal])))
        # s = np.concatenate((np.array([self.rL]), np.array([self.L]),np.array([self.C]), np.array([self.t_sample_feature]), np.array(self.output),np.array([self.goal])))
        s = np.concatenate((np.array([self.last2_input]),np.array([self.last1_input]), np.array([self.last2_output]),np.array([self.last1_output]), np.array([self.input]), np.array([self.output]),np.array([self.goal])))
        self.firstflag = 0
        return s


    def render(self):
        plt.cla()

        t_x=np.arange(0,self.t_sample_feature * (len(self.final_output_list)), self.t_sample_feature)

        plt.plot(t_x,self.final_output_list, label='Uo')
        plt.plot(t_x,self.goal_list, label='Uref')
        # plt.plot(self.action_list)
        plt.legend(loc='upper right')
        plt.xlabel('time (s)')
        plt.ylabel('OUTPUT')
        plt.grid(True)
        # plt.xlim([0,210])
        # plt.ylim(self.action_bound)
        # plt.draw()
        plt.pause(0.0001)
        # plt.show()

    def sample_action(self):
        return 120*np.random.random()-60    # two radians
# class Viewer(pyglet.window.Window):
#     def __init__(self):
#         # vsync=False to not use the monitor FPS, we can speed up training
#         super(Viewer, self).__init__(width=400, height=400, resizable=False, caption='Inverter output', vsync=False)
#         pyglet.gl.glClearColor(1, 1, 1, 1)
#
#         self.batch = pyglet.graphics.Batch()    # display whole batch at once
#         self.point1 = self.batch.add(
#             4, pyglet.gl.GL_QUADS, None,    # 4 corners
#             ('v2f', [50, 50,                # location
#                      50, 100,
#                      100, 100,
#                      100, 50]),
#             ('c3B', (86, 109, 249) * 4))    # color
#         self.point2 = self.batch.add(
#             4, pyglet.gl.GL_QUADS, None,    # 4 corners
#             ('v2f', [50, 200,                # location
#                      50, 250,
#                      100, 250,
#                      100, 200]),
#             ('c3B', (86, 109, 249) * 4))    # color
#         self.point3 = self.batch.add(
#             4, pyglet.gl.GL_QUADS, None,    # 4 corners
#             ('v2f', [200, 200,                # location
#                      200, 250,
#                      250, 250,
#                      250, 200]),
#             ('c3B', (86, 109, 249) * 4))    # color
#         self.point4 = self.batch.add(
#             4, pyglet.gl.GL_QUADS, None,    # 4 corners
#             ('v2f', [200, 50,                # location
#                      200, 100,
#                      250, 100,
#                      250, 50]),
#             ('c3B', (86, 109, 249) * 4))    # color
#     def render(self):
#         self._update_inverter()
#         self.switch_to()
#         self.dispatch_events()
#         self.dispatch_event('on_draw')
#         self.flip()
#
#     def on_draw(self):
#         self.clear()
#         self.batch.draw()

if __name__ == '__main__':
    env = InverterEnv()
    # 参考信号 谐波生成
    n = 1
    t = 1
    fs = 1000
    setpoint_list = Signal_Generator.mix_signal(n,t,fs)# 参考/给定输入
    n = np.arange(0, t, 1 / fs)
    L = len(n)  # L仿真步数
    for i in range(0, L):
        env.goal=setpoint_list[i]
        env.step(60)

    env.render()