import numpy as np
import PR_CON
import Plant
import Signal_Generator
import math
import matplotlib.pyplot as plt
import pysnooper
from scipy.fftpack import fft

class InverterEnv(object):
    viewer = None
    action_bound = [0.1,120,0.1,120,0.1,120,0.1,120,0.1,120]
    goal = 220
    state_dim = 5
    action_dim = 10

    def __init__(self, rl=0.1, l=200e-6, c=5e-6,x=10, t_sample=0.0001):
        self.a_fft1 = 0
        self.a_fft3 = 0
        self.a_fft25 = 0
        self.a_fft47 = 0
        self.a_fft50 = 0
        self.rL = rl
        self.L = l
        self.C = c
        self.X = x
        self.t_sample_feature = t_sample
        self.reset()
        self.A1 = 220
        self.A3 = 150
        self.A25 = 100
        self.A47 = 220
        self.A50 = 150
        self.input = 0.0
        self.output = 0.0
        self.final_output_list = []
        self.goal_list = []
        self.action_view = np.array([0., 0., 0., 0., 0.])
        self.my_plant = Plant.PlantLC(rl=rl, l=l, c=c, t_sample=t_sample)
        self.my_pr = PR_CON.PRControl(kp=0.1, kr=100, wc=5, wr=100 * math.pi, t_sample= self.t_sample_feature)
        self.my_pr3 = PR_CON.PRControl(kp=0.1, kr=70, wc=5, wr=300 * math.pi, t_sample=self.t_sample_feature)
        self.my_pr25 = PR_CON.PRControl(kp=0.1, kr=20, wc=5, wr=2500 * math.pi, t_sample=self.t_sample_feature)
        self.my_pr47 = PR_CON.PRControl(kp=0.1, kr=20, wc=5, wr=4700 * math.pi, t_sample=self.t_sample_feature)
        self.my_pr50 = PR_CON.PRControl(kp=0.1, kr=20, wc=5, wr=5000 * math.pi, t_sample=self.t_sample_feature)
        self.last_step_j = 0
        self.last_sum_r = 0

    def reset(self):
        self.a_fft1 = 0
        self.a_fft3 = 0
        self.a_fft25 = 0
        self.a_fft47 = 0
        self.a_fft50 = 0
        self.rL = np.random.random()
        self.L = np.random.random()*400e-6
        self.C = np.random.random()*100e-6
        self.X = np.random.random()*50
        # self.A1 = 100 + 10 * round(20*np.random.random())
        # self.A3 = 100 + 10 * round(20*np.random.random())
        # self.A25 = 100 + 10 * round(20*np.random.random())
        # self.A47 = 100 + 10 * round(20*np.random.random())
        # self.A50 = 100 + 10 * round(20*np.random.random())

        # self.rL = 0.1
        # self.L = 200e-6
        # self.C = 5e-6
        self.A1 = 220
        self.A3 = 150
        self.A25 = 100
        self.A47 = 220
        self.A50 = 150
        self.last_step_j = 0
        self.last_sum_r = 0
        # self.t_sample_feature=0.001*np.random.random()
        self.input = 0.0
        self.output = 0.0
        self.final_output_list = []
        self.goal_list = []
        self.action_view = np.array([0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0.])
        self.my_pr = PR_CON.PRControl(kp=0.1, kr=150, wc=5, wr=100 * math.pi, t_sample=self.t_sample_feature)
        self.my_pr3 = PR_CON.PRControl(kp=0.1, kr=20, wc=5, wr=300 * math.pi, t_sample=self.t_sample_feature)
        self.my_pr25 = PR_CON.PRControl(kp=0.1, kr=20, wc=5, wr=2500 * math.pi, t_sample=self.t_sample_feature)
        self.my_pr47 = PR_CON.PRControl(kp=0.1, kr=20, wc=5, wr=4700 * math.pi, t_sample=self.t_sample_feature)
        self.my_pr50 = PR_CON.PRControl(kp=0.1, kr=20, wc=5, wr=5000 * math.pi, t_sample=self.t_sample_feature)
        self.my_plant = Plant.PlantLC(rl=0.1, l=200e-6, c=5e-6,x= self.X, t_sample= self.t_sample_feature)
        s = np.concatenate((np.array([self.a_fft1]), np.array([self.a_fft3]), np.array([self.a_fft25]),
                            np.array([self.a_fft47]), np.array([self.a_fft50])))
        # s = np.concatenate((np.array([self.A1]), np.array([self.A3]), np.array([self.A25]),
        #                     np.array([self.A47]), np.array([self.A50]), np.array([self.L * 1e6]),
        #                     np.array([self.C * 1e6]),  np.array([self.X]) ,np.array([self.rL])))
        return s

    def step(self, action):
        # self.rL = np.random.random()
        # self.L = np.random.random()*400e-6
        # self.C = np.random.random()*100e-6
        # self.X = np.random.random()*50
        self.A1 = 100 + 10 * round(20*np.random.random())
        self.A3 = 100 + 10 * round(20*np.random.random())
        self.A25 = 100 + 10 * round(20*np.random.random())
        self.A47 = 100 + 10 * round(20*np.random.random())
        self.A50 = 100 + 10 * round(20*np.random.random())
        self.my_plant = Plant.PlantLC(rl=self.rL, l=self.L, c=self.C,x=self.X, t_sample=self.t_sample_feature)
        # action = np.clip(action, *self.action_bound) # 将数组中的元素限制在a_min, a_max之间
        self.action_view[0] = action[0]
        self.action_view[1] = action[1]
        self.action_view[3] = action[2]
        self.action_view[4] = action[3]

        self.action_view[5] = action[4]
        self.action_view[6] = action[5]
        self.action_view[7] = action[6]
        self.action_view[8] = action[7]
        self.action_view[9] = action[8]
        self.action_view[10] = action[9]

        self.my_pr = PR_CON.PRControl(kp=action[0], kr=action[1], wc=5, wr=100 * math.pi, t_sample= self.t_sample_feature)
        self.my_pr3 = PR_CON.PRControl(kp=action[2], kr=action[3], wc=5, wr=300 * math.pi, t_sample=self.t_sample_feature)
        self.my_pr25 = PR_CON.PRControl(kp=action[4], kr=action[5], wc=5, wr=2500 * math.pi, t_sample= self.t_sample_feature)
        self.my_pr47 = PR_CON.PRControl(kp=action[6], kr=action[7], wc=5, wr=4700 * math.pi, t_sample=self.t_sample_feature)
        self.my_pr50 = PR_CON.PRControl(kp=action[8], kr=action[9], wc=5, wr=5000 * math.pi, t_sample= self.t_sample_feature)
        step_t = 0.24
        setpoint_list = Signal_Generator.mix_signal(self.A1, self.A3, self.A25, self.A47, self.A50, step_t, 1/self.t_sample_feature)  # 参考/给定输入
        #对输入信号的FFT测试
        # fft_setpoint_list = fft(setpoint_list,(int)(step_t/self.t_sample_feature))
        # mX = np.abs(fft_setpoint_list)  # magnitude
        # freq_axis = np.arange((int)(step_t/self.t_sample_feature)) / (int)(step_t/self.t_sample_feature) * 1/self.t_sample_feature
        #
        # a_fft1=mX[(int)(50*step_t)]
        # a_fft3=mX[(int)(150*step_t)]
        # a_fft25=mX[(int)(1250*step_t)]
        # a_fft47=mX[(int)(2350*step_t)]
        # a_fft50=mX[(int)(2500*step_t)]
        #
        # print(a_fft1,a_fft3,a_fft25,a_fft47,a_fft50)

        step_every_step = len(np.arange(0, step_t, self.t_sample_feature))  # 每个step里运行仿真步数
        sum_r = 0
        self.input = 0.0
        self.output = 0.0
        self.final_output_list = []
        self.goal_list = []
        for step_j in range(step_every_step):
            self.goal = setpoint_list[step_j]
            self.input = self.goal - self.output
            self.my_pr.update(self.input)  # e(t)→u(t)
            self.my_pr3.update(self.input)  # e(t)→u(t)
            output_control = self.my_pr.output + self.my_pr3.output  # e(t)→u(t)  # pr输出控制量
            self.my_plant.update(output_control)  # 控制量输入被控对象 e(t)→u(t)
            self.output = self.my_plant.output
            self.final_output_list.append(self.output)
            self.goal_list.append(self.goal)
            r = -np.abs((self.output - self.goal))/(self.A1+self.A3+self.A25+self.A47+self.A50)
            if np.abs(r)>1000:
                r = 1000 * r/np.abs(r)
                break
            sum_r = sum_r + r

        self.last_step_j = step_j
        ave_sum_r=0
        if step_j > step_every_step - 10:
            out_sum_r = 100
            ave_sum_r = sum_r / step_j
            # if np.abs(ave_sum_r) > 20:
            #     out_sum_r = -np.abs(ave_sum_r) + out_sum_r
            if abs(ave_sum_r) < abs(self.last_sum_r):
                out_sum_r = out_sum_r + 100
            # elif np.abs(ave_sum_r) < 20:
                if(np.abs(ave_sum_r)<0.01):
                    out_sum_r= out_sum_r + 10000
                else:
                    out_sum_r = (1/np.abs(ave_sum_r)) * 100 + out_sum_r
        else:
            out_sum_r = -1000 + 500*step_j/step_every_step
        self.action_view[2] = out_sum_r
        self.last_sum_r = ave_sum_r
        # if ave_sum_r>0:
        self.render()
        fft_output_list = fft(self.final_output_list,(int)(step_t/self.t_sample_feature))
        mX = np.abs(fft_output_list)  # magnitude
        self.a_fft1=mX[(int)(50*step_t)]
        self.a_fft3=mX[(int)(3*50*step_t)]
        self.a_fft25=mX[(int)(25*50*step_t)]
        self.a_fft47=mX[(int)(47*50*step_t)]
        self.a_fft50=mX[(int)(50*50*step_t)]
        s = np.concatenate((np.array([self.a_fft1]), np.array([self.a_fft3]), np.array([self.a_fft25]),
                            np.array([self.a_fft47]), np.array([self.a_fft50])))

        # s = np.concatenate((np.array([self.A1]), np.array([self.A3]), np.array([self.A25]),
                            # np.array([self.A47]), np.array([self.A50]), np.array([self.L * 1e6]),
                            # np.array([self.C * 1e6]),  np.array([self.X]) ,np.array([self.rL])))
        return s, out_sum_r

    def render(self):
        plt.cla()
        plt.plot(self.final_output_list)
        plt.plot(self.goal_list)
        # plt.plot(self.action_list)
        plt.xlabel('time (s)')
        plt.ylabel('OUTPUT')
        # print('PR RL CONTROL| kp=%.3f kr=%.3f | r: %.3f' % (self.action_view[0], self.action_view[1], self.action_view[2]))
        plt.title(r'PR RL CONTROL|out_sum_r: %.3f|ave_sum_r:%.3f' % ( self.action_view[2],self.last_sum_r) + '\n' +
                  'A1=%.0f A3=%.0f rL=%.3f L=%.3fe-6 C=%.3fe-6 X=%.3f'  % (self.A1, self.A3, self.rL, self.L*1e6, self.C*1e6,self.X) + '\n' +
                  'kp1=%.3f kr1=%.3f' % (self.action_view[0], self.action_view[1]) +
                  'kp3=%.3f kr3=%.3f' % (self.action_view[3], self.action_view[4]) +
                  'kp25=%.3f kr25=%.3f' % (self.action_view[5], self.action_view[6]) + '\n' +
                  'kp47=%.3f kr47=%.3f' % (self.action_view[7], self.action_view[8])+
                  'kp50=%.3f kr50=%.3f' % (self.action_view[9], self.action_view[10])
                  )
        # plt.grid(True)
        # plt.xlim([0,210])
        # plt.ylim(self.action_bound)
        plt.draw()
        plt.pause(0.00001)


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