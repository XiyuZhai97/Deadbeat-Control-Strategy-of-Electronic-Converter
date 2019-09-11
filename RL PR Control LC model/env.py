import numpy as np
import PR_CON
import Plant
import Signal_Generator
import math
import matplotlib.pyplot as plt
import pysnooper

class InverterEnv(object):
    viewer = None
    # action_bound = [0.1,120,0.1,120,0.1,120,0.1,120,0.1,120]
    action_bound = [0.1,100,100,100,100,100,100,100]

    goal = 220
    state_dim = 11
    action_dim = 8

    def __init__(self, rl=0.1, l=200e-6, c=5e-6, t_sample=0.00001):
        self.rL = rl
        self.L = l
        self.C = c
        self.X = 12.5
        self.t_sample_feature = t_sample
        self.reset()
        self.A1 = 50
        self.A3 = 70
        self.A5 = 50
        self.A25 = 30
        self.A27 = 30
        self.A47 = 20
        self.A49 = 20
        self.input = 0.0
        self.output = 0.0
        self.out_sum_r_list = []
        self.final_output_list = []
        self.goal_list = []
        self.action_view = np.array([0., 0., 0., 0., 0.])
        self.my_plant = Plant.PlantLC(rl=rl, l=l, c=c, t_sample=t_sample)
        self.my_pr = PR_CON.PRControl(kp=0.1, kr=100, wc=5, wr=100 * math.pi, t_sample= self.t_sample_feature)
        self.my_pr3 = PR_CON.PRControl(kp=0.1, kr=70, wc=5, wr=300 * math.pi, t_sample=self.t_sample_feature)
        self.my_pr5 = PR_CON.PRControl(kp=0.1, kr=70, wc=5, wr=500 * math.pi, t_sample=self.t_sample_feature)
        self.my_pr25 = PR_CON.PRControl(kp=0.1, kr=70, wc=5, wr=2500 * math.pi, t_sample=self.t_sample_feature)
        self.my_pr27 = PR_CON.PRControl(kp=0.1, kr=70, wc=5, wr=2700 * math.pi, t_sample=self.t_sample_feature)
        self.my_pr47 = PR_CON.PRControl(kp=0.1, kr=70, wc=5, wr=4700 * math.pi, t_sample=self.t_sample_feature)
        self.my_pr49 = PR_CON.PRControl(kp=0.1, kr=70, wc=5, wr=4900 * math.pi, t_sample=self.t_sample_feature)
        self.last_step_j = 0
        self.last_sum_r = 0

    def reset(self):
        self.X = 12.5
        # self.rL = 4.44425
        # self.L = 1950e-6
        # self.C = 15e-6
        self.rL = 0.1
        self.L = 200e-6
        self.C = 1e-6
        # self.rL = np.random.random()
        # self.L = np.random.random()*400e-6
        # self.C = np.random.random()*100e-6
        # self.X = np.random.random()*10
        self.A1 = 20
        self.A3 = 5
        self.A5 = 5
        self.A25 = 0
        self.A27 = 0
        self.A47 = 0
        self.A49 = 0
        self.last_step_j = 0
        self.last_sum_r = 0
        self.out_sum_r_list = []
        # self.t_sample_feature=0.001*np.random.random()
        self.input = 0.0
        self.output = 0.0
        self.final_output_list = []
        self.goal_list = []
        self.action_view = np.array([0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0.])
        self.my_pr = PR_CON.PRControl(kp=0.1, kr=100, wc=5, wr=100 * math.pi, t_sample= self.t_sample_feature)
        self.my_pr3 = PR_CON.PRControl(kp=0.1, kr=70, wc=5, wr=300 * math.pi, t_sample=self.t_sample_feature)
        self.my_pr5 = PR_CON.PRControl(kp=0.1, kr=70, wc=5, wr=500 * math.pi, t_sample=self.t_sample_feature)
        self.my_pr25 = PR_CON.PRControl(kp=0.1, kr=70, wc=5, wr=2500 * math.pi, t_sample=self.t_sample_feature)
        self.my_pr27 = PR_CON.PRControl(kp=0.1, kr=70, wc=5, wr=2700 * math.pi, t_sample=self.t_sample_feature)
        self.my_pr47 = PR_CON.PRControl(kp=0.1, kr=70, wc=5, wr=4700 * math.pi, t_sample=self.t_sample_feature)
        self.my_pr49 = PR_CON.PRControl(kp=0.1, kr=70, wc=5, wr=4900 * math.pi, t_sample=self.t_sample_feature)
        self.my_plant = Plant.PlantLC(rl=0.1, l=200e-6, c=5e-6, t_sample= self.t_sample_feature)
        s = np.concatenate((np.array([self.A1]), np.array([self.A3]), np.array([self.A5]),
                            np.array([self.A25]), np.array([self.A27]), np.array([self.A47]),
                            np.array([self.A49]),np.array([self.L*1e6]), np.array([self.C*1e6]), np.array([self.rL]),np.array([self.X])))
        # s = np.concatenate((np.array([self.A1]),np.array([self.A3])))
        return s

    def step(self, action):

        self.A1 = 30 + 5 * round(10*np.random.random())
        self.A3 = 50 + 5 * round(10*np.random.random())
        self.A5 = 30 + 5 * round(10*np.random.random())
        self.A25 = 10 + 5 * round(10*np.random.random())
        self.A27 = 10 + 5 * round(10*np.random.random())
        self.A47 = 10 + 5 * round(10*np.random.random())
        self.A49 = 10 + 5 * round(10*np.random.random())
        # self.rL = np.random.random()
        # self.L = np.random.random()*400e-6
        # self.C = np.random.random()*100e-6
        # self.X = np.random.random()*10
        # plt.cla()
        # plt.draw()
        self.my_plant = Plant.PlantLC(rl=self.rL, l=self.L, c=self.C,x=self.X, t_sample=self.t_sample_feature)
        # action = np.clip(action, *self.action_bound) # 将数组中的元素限制在a_min, a_max之间
        # self.action_view[0] = action[0]
        # self.action_view[1] = action[1]
        # self.action_view[3] = action[2]
        # self.action_view[4] = action[3]
        #
        # self.action_view[5] = action[4]
        # self.action_view[6] = action[5]
        # self.action_view[7] = action[6]
        # self.action_view[8] = action[7]
        # self.action_view[9] = action[8]
        # self.action_view[10] = action[9]

        self.action_view[1] = action[0]
        self.action_view[2] = action[1]
        self.action_view[3] = action[2]
        self.action_view[4] = action[3]
        self.action_view[5] = action[4]
        self.action_view[6] = action[5]
        self.action_view[7] = action[6]
        self.action_view[8] = action[7]
        self.my_pr = PR_CON.PRControl(kp=action[0], kr=action[1], wc=5, wr=100 * math.pi, t_sample= self.t_sample_feature)
        self.my_pr3 = PR_CON.PRControl(kp=action[0], kr=action[2], wc=5, wr=300 * math.pi, t_sample=self.t_sample_feature)
        self.my_pr5 = PR_CON.PRControl(kp=action[0], kr=action[3], wc=5, wr=500 * math.pi, t_sample=self.t_sample_feature)
        self.my_pr25 = PR_CON.PRControl(kp=action[0], kr=action[4], wc=5, wr=2500 * math.pi, t_sample= self.t_sample_feature)
        self.my_pr27 = PR_CON.PRControl(kp=action[0], kr=action[5], wc=5, wr=2700 * math.pi, t_sample=self.t_sample_feature)
        self.my_pr47 = PR_CON.PRControl(kp=action[0], kr=action[6], wc=5, wr=4700 * math.pi, t_sample=self.t_sample_feature)
        self.my_pr49 = PR_CON.PRControl(kp=action[0], kr=action[7], wc=5, wr=4900 * math.pi, t_sample= self.t_sample_feature)
        # self.my_pr = PR_CON.PRControl(kp=0.02, kr=100, wc=5, wr=100 * math.pi, t_sample= self.t_sample_feature)
        # self.my_pr3 = PR_CON.PRControl(kp=0.02, kr=100, wc=5, wr=300 * math.pi, t_sample=self.t_sample_feature)
        # self.my_pr25 = PR_CON.PRControl(kp=0.02, kr=100, wc=5, wr=2500 * math.pi, t_sample= self.t_sample_feature)
        # self.my_pr47 = PR_CON.PRControl(kp=0.02, kr=100, wc=5, wr=4700 * math.pi, t_sample=self.t_sample_feature)
        # self.my_pr50 = PR_CON.PRControl(kp=0.02, kr=100, wc=5, wr=5000 * math.pi, t_sample= self.t_sample_feature)
        step_t = 0.24
        setpoint_list = Signal_Generator.mix_signal(self.A1, self.A3, self.A5, self.A25,self.A27,self.A47, self.A49, step_t, 1/self.t_sample_feature)  # 参考/给定输入
        step_every_step = len(np.arange(0, step_t, self.t_sample_feature))  # 每个step里运行仿真步数
        sum_r = 0
        self.input = 0.0
        self.output = 0.0
        self.final_output_list = []
        self.goal_list = []
        global step_j
        for step_j in range(step_every_step):
            self.goal = setpoint_list[step_j]
            self.input = self.goal - self.output
            self.my_pr.update(self.input)  # e(t)→u(t)
            self.my_pr3.update(self.input)  # e(t)→u(t)
            self.my_pr5.update(self.input)  # e(t)→u(t)
            self.my_pr25.update(self.input)  # e(t)→u(t)
            self.my_pr27.update(self.input)  # e(t)→u(t)
            self.my_pr47.update(self.input)  # e(t)→u(t)
            self.my_pr49.update(self.input)  # e(t)→u(t)
            output_control = self.my_pr.output + self.my_pr3.output + self.my_pr5.output + self.my_pr25.output + self.my_pr27.output + self.my_pr47.output+ self.my_pr49.output # e(t)→u(t)  # pr输出控制量
            self.my_plant.update(output_control)  # 控制量输入被控对象 e(t)→u(t)
            self.output = self.my_plant.output
            self.final_output_list.append(self.output)
            self.goal_list.append(self.goal)
            r = -np.abs((self.output - self.goal))/(self.A1+self.A3+self.A5+self.A25+self.A27+self.A47+self.A49)
            if np.abs(r)>1000:
                r = 1000 * r/np.abs(r)
                break
            sum_r = sum_r + r
        self.last_step_j = step_j
        ave_sum_r=0
        if step_j > step_every_step - 10:
            out_sum_r = 0
            ave_sum_r = sum_r / step_j
            if abs(ave_sum_r) < abs(self.last_sum_r):
                out_sum_r = out_sum_r + 100
                if(np.abs(ave_sum_r)<0.01):
                    out_sum_r= out_sum_r + 10000
                else:
                    out_sum_r = (1/np.abs(ave_sum_r)) * 100 + out_sum_r
        else:
            out_sum_r = -1000 + 500*step_j/step_every_step
        self.action_view[0] = out_sum_r
        self.out_sum_r_list.append(out_sum_r)
        self.last_sum_r = ave_sum_r
        # if out_sum_r>1000:
        # f = open("./interest2.txt", 'a+')

        print('out_sum_r',out_sum_r , '\r')
        self.render()
        # s = np.concatenate((np.array([self.A1]),np.array([self.A3])))
        s = np.concatenate((np.array([self.A1]), np.array([self.A3]), np.array([self.A5]),
                            np.array([self.A25]), np.array([self.A27]), np.array([self.A47]),
                            np.array([self.A49]),np.array([self.L*1e6]), np.array([self.C*1e6]), np.array([self.rL]), np.array([self.X])))
        return s, out_sum_r

    def render(self):
        plt.cla()
        # f = open("./interest.txt", 'a+')
        # t_x=np.arange(0,self.t_sample_feature * (len(self.final_output_list)), self.t_sample_feature)
        plt.plot(self.final_output_list, label='Uo')
        plt.plot(self.goal_list, label='Uref')
        plt.xlabel('time (s)')
        plt.ylabel('OUTPUT')
        print('ave_sum_r:%.5f' %(self.last_sum_r) ,'A1=%.0f A3=%.0f A5=%.0f A25=%.0f A27=%.0f A47=%.0f A49=%.0f || rL=%.3f L=%.3fe-6 C=%.3fe-6'  %
              (self.A1, self.A3,self.A5, self.A25,self.A27, self.A47,self.A49, self.rL, self.L*1e6, self.C*1e6) + '\n' +
                  'kp=%.3f kr1=%.3f kr3=%.3f kr5=%.3f kr25=%.3f kr27=%.3f kr47=%.3f kr49=%.3f' %
              (self.action_view[1], self.action_view[2], self.action_view[3], self.action_view[4], self.action_view[5], self.action_view[6], self.action_view[7], self.action_view[8] ))
        plt.grid(True)
        plt.xlim([2000,7000])
        # plt.ylim(self.action_bound)
        plt.legend(loc='upper right')
        # plt.show
        plt.ion()
        plt.draw()
        # plt.savefig ('image\Outputfig' + str(i) + '.jpg')

        # plt.show()
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