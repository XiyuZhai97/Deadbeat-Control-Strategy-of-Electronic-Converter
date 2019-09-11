#!/usr/bin/python
# -*- coding: utf-8 -*-

import PR_CON
import Plant
import Signal_Generator
import math
import matplotlib.pyplot as plt
import numpy as np
# import harold as ha
# from scipy.interpolate import spline
#from scipy.interpolate import BSpline, make_interp_spline #  Switched to BSpline


def test_pr(t=0.24, fs=1000, num_f=1):
    # 参考信号 谐波生成
    x = Signal_Generator.mix_signal(num_f, t, fs)
    n = np.arange(0, t, 1 / fs)
    L = len(n)# L仿真步数
    # plt.plot(n, x)
    # =========================================
    final_output = 0 # 也即plant最后输出
    final_output_list = []
    setpoint_list = x # 参考/给定输入
    my_pr = PR_CON.PRControl(kp=0.1, kr=100, wc=5, wr=100*math.pi, t_sample=1 / fs)
    my_plant= Plant.PlantLC(rl=0.1, l=200e-6, c=5e-6, t_sample=1 / fs)
    for i in range(0, L):
        my_pr.update(x[i]- final_output) # e(t)→u(t)
        output_control = my_pr.output # pr输出控制量
        my_plant.update(output_control) # 控制量输入被控对象 e(t)→u(t)
        final_output = my_plant.output
        final_output_list.append(final_output) # 在列表末尾添加新的对象。
    plt.plot(n, final_output_list)
    plt.plot(n, setpoint_list)
    plt.xlabel('time (s)')
    plt.ylabel('PR')
    plt.title('TEST PR')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    t = 0.24
    fs = 4000
    num_f=1
    test_pr(t, fs, num_f)
