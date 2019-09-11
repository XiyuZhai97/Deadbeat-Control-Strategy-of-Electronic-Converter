import numpy as np
import matplotlib.pyplot as plt
'''
t  (float) : 生成序列的时长
A  (float) : amplitude
f0 (float) : frequency
fs (float) : sample rate
phi(float) : initial phase
returns
x (numpy array): sinusoid signal sequence
'''

def generate_sinusoid(t, A, f0, fs, phi):

    T = 1.0 / fs
    N = t / T
    n = np.arange(N)  # [0,1,..., N-1]
    x = A * np.sin(2 * f0 * np.pi * n * T + phi)
    return x

def mix_signal(A1,A3,A25,A47,A50,t,fs): #叠加n-1个谐波,生成t秒波形
    f0 = 50
    phi = 0
    mixsignal=0;
    mixsignal = mixsignal + generate_sinusoid(t, A1, f0 * (2*1-1), fs, phi)
    mixsignal = mixsignal + generate_sinusoid(t, A3, f0 * (2*2-1), fs, phi)
    mixsignal = mixsignal + generate_sinusoid(t, A25, f0 * (25), fs, phi)
    mixsignal = mixsignal + generate_sinusoid(t, A47, f0 * (47), fs, phi)
    mixsignal = mixsignal + generate_sinusoid(t, A50, f0 * (50), fs, phi)

    return mixsignal
# t=0.04
# fs = 1000000
# x = mix_signal(3,t,fs)
# n = np.arange(0, t, 1/fs)
# plt.plot(n, x)
# print (x)
# print (n)
# plt.show()