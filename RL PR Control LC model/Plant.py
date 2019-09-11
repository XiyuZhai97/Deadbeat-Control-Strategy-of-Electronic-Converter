#!/usr/bin/python
# -*- coding: utf-8 -*-
# ==============================================================================
import math


class PlantLC:
    """LC滤波器
    G(s)=1/(L*C*s^2 +C*rL*s+1)
    """
    def __init__(self, rl=0.1, l=200e-6, c=5e-6,x=10, t_sample=0.0000001):
        self.rL = rl
        self.L = l
        self.C = c
        self.X = x
        # self.plantb0 = pow(t_sample,2)/(4*l*c+2*c*rl*t_sample+pow(t_sample,2))
        # self.plantb1 = 2*pow(t_sample,2)/(4*l*c+2*c*rl*t_sample+pow(t_sample,2))
        # self.plantb2 = pow(t_sample,2)/(4*l*c+2*c*rl*t_sample+pow(t_sample,2))
        # self.planta1 = (2*pow(t_sample,2) - 8*l*c)/(4*l*c+2*c*rl*t_sample+pow(t_sample,2))
        # self.planta2 = (pow(t_sample,2) + 4*l*c - 2*c*rl*t_sample)/(4*l*c+2*c*rl*t_sample+pow(t_sample,2))
        self.plantb0 = 0
        self.plantb1 = 0
        self.plantb2 = pow(t_sample,2)/(c*l)
        self.planta1 = t_sample * (c*rl*x+l)/(c*l*x)-2
        self.planta2 = 1+ t_sample *( -(c*rl*x+l)+t_sample*(x+rl))/(c*l*x)
        self.clear()
        self.input = 0.0
        self.last1_input=0.0
        self.last2_input=0.0
        self.output = 0.0
        self.last1_output=0.0
        self.last2_output=0.0
    def clear(self):
        """Clears PID computations and coefficients"""
        self.input = 0.0
        self.last1_input=0.0
        self.last2_input=0.0
        self.output = 0.0
        self.last1_output=0.0
        self.last2_output=0.0
    def update(self, conin_value):
        self.input = conin_value
        self.output = self.plantb0*self.input + self.plantb1*self.last1_input + self.plantb2*self.last2_input - self.planta1*self.last1_output - self.planta2*self.last2_output
        self.last2_output = self.last1_output
        self.last1_output = self.output
        self.last2_input = self.last1_input
        self.last1_input = self.input

    def setSampleTime(self, sample_time):
        """be updated at a regular interval.
        Based on a pre-determined sampe time, the PID decides if it should compute or return immediately.
        """
        self.sample_time = sample_time