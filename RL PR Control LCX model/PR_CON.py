#!/usr/bin/python
# -*- coding: utf-8 -*-
# ==============================================================================
import math


class PRControl:
    """PR Controller
    G(s)=Kp+(2*Kr*Wc*s)/(s^2+2*Wc*s+Wr)
    Kp--PR控制器的比例系数
    Kr--谐振系数
    Wc--截止频率
    Wr--谐振频率
    """
    def __init__(self, kp=0.1, kr=100, wc=5, wr=100*math.pi, t_sample=0.0000001):
        # self.Kp = kp
        # self.Kr = kr
        # self.Wc = wc
        # self.Wr = wr
        self.prb0 = (kp*pow(wr,2)*pow(t_sample,2) + 4*(kp+kr)*wc*t_sample + 4*kp)/(pow(wr,2)*pow(t_sample,2) + 4*wc*t_sample +4)
        self.prb1 = (2*kp*pow(wr,2)*pow(t_sample,2) - 8*kp)/(pow(wr,2)*pow(t_sample,2) + 4*wc*t_sample +4)
        self.prb2 = (kp*pow(wr,2)*pow(t_sample,2) - 4*(kp+kr)*wc*t_sample + 4*kp)/(pow(wr,2)*pow(t_sample,2) + 4*wc*t_sample +4)
        self.pra1 = (2* pow(wr,2)*pow(t_sample,2) - 8)/(pow(wr,2)*pow(t_sample,2) + 4*wc*t_sample +4)
        self.pra2 = (pow(wr,2)*pow(t_sample,2) - 4*wc*t_sample +4)/(pow(wr,2)*pow(t_sample,2) + 4*wc*t_sample +4)
        self.clear()
        # self.In_Reference = 0.0
        self.error=0.0
        self.last1_error=0.0
        self.last2_error=0.0
        self.output = 0.0
        self.last1_output=0.0
        self.last2_output=0.0
    def clear(self):
        # self.In_Reference = 0.0
        self.output = 0.0
        self.error=0.0
        self.last1_error=0.0
        self.last2_error=0.0
    # def tuner(self, kp, kr, wc, wr):
    #     self.prb0 = (kp*pow(wr,2)*pow(self.t_sample,2) + 4*(kp+kr)*wc*self.t_sample + 4*kp)/(pow(wr,2)*pow(self.t_sample,2) + 4*wc*self.t_sample +4)
    #     self.prb1 = (2*kp*pow(wr,2)*pow(self.t_sample,2) - 8*kp)/(pow(wr,2)*pow(self.t_sample,2) + 4*wc*self.t_sample +4)
    #     self.prb2 = (kp*pow(wr,2)*pow(self.t_sample,2) - 4*(kp+kr)*wc*self.t_sample + 4*kp)/(pow(wr,2)*pow(self.t_sample,2) + 4*wc*self.t_sample +4)
    #     self.pra1 = (2* pow(wr,2)*pow(self.t_sample,2) - 8)/(pow(wr,2)*pow(self.t_sample,2) + 4*wc*self.t_sample +4)
    #     self.pra2 = (pow(wr,2)*pow(self.t_sample,2) - 4*wc*self.t_sample +4)/(pow(wr,2)*pow(self.t_sample,2) + 4*wc*self.t_sample +4)


    def update(self, feedback_value):
        """Clears PID computations and coefficients
        u(k)=-a1*u(k-1)-a2*u(k-2)+b0*e(k)+
        """
        self.error = feedback_value # self.In_Reference - feedback_value
        self.output = self.prb0*self.error + self.prb1*self.last1_error + self.prb2*self.last2_error - self.pra1*self.last1_output - self.pra2*self.last2_output
        self.last2_output = self.last1_output
        self.last1_output = self.output
        self.last2_error = self.last1_error
        self.last1_error = self.error


    # def setKp(self, proportional_gain):
    #     """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
    #     self.Kp = proportional_gain
    #
    # def setKi(self, integral_gain):
    #     """Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
    #     self.Ki = integral_gain
    #
    # def setKd(self, derivative_gain):
    #     """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
    #     self.Kd = derivative_gain

    # def setWindup(self, windup):
    #     """Integral windup, also known as integrator windup or reset windup,
    #     refers to the situation in a PID feedback controller where
    #     a large change in setpoint occurs (say a positive change)
    #     and the integral terms accumulates a significant error
    #     during the rise (windup), thus overshooting and continuing
    #     to increase as this accumulated error is unwound
    #     (offset by errors in the other direction).
    #     The specific problem is the excess overshooting.
    #     """
    #     self.windup_guard = windup



