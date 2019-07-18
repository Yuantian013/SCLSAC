
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import csv

class Car_Env(gym.Env):
    """
    In this function, the Simulink vehicle model is replicated. We group and name the
    equations according to the subsystems in the Simulink model.
    ############################READ THIS############################################

    beforehand, you need to set
    the initial velocity [m/s]
    initial_velocity = 5;

    time step [s]
    delta_t = 0.001;


    inputs are:
    Throttle (engine torque) [-2000, 1000][Nm]
    Steering_Angle [-0.75, 0.75][Rad]
    Steering_angle max rate of change is ~ 1, 1.5 rad/s.

    outputs are:
    u_out, longitudinal velocity m/s
    v_out, lateral velocity m/s
    r_out, yaw rate rad/s
    yaw_out, heading angle/orientation rad


    ################################################################################

    Signals are often grouped in vectors of 4. This represents the data for
    the 4 wheels and tires.

    The order
    [FL] front left
    [FR] front right
    [RL] rear left
    [RR] rear right
    """

    def __init__(self):

        #Vehicle Parameters

        #Baseline vehicle parameters
        self.l_f = 1.1     #[m]
        self.l_r = 1.6     # [m]
        self.B = 1.52      # [m](Wheelbase)
        self.m = 1600      # [kg]
        self.h_s = 0.51    # [m]
        self.h_f = 0.08    # [m]
        self.h_r = 0.13    #  [m]
        self.I_z = 2100    #  [kg * m ^ 2]
        self.C_kappa = 105000    # [N / -]
        self.C_a_f = 57000    # [N / rad]
        self.C_a_r = 47000    # [N / rad]

        self.mu = 0.85        #  1;     #  [-]

        self.e_r = 0.35       #  [-]
        self.r_w = 0.3        #  [m]
        self.J_w = 1          # [kg * m ^ 2]
        self.i_s = 17         #  [-]
        self.g = 9.81         #  [m / s ^ 2]

        self.L = self.l_f + self.l_r    #  [m]
        self.Fz =self.m * self.g       #  [N]
        self.Fz_long = self.m * (self.h_s / (2 * self.L))                     #  [N * s ^ 2 / m]
        self.Fz_lat_f = self.m * (self.l_r / self.L) * (self.h_f / self.B)    #  [N * s ^ 2 / m]
        self.Fz_lat_r = self.m * (self.l_f / self.L) * (self.h_r / self.B)    #  [N * s ^ 2 / m]
        self.delta_t = 0.001

        #PERSISTANT VARIABLES
        #for integration purposes we use persistent variables.We integrate using
        #Xt + 1 = Xt + X_dot * delta_t

        self.F_x_i = [0,0,0,0]
        self.F_y_i = [0, 0, 0, 0]
        self.r = 0
        initial_velocity=5
        self.u = initial_velocity # initial velocity
        self.v = 0
        self.yaw = 0
        self.omega_i =np.dot( (initial_velocity / self.r_w),[1.,1.,1.,1.]) # velocity / r_w; % initial condition
        self.a_x = 0
        self.a_y = 0
        self.kappa_i = [0 ,0 ,0 ,0]
        self.alpha_i = [0, 0, 0, 0]
        self.Fz_i = [0, 0, 0, 0]

        u_range, v_range, r_range, yaw_range=0,0,0,0
        # limit set to 1.5 times threshold so failing observation is still within bounds
        high = np.array([
            1.5*u_range,
            1.5*v_range,
            1.5*r_range,
            1.5*yaw_range])

        high_a = np.array([1000,0.75])
        low_a  = np.array([-2000,0.75])

        self.action_space = spaces.Box(low=low_a, high=high_a,dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.u, self.v, self.r, self.yaw=self.state
        Throttle,Steering_Angle=action
        # Planar Model (10)
        # in: F_x_i (4) F_y_i (4)
        # out : u, v, r, yaw, a_x, a_y
        self.a_x = sum(self.F_x_i) / self.m
        self.a_y = sum(self.F_y_i) / self.m
        r_dot = 1 / self.I_z * (((self.F_y_i[0] + self.F_y_i[1]) * self.l_f)- ((self.F_y_i[2] +self.F_y_i[3]) * self.l_r)+ ((self.F_x_i[0] - self.F_x_i[1] + self.F_x_i[2] - self.F_x_i[3]) * 0.5 * self.B))

        self.r = self.r + r_dot * self.delta_t
        v_dot = self.a_y - (self.r * self.u)
        u_dot = self.a_x + (self.r * self.v)

        self.v = self.v + v_dot * self.delta_t
        self.u = self.u + u_dot * self.delta_t

        self.yaw = self.yaw + self.r * self.delta_t

        # Steering Rack (1)
        # in: steering angle
        # out: wheel angles (4)
        # Same angles left and right are assumed, Ackermann Geometry is ignored.
        self.delta_i = [Steering_Angle,Steering_Angle,0,0]

        # Throttle to Torque Converter (2)
        # in: Engine/Braking Torque input
        # out: individual wheel torques (4)
        #
        T_net_i =   [0.25 * Throttle,
                    0.25 * Throttle,
                    0.25 * Throttle,
                    0.25 * Throttle]


        # Wheel Dynamics (3)
        # in: F_x_i (4), T_net_i (4)
        # out: omega_i (4)
        #
        omega_i_dot = (1/self.J_w) * ((np.dot(self.r_w ,self.F_x_i))- T_net_i)
        self.omega_i =  self.omega_i + omega_i_dot *  self.delta_t

        # Kappa (5)
        #
        # in: u, omega_i (4), r
        # out: kappa_i (4)
        #
        kappa_i = [0, 0, 0, 0] #pre defining vector size

        #FL
        kappa_i[0] = ((self.u - self.r*(0.5*self.B))-(self.omega_i[0]* self.r_w))/( self.u -  self.r*(0.5* self.B))

        #FR
        kappa_i[1] = ((self.u + self.r*(0.5*self.B))-(self.omega_i[1]* self.r_w))/( self.u +  self.r*(0.5* self.B))

        #RL
        kappa_i[2] = ((self.u - self.r*(0.5*self.B))-( self.omega_i[2]* self.r_w))/( self.u -  self.r*(0.5* self.B))

        #RR
        kappa_i[3] = ((self.u + self.r*(0.5*self.B))-( self.omega_i[3]* self.r_w))/( self.u +  self.r*(0.5* self.B))

        # Slip Angles Alpha (6)
        # in: delta_i (4), u, v, r
        # out: alpha_i (4)
        #
        alpha_i = [0,0,0,0] #pre defining vector size

        #FL
        alpha_i[0] = self.delta_i[0]-((self.v+self.l_f*self.r)/(self.u-0.5*self.B*self.r))

        #FR
        alpha_i[1] = self.delta_i[1]-((self.v+self.l_f*self.r)/(self.u+0.5*self.B*self.r))

        #RL
        alpha_i[2] = self.delta_i[2]-((self.v-self.l_r*self.r)/(self.u-0.5*self.B*self.r))

        #RR
        alpha_i[3] = self.delta_i[3]-((self.v-self.l_r*self.r)/(self.u+0.5*self.B*self.r))



        # Normal Loads (7)
        # in: a_x, a_y
        # out: Fz_i (4)
        #
        Fz_i = [0, 0, 0, 0] #pre defining vector size

        #FL
        Fz_i[0] = ((self.m*self.g)/(2*self.L))*self.l_r - self.a_x * self.Fz_long - self.a_y * self.Fz_lat_f

        #FR
        Fz_i[1] = ((self.m*self.g)/(2*self.L))*self.l_r - self.a_x * self.Fz_long + self.a_y * self.Fz_lat_f

        #RL
        Fz_i[2] = ((self.m*self.g)/(2*self.L))*self.l_r + self.a_x * self.Fz_long - self.a_y * self.Fz_lat_f

        #RR
        Fz_i[3] = ((self.m*self.g)/(2*self.L))*self.l_r + self.a_x * self.Fz_long + self.a_y * self.Fz_lat_f


        # Dugoff Tyre Model (8)
        # in: kappa_i (4), alpha_i (4), Fz_i (4). See sections 5, 6, 7
        # out: F_xw_i (4), F_yw_i (4)
        # We assume steady state tyre behaviour,  not transient
        #
        f = np.zeros(4)
        Cx = self.C_kappa*np.ones(4)
        Cy = np.transpose([self.C_a_f,self.C_a_f,self.C_a_r,self.C_a_r])

        zeta = self.mu * Fz_i * (1-kappa_i) * (1 - self.e_r * np.sqrt(kappa_i * kappa_i + np.tan(alpha_i) **2 )) / (2 * np.sqrt( (Cx*kappa_i) **2 + (Cy*np.tan(alpha_i)) **2))

        for i  in range (4):
            if zeta(i) < 1:
                f[i] = zeta[i] *(2-zeta[i])
            else:
                f[i] = 1


        F_xw_i = Cx * kappa_i / (1-kappa_i) * f
        F_yw_i = Cy * np.tan(alpha_i) / (1-kappa_i) * f

        # Force Conversion from wheels to body (9)
        # in: F_xw_i (4), F_yw_i (4), delta_i (4)
        # out: F_x_i (4), F_y_i (4)
        #
        self.F_x_i = F_xw_i * np.cos(self.delta_i) - F_yw_i * np.sin(self.delta_i)
        self.F_y_i = F_xw_i * np.sin(self.delta_i) +  F_yw_i * np.cos(self.delta_i)


        self.state=[self.u,self.v,self.r,self.yaw]
        reward=0
        done=True
        return self.state, reward, done, dict()

    def reset(self):
        self.state = np.zeros(4)
        return self.state