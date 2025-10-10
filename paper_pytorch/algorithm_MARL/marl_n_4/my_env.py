from __future__ import division
import numpy as np
import time
import random
import math
from scipy.special import jv

np.random.seed(1234)

class Communication_Env:
    def __init__(self):
        self.Decorrelation_distance = 50
        self.BS_position = [0, 0, 25]
    
    def small_scale_path_loss(self, kappa, h_path, e_path):
        return kappa*h_path + e_path


    def get_path_loss(self, position_A):
        d1 = abs(position_A[0] - self.BS_position[0])
        d2 = abs(position_A[1] - self.BS_position[1])
        d3 = abs(position_A[2] - self.BS_position[2])
        distance = math.hypot(d1, d2, d3)
        return 128.1 + 37.6 * np.log10(distance) #+ (self.h_bs - self.h_ms) ** 2) / 1000) # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):
        nVeh = len(shadowing)
        self.R = np.sqrt(0.5 * np.ones([nVeh, nVeh]) + 0.5 * np.identity(nVeh))
        return np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), shadowing) \
               + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0, 8, nVeh)

    def channel_gain(self,small_scale, large_scale):
        G_channel_gain = (abs(small_scale) ** 2) * large_scale
        return G_channel_gain

class Vehicle:
    def __init__(self, lane_num):
        self.lane_num = lane_num
        self.lane_width = 3.75
        self.L0 = 8
    
    def ini_position(self):
        ind = np.random.randint(self.lane_num)
        velocity = np.random.randint(10,15)
        ini_x = np.random.poisson(20)
        ini_y = ind*self.lane_width + self.L0
        return ind, velocity, [ini_x, ini_y, 0]
    
    def dt_tk_generate(self):
        generate_tk = np.random.uniform(1000,1500) #Byte
        generate_dt = np.random.uniform(10, 200) #Byte
        dt = generate_dt
        tk = generate_tk
        return dt, tk

    def vehicle_property(self):
        DT, TK = self.dt_tk_generate()
        lane, velocity, ini_pos = self.ini_position()
        return {'v_lane': lane, 'position': ini_pos, 'velocity': velocity, 
                'compute_task': TK,'updated_data': DT}

class Environment:
    def __init__(self,lane_num, n_veh):
        self.lane_num = lane_num
        self.n_Veh = n_veh
        #self.width = width
        self.channel = Communication_Env()

        self.trans_power = 23 #dbm
        #self.T = 4
        self.noise = -110 #dBm
        self.noise1 = [10**(self.noise/10) for i in range(self.n_Veh)]
        self.fc = 2#GHZ
        self.c = 3e8
        self.bsAntgain = 8
        self.bsnoise = 5
        self.veAntgain = 3
        self.t_fast = 0.001 
        self.t_slot = 0.1
        self.band_width = 20 #MHZ
        self.bs_max_fre = 100 #GHZ
        self.byte_fc = 0.25#MHZ/Byte
        self.time_limit_dt = 0.1#s
        self.time_limit_tk = 0.5#s
        self.penalty = 1
        
        self.g_channel = []
        self.h_path = []
        self.large_scaled = []
        self.small_scaled = []
        self.V2I_path_loss = []
        self.V2I_shadowing = []
        #self.shadowing = []
        self.delt_distance = []
        self.vehicles = []
        self.ve_v = []
        self.ve_l = []
        self.upd_size = []
        self.task_size = []
        self.taks_fc = []
        self.upd_fc = []
        self.upd_t_limit = []
        self.tk_t_limit = []
        self.t_DT = []
        self.t_TK = []
    
    def add_new_vehicles_by_number(self,n):
        for i in range(n):
            self.vehicles.append(Vehicle(self.lane_num).vehicle_property())

        self.upd_size = np.zeros(self.n_Veh)
        self.task_size = np.zeros(self.n_Veh)
        self.taks_fc = np.zeros(self.n_Veh)
        self.upd_fc = np.zeros(self.n_Veh)
        self.upd_t_limit = np.zeros(self.n_Veh)
        self.tk_t_limit = np.zeros(self.n_Veh)
        self.ve_v = np.zeros(self.n_Veh)
        self.ve_l = np.zeros(self.n_Veh)
        self.t_TK = np.zeros(self.n_Veh)
        self.t_DT = np.zeros(self.n_Veh)
        
        for i in range(self.n_Veh):
            self.upd_size[i] = self.vehicles[i]['updated_data']
            self.task_size[i] = self.vehicles[i]['compute_task']
            self.upd_t_limit[i] = self.time_limit_dt
            self.tk_t_limit[i] = self.time_limit_tk
            self.taks_fc[i] = self.byte_fc
            self.upd_fc[i] = self.byte_fc
            self.ve_v[i] = self.vehicles[i]['velocity']
            self.ve_l[i] = self.vehicles[i]['position'][0]
            

        #initialized channel
        self.V2I_path_loss = np.zeros(self.n_Veh)
        self.e_path = np.zeros(self.n_Veh)
        self.large_scaled = np.zeros(self.n_Veh)
        self.small_scaled = np.zeros(self.n_Veh)
        self.g_channel = np.zeros(self.n_Veh)
        self.V2I_shadowing = np.random.normal(0, 8, self.n_Veh)
        self.delt_distance = np.asarray([c['velocity']*self.t_slot for c in self.vehicles])
        real = np.random.normal(0, 1, self.n_Veh)
        imag = np.random.normal(0, 1, self.n_Veh)
        self.h_path = real + 1j*imag


    '车辆的位置更新'
    def vehicle_renew_position(self):
        i = 0
        while i < self.n_Veh:
            v_position = self.vehicles[i]['position']
            velocity = self.ve_v[i]
            v_x = v_position[0] + velocity * self.t_slot
            #if v_position[0] < self.width/2 and v_x < self.width/2:
            self.vehicles[i]['position'][0] = v_x
            self.ve_l[i] = self.vehicles[i]['position'][0]
            #else:
            #    self.vehicles[i]['position'][0] = -self.width/2
            #    self.ve_l[i] = -self.width/2
            i += 1

    def renew_channel(self):
        fd = np.zeros(self.n_Veh)
        cofe = np.zeros(self.n_Veh)
        self.V2I_shadowing = self.channel.get_shadowing(self.delt_distance, self.V2I_shadowing)

        for x in range(self.n_Veh):
            fd[x] = self.ve_v[x]*self.fc*1e9/self.c #Doppler frequency
            
            cofe[x] = jv(0, 2*math.pi*fd[x]*self.t_slot)
            e_real = np.random.normal(0, 1-cofe[x]**2, self.n_Veh)
            e_imag = np.random.normal(0, 1-cofe[x]**2, self.n_Veh)
            self.e_path = e_real + 1j*e_imag 

        for i in range(self.n_Veh):
            self.V2I_path_loss[i] = self.channel.get_path_loss(self.vehicles[i]['position'])
            self.small_scaled[i] = self.channel.small_scale_path_loss(cofe[i], self.h_path[i], self.e_path[i])
            self.h_path[i] = self.small_scaled[i]
        
        self.large_scaled = self.V2I_path_loss + self.V2I_shadowing

        for i in range(self.n_Veh):
            self.g_channel[i] = self.channel.channel_gain(self.small_scaled[i], self.large_scaled[i])
    
    def step(self, action):
        per_user_reward = np.zeros(self.n_Veh)
        self.V2I_rate = []

        signal = 10**((self.trans_power + self.g_channel)/10)
        self.V2I_rate = self.band_width*np.log2(1 + np.divide(signal, self.noise1))  # transmission rate from V to BS

        for i in range(self.n_Veh):
            #task_size = self.task_size[i]
            #upd_size = self.upd_size[i]
            #f_i_TK = action[i][0]
            #c_i_tk = self.byte_fc
            #f_i_DT = action[i][1]
            #c_i_DT = self.byte_fc
            V2I_rate = self.V2I_rate
            t_tk = self.task_size[i]*8 / (V2I_rate[i]*1e6) + self.task_size[i]*self.taks_fc[i] / (action[i][0]*1000) 
            t_dt = self.upd_size[i]*8 / (V2I_rate[i]*1e6) + self.upd_size[i]*self.upd_fc[i]*self.n_Veh  / (action[i][1]*1000)
            self.t_TK[i] = t_tk
            self.t_DT[i] = t_dt


            if np.sum(action) > self.bs_max_fre:
                for j in range(self.n_Veh):
                    per_user_reward[j] =  -(0.8*(1-np.log(1+np.divide(self.t_DT[j], self.time_limit_dt))) + \
                                            0.2*(1-np.log(1+np.divide(self.t_TK[j], self.time_limit_tk))))+0.01*(self.bs_max_fre-np.sum(action))
            else:    
                for k in range(self.n_Veh):
                    per_user_reward[k] = 0.8*(1-np.log(1+np.divide(self.t_DT[k], self.time_limit_dt))) + \
                                         0.2*(1-np.log(1+np.divide(self.t_TK[k], self.time_limit_tk)))
        #else:
        #    for m in range(self.n_Veh): 
        #            per_user_reward[m] = 0.5*(1-np.log(1+np.divide(self.t_DT[m], self.time_limit_dt))) + \
        #                                 0.5*(1-np.log(1+np.divide(self.t_TK[m], self.time_limit_tk))) -0.1
        
        s_next = self.state()
        
        global_reward = np.mean(per_user_reward)
        return  s_next, per_user_reward, global_reward

    def state(self):
        self.vehicle_renew_position()
        self.renew_channel()
        s = np.reshape(np.concatenate((self.task_size, self.taks_fc, self.tk_t_limit, 
                                       self.upd_size, self.upd_fc, self.upd_t_limit, self.ve_v, self.ve_l, self.g_channel)),[9, self.n_Veh])
        s = np.transpose(s)
        return s
    
    def make_new_game(self, n_veh=0):
        self.vehicles = []
        if n_veh > 0:
            self.n_Veh = n_veh
        self.add_new_vehicles_by_number(self.n_Veh)
        self.renew_channel()
        ini_s = np.reshape(np.concatenate((self.task_size, self.taks_fc, self.tk_t_limit, 
                                       self.upd_size, self.upd_fc, self.upd_t_limit, self.ve_v, self.ve_l, self.g_channel)),[9, self.n_Veh])
        ini_s = np.transpose(ini_s)
        return ini_s