from __future__ import division
import numpy as np
import time
import random
import math

class Communication_Env:
    def __init__(self):
        self.h_bs = 25
        self.h_ms = 1.5
        self.cofe = 0.2
        self.path_loss_indicator = 3
        self.Decorrelation_distance = 50
        self.BS_position = [0, 0, 25]
    
    def small_scale_path_loss(self, position, h_path):
        d1 = abs(position[0] - self.BS_position[0])
        d2 = abs(position[1] - self.BS_position[1])
        d3 = abs(position[2] - self.BS_position[2])
        distance = math.hypot(d1, d2, d3)
        return self.cofe*h_path + 1/(np.linalg.norm([distance,self.h_bs])**self.path_loss_indicator)


    def get_path_loss(self, position_A):
        d1 = abs(position_A[0] - self.BS_position[0])
        d2 = abs(position_A[1] - self.BS_position[1])
        d3 = abs(position_A[2] - self.BS_position[2])
        distance = math.hypot(d1, d2, d3)
        return 128.1 + 37.6 * np.log10(math.sqrt(distance ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000) # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):
        nVeh = len(shadowing)
        self.R = np.sqrt(0.5 * np.ones([nVeh, nVeh]) + 0.5 * np.identity(nVeh))
        return np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), shadowing) \
               + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0, 8, nVeh)

    def channel_gain(self,small_scale, large_scale):
        G_channel_gain = (abs(small_scale) ** 2) * large_scale
        return G_channel_gain
    
    def SINR(self, g_channel_gain, power, noise):
        Y_sinr = (g_channel_gain * power) / (noise ** 2)
        return Y_sinr

class Vehicle:
    def __init__(self, lane, position, velocity):
        self.lane = lane
        self.position = position
        self.velocity = velocity
        self.upd_period = 0.5
        self.task = []
        self.upd_data = []
    
    def task_generate(self):
        task_size = np.random.uniform(1024, 1536) #Byte
        unit_frequency = 0.25
        max_time_limit = self.upd_period
        return [task_size, unit_frequency, max_time_limit]
     
    def update_data(self):
        data_size = np.random.uniform(1024, 1536) #Byte
        data_frequency = 0.25 #MHZ/Byte
        data_time = self.upd_period
        return[data_size, data_frequency, data_time]

    def vehicle_property(self):
        for i in self.task_generate():
            self.task.append(i)
        for i in self.update_data():
            self.upd_data.append(i)
        return {'v_lane': self.lane, 'position': self.position, 'velocity': self.velocity, 
                'compute_task': self.task,'updated_data': self.upd_data}

class Environment:
    def __init__(self,lane_num, n_veh, width):
        self.lane_num = lane_num
        self.n_Veh = n_veh
        self.width = width
        self.channel = Communication_Env()

        self.trans_power = 200 #mW
        self.lane_width = 3.75
        self.T = 4
        self.L0 = 8
        self.noise = -110#dBm
        self.noise1 = 10**(self.noise/10)
        self.t_slot = 0.1
        self.band_width = 150 #MHZ
        self.bs_max_fre = 100 #GHZ
        
        self.g_channel = []
        self.h_path = []
        self.large_scaled = []
        self.small_scaled = []
        self.V2I_path_loss = []
        self.V2I_shadowing = []
        self.shadowing = []
        self.delt_distance = []
        self.vehicles = []
        self.ve_v = []
        self.ve_l = []
        self.upd_fc = []
        self.task_fc = []
        self.upd_size = []
        self.task_size = []
        self.upd_t_limit = []
        self.tk_t_limit = []
        self.t_DT = []
        self.t_TK = []
    
    def add_new_vehicles(self, lane, position, velocity):
        self.vehicles.append(Vehicle(lane, position, velocity).vehicle_property())
    
    def add_new_vehicles_by_number(self, n):
        for i in range(n):
            ind = np.random.randint(0,self.lane_num)
            lane_y = ind * self.lane_width + self.L0
            start_position = [np.random.randint(-self.width/2,self.width/2), lane_y, 0]
            self.add_new_vehicles(ind, start_position, np.random.randint(10,15))

        self.upd_size = np.zeros(len(self.vehicles))
        self.task_size = np.zeros(len(self.vehicles))
        self.upd_fc = np.zeros(len(self.vehicles))
        self.task_fc = np.zeros(len(self.vehicles))
        self.upd_t_limit = np.zeros(len(self.vehicles))
        self.tk_t_limit = np.zeros(len(self.vehicles))
        self.ve_v = np.zeros(len(self.vehicles))
        self.ve_l = np.zeros(len(self.vehicles))
        self.t_TK = np.zeros(len(self.vehicles))
        self.t_DT = np.zeros(len(self.vehicles))
        
        for i in range(len(self.vehicles)):
            self.upd_size[i] = self.vehicles[i]['updated_data'][0]
            self.task_size[i] = self.vehicles[i]['compute_task'][0]
            self.upd_fc[i] = self.vehicles[i]['updated_data'][1]
            self.task_fc[i] = self.vehicles[i]['compute_task'][1]
            self.upd_t_limit[i] = self.vehicles[i]['updated_data'][2]
            self.tk_t_limit[i] = self.vehicles[i]['compute_task'][2]
            self.ve_v[i] = self.vehicles[i]['velocity']
            self.ve_l[i] = self.vehicles[i]['position'][0]
            

        #initialized channel
        self.g_channel = np.zeros(len(self.vehicles))
        self.V2I_shadowing = np.random.normal(0, 8, len(self.vehicles))
        self.delt_distance = np.asarray([c['velocity']*self.t_slot*5 for c in self.vehicles])
        real = np.random.normal(0, 1/2, len(self.vehicles))
        imag = np.random.normal(0, 1/2, len(self.vehicles))
        self.h_path = real + 1j*imag/math.sqrt(2)
        #self.h_path = np.random.randn(len(self.vehicles)) + 1j * np.random.randn(len(self.vehicles))

    '车辆的位置更新'
    def vehicle_renew_position(self):
        i = 0
        while i < len(self.vehicles):
            v_position = self.vehicles[i]['position']
            velocity = self.ve_v[i]
            v_x = v_position[0] + velocity * self.t_slot
            if v_position[0] < self.width/2 and v_x < self.width/2:
                self.vehicles[i]['position'][0] = v_x
                self.ve_l[i] = self.vehicles[i]['position'][0]
            else:
                self.vehicles[i]['position'][0] = -self.width/2
                self.ve_l[i] = -self.width/2
            i += 1

    def renew_channel(self):
        self.V2I_path_loss = np.zeros(len(self.vehicles))
        self.large_scaled = np.zeros(len(self.vehicles))
        self.small_scaled = np.zeros(len(self.vehicles))
        self.V2I_shadowing = self.channel.get_shadowing(self.delt_distance, self.V2I_shadowing)

        for i in range(len(self.vehicles)):
            self.V2I_path_loss[i] = self.channel.get_path_loss(self.vehicles[i]['position'])
            self.small_scaled[i] = self.channel.small_scale_path_loss(self.vehicles[i]['position'], self.h_path[i])
            self.h_path[i] = self.small_scaled[i]
        
        self.large_scaled = self.V2I_path_loss + self.V2I_shadowing
    
    def R_V2I(self):
        self.V2I_rate = []
        for i in range(len(self.vehicles)):
            self.g_channel[i] = self.channel.channel_gain(self.small_scaled[i], self.large_scaled[i])
            self.g_channel[i] = 10**(self.g_channel[i]/10)
            y_sinr = self.channel.SINR(self.g_channel[i], self.trans_power/1000, self.noise1)
            v2i_rate = self.band_width*np.log2(1 + y_sinr)
            self.V2I_rate.append(v2i_rate)    # transmission rate from V to relay_BS
    
    def time(self, f_ij):
        for i in range(len(self.vehicles)):
            task_size = self.task_size[i]
            upd_size = self.upd_size[i]
            f_i_TK = f_ij[i][0]
            c_i_tk = self.task_fc[i]
            f_i_DT = f_ij[i][1]
            c_i_DT = self.upd_fc[i]
            V2I_rate = self.V2I_rate
            t_tk = task_size*c_i_tk / V2I_rate[i] + task_size*c_i_tk / (f_i_TK*1000) 
            t_dt = upd_size*c_i_DT / V2I_rate[i] + (upd_size*c_i_DT / (f_i_DT*1000.0))*self.n_Veh 
            self.t_TK[i] = t_tk
            self.t_DT[i] = t_dt
    
    def act_for_training(self, actions):
        action_temp = actions.copy()
        reward = np.sum(self.reward(action_temp))
        return reward/self.n_Veh
    
    def reward(self, action):
        self.time(action)
        alpha = 0.5
        #omea1 = 0.1
        #omea2 = 0.1
        p1 = np.divide(self.t_DT, self.upd_t_limit)
        p2 = np.divide(self.t_TK, self.tk_t_limit)
        Utility = alpha * (1-np.log(1+np.divide(self.t_DT, self.upd_t_limit))) + (1 - alpha)*(1 - np.log(1+np.divide(self.t_TK, self.tk_t_limit)))
        if self.bs_max_fre - np.sum(action)>=0:
            #if np.mean(p1)<=1 and np.mean(p2)<=1:
                reward = Utility
            #else:
            #    reward = -Utility
        else:
            #if np.mean(p1)<=1 and np.mean(p2)<=1:
                reward = -Utility
            #else:
            #    reward = Utility
        return reward
        
    def make_new_game(self):
        self.vehicles = []
        self.add_new_vehicles_by_number(int(self.n_Veh))
        self.renew_channel()
        self.R_V2I()