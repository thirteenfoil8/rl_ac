import mmap
import functools
import ctypes
from ctypes import c_int32, c_float, c_wchar,c_bool, c_uint8, c_uint32
import time
import numpy as np
import struct
import pandas as pd
import ray
import os
from controller import VehiclePIDController
import multiprocessing
import subprocess
import matplotlib.pyplot as plt
from lidar import init_track_data,compute_lidar_distances
features_needed = ['gas',
                   'brake',
                   'gear',
                   'rpm',
                   'steer',
                   'speedKmh',
                   'velocity',
                   'acceleration', 
                   'look',
                   'up',
                   'position',
                   'local_velocity',
                   'local_angular_velocity',
                   'slip_ratio',
                   'angular_velocity',
                   'slip',
                   'slip_angle_deg',
                   'nd_slip',
                   'lap_time_ms',
                   'best_lap_time_ms',
                   'spline_position']


class AP_wheel(ctypes.Structure):
    _pack_ = 4
    _fields_ = [('wheel_position', c_float * 3),
        ('contact_point', c_float * 3),
        ('contact_normal', c_float * 3),
        ('wheel_look', c_float * 3),
        ('side', c_float * 3),
        ('wheel_velocity', c_float * 3),
        ('slip_ratio', c_float ), 
        ('load', c_float ),
        ('pressure', c_float ),
        ('angular_velocity', c_float ),
        ('wear', c_float ),
        ('dirty_level', c_float ),
        ('core_temperature', c_float ),
        ('camber_rad', c_float ),
        ('disc_temperature', c_float ),
        ('slip', c_float ),
        ('slip_angle_deg', c_float ),
        ('nd_slip', c_float ),
        ('sideways_velocity', c_float),
        ]
class SPageFilePhysics(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ('packetId', c_int32), 
        ('gas', c_float),  
        ('brake', c_float), 
        ('clutch', c_float), 
        ('steer', c_float), 
        ('handbrake', c_float),
        ('fuel', c_float),
        ('gear', c_int32), 
        ('rpm', c_float), 
        ('speedKmh', c_float), 
        ('velocity', c_float * 3), 
        ('acceleration', c_float * 3), 
        ('look', c_float * 3), 
        ('up', c_float * 3), 
        ('position', c_float * 3), 
        ('local_velocity', c_float * 3),
        ('local_angular_velocity', c_float * 3),
        ('cg_height', c_float),
        ('car_damage', c_float*5),
        #Wheels states
        ("wheels", AP_wheel * 4),

        ('turbo_boost', c_float),
        ('final_ff', c_float),
        ('final_pure_ff', c_float),
        ('pit_limiter', c_bool),
        ('abs_in_action', c_bool),
        ('traction_control_in_action', c_bool),
        ('lap_time_ms', c_uint32),
        ('best_lap_time_ms', c_uint32),
        ('drivetrain_torque', c_float),
        ('spline_position', c_float),
        ('collision_depth', c_float),
        ('collision_counter', c_uint32)
        
    ]

controls_needed = ['gas',
                   'break',
                   'steer',
                   'gear_up']
class SPageFileControls(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ('gas', c_float),  
        ('brake', c_float), 
        ('clutch', c_float), 
        ('steer', c_float), 
        ('handbrake', c_float),
        ('gear_up',  c_bool),
        ('gear_dn',  c_bool), 
        ('drs',  c_bool), 
        ('kers',  c_bool), 
        ('brake_balance_up',  c_bool), 
        ('brake_balance_dn',  c_bool), 
        ('abs_up',  c_bool), 
        ('abs_dn',  c_bool), 
        ('tc_up',  c_bool), 
        ('tc_dn',  c_bool),
        ('turbo_up',  c_bool),
        ('turbo_dn',  c_bool),
        ('engine_brake_up',  c_bool),
        ("engine_brake_dn",  c_bool),
        ('mguk_delivery_up',  c_bool),
        ('mguk_delivery_dn',  c_bool),
        ('mguk_recovery_up',  c_bool),
        ('mguk_recovery_dn', c_bool),
        ('mguh_mode', c_uint8), 
        ('headlights', c_bool),
        ('teleport_to', c_uint8),
        ('autoclutch_on_start', c_bool),
        ('autoclutch_on_change', c_bool),
        ('autoblip_active', c_bool),
        ('teleport_pos', c_float * 3),
        ('teleport_dir', c_float * 3),
        
    ]
class SPageSimControls(ctypes.Structure):
    _fields_ = [
        ('pause', c_bool),  
        ('restart_session', c_bool), 
        ('disable_collisions', c_bool), 
        ('delay_ms', c_uint8), 
    ]

class SimInfo:
    def __init__(self,file ="AcTools.CSP.ModuleLDM.AIProject.Car0.v0" ):

        self._acpmf_physics = mmap.mmap(0, ctypes.sizeof(SPageFilePhysics), file)
        self.physics = SPageFilePhysics.from_buffer(self._acpmf_physics)
        self.init_pos = self.physics.position
        self.init_dir = self.physics.look
        self.sideleft_xy,self.sideright_xy,self.centerline_xy,self.normal_xyz,self.track_data = init_track_data()

    def read_states(self):
        states = dict.fromkeys(features_needed, 0)
        _struct = self.physics
        _dict= self.getdict(_struct)
        self.unified_wheels(_dict)
        for key in _dict.keys():
            if key in states.keys():
                states[key] = _dict[key]
        return states,_dict
    def getdict(self,struct):
        result = {}
        for field, _ in struct._fields_:
             value = getattr(struct, field)
             # if the type is not a primitive and it evaluates to False ...
             if hasattr(value, "_length_") and hasattr(value, "_type_"):
                 # Probably an array
                 value = np.ctypeslib.as_array(value).tolist()
             elif hasattr(value, "_fields_"):
                 # Probably another struct
                 value = getdict(value)
             result[field] = value
        return result
    def unified_wheels(self,_dict):
        keys= ['wheel_position',
        'contact_point',
        'contact_normal',
        'wheel_look',
        'side',
        'wheel_velocity',
        'slip_ratio', 
        'load',
        'pressure',
        'angular_velocity',
        'wear',
        'dirty_level', #delete
        'core_temperature',
        'camber_rad',
        'disc_temperature',
        'slip',
        'slip_angle_deg',
        'nd_slip',
        'sideways_velocity',]
        states = dict.fromkeys(keys, 0)
        for j in range(len(_dict['wheels'][0])):
            values = []
            for i in range(len(_dict['wheels'])):
                values.append(_dict['wheels'][i][j])
            values = np.array(values)
            states[keys[j]] = values
        _dict.update(states)
    #def close(self):
    #    self._acpmf_physics.close()

    #def __del__(self):
    #    self.close()

class SimControl:
    def __init__(self,file = "AcTools.CSP.ModuleLDM.AIProject.CarControls0.v0"):
        self.file = file
        self.init_file()
        print('file created')
        self._controls = mmap.mmap(0, ctypes.sizeof(SPageFileControls), file)
        self.controls =SPageFileControls.from_buffer(self._controls)
        self.controls.clutch=1
        self.last_gear = 1
        self.treshold_up = 7500 
        self.treshold_dn = 4600
        self.changed_gear=False
        self.controls.autoclutch_on_change=True
        

    def init_file(self):
        p = subprocess.Popen('Drivemaster.AiProjectWriter.exe {}'.format(self.file))
        time.sleep(2)
        p.terminate()
        p.kill()
    def update_control(self):
        new_controls =bytearray(self.controls)
        self._controls.seek(0)
        self._controls.write(new_controls)

    def change_controls(self,states,input,steer=0):
        self.controls.clutch=1
        gas = 0
        brake = 0 
        if input > 0:
            gas = input
        elif input < 0:
            brake = np.abs(input)
        elif input == 0:
            gas = 0
            brake = 0
        self.controls.gas = gas
        self.controls.brake = brake
        self.controls.steer = steer
        self.controls.gear_up = 0
        self.controls.teleport_to = 0
        self.controls.gear_dn = 0
        if gas > 0:
            if states['gear'] == 1  and states['rpm']> 1500:
                self.change_gear(0)
                
            if states['rpm'] > self.treshold_up and states['gear'] != 1 :
                self.change_gear(0) #increase

        if brake > 0:
            if states['rpm'] < self.treshold_dn and (states['gear'] not in [1,2]) :
                self.change_gear(1) #decrease
                
            #if states['speedKmh'] < 1:
            #   if states['gear']== 0:
            #        self.change_gear(0)
            #   if states['gear'] == 2: 
            #       self.change_gear(1)

        self.update_control()

    def teleport(self,position= c_float * 3,dir=c_float * 3):
        self.controls.teleport_to = 2
        self.controls.teleport_pos = position
        self.controls.teleport_dir = dir
        self.update_control()
        print('Teleportation')

    def change_gear(self,state):
        if state == 0:
            self.controls.gear_up= 1
        else:
            self.controls.gear_dn= 1
            
        self.changed_gear=True

    #def close(self):
    #    self._controls.close()

    #def __del__(self):
    #    self.close()
class SimStates:
    def __init__(self,file = "AcTools.CSP.ModuleLDM.AIProject.SimState.v0"):
        self.file = file
        self.init_file()
        self._sim = mmap.mmap(0, ctypes.sizeof(SPageSimControls), self.file)
        self.sim =SPageSimControls.from_buffer(self._sim)

    def init_file(self):
        p =subprocess.Popen('Drivemaster.AiProjectWriter.exe {}'.format(self.file))
        time.sleep(2)
        p.terminate()
        p.kill()
    def change_states(self):
        self.sim.disable_collisions = True

    def update_sim(self):
        new_controls =bytearray(self.sim)
        self._sim.seek(0)
        self._sim.write(new_controls)
    def pause(self):
        self.sim.pause = True
        self.update_sim()
    def play(self):
        self.sim.pause = False
        self.update_sim()
    def speed(self,value=3):
        self.sim.delay_ms = value
        self.update_sim()
    def reset(self):
        self.sim.restart_session=True
        self.update_sim()
        time.sleep(0.1)
        self.sim.restart_session=False
        self.update_sim()


def teleport(controller,control,info):
    states,_ = info.read_states()
    waypoint,_,_ =controller.find_nearest()
    waypoint= waypoint.tolist()
    FloatArr = c_float * 3

    _waypoint = FloatArr()
    _waypoint[0] = waypoint[0]
    _waypoint[1] = waypoint[1]
    _waypoint[2] = waypoint[2]
    
    look = states['look']
    _look = FloatArr()
    _look[0] = -look[0]
    _look[1] = look[1]
    _look[2] = look[2]
    
    control.teleport(_waypoint,_look)


def PID_unit(states,df,info,control,sim):
    controller = VehiclePIDController(info,states,control,df,sim)
    #sim.speed()
    #teleport(controller,control,info)
    while controller._vehicle['best_lap_time_ms'] > 59400 or controller._vehicle['best_lap_time_ms'] == 0:
        #sim.pause()
        #_time = time.time()
        #while time.time()- _time <0.3:
        #    input,steering =controller.run_step()
        #sim.play()

        distances = compute_lidar_distances(states["look"],states["position"],info.sideleft_xy,info.sideright_xy)
        
        controller.run_step()
        

    control.change_controls(states,0,0) 

#@ray.remote
def run_pid(controller):
    while controller._vehicle['best_lap_time_ms'] > 59400 or controller._vehicle['best_lap_time_ms'] == 0:
        input,steering =controller.run_step()


def PID_multiple(states_list,df,info_list,control_list,controller_list,sim,nbr_car):
    best_lap_time = []
    
    for i in range(nbr_car):
        best_lap_time.append(states_list[i]['best_lap_time_ms'])
    #p1 = multiprocessing.Process(target=run_pid, args=(controller_list[0],))
    #p2 = multiprocessing.Process(target=run_pid, args=(controller_list[1],))

    #p1.start()
    #p2.start()

    #ray.register_class(VehiclePIDController)
    #ray.init()
    #ray.get(run_pid.remote(controller_list[0]),run_pid.remote(controller_list[1]))
    best_lap_time = []
    
    for i in range(nbr_car):
        best_lap_time.append(states_list[i]['best_lap_time_ms'])
    while min(best_lap_time) > 59400 or min(best_lap_time) == 0:
        for i in range(nbr_car):
            controller_list[i].run_step()
    for i in range(nbr_car):
        control_list[i].change_controls(states,0,0) 
def create_cars(nbr_car=1):
    template_name = "AcTools.CSP.ModuleLDM.AIProject.CarControls0.v0"
    final_control =""
    for i in range(nbr_car):
        final_control = final_control  +template_name[:43]+str(i) +template_name[44:]
    p =subprocess.Popen('Drivemaster.AiProjectWriter.exe {}'.format(final_control))
    time.sleep(2)
    p.terminate()
    p.kill()
def main_multiple(nbr_car):
    create_cars(nbr_car)
    sim =SimStates()
    sim.change_states()
    sim.update_sim()
    #sim.speed()
    template_name = ["AcTools.CSP.ModuleLDM.AIProject.Car0.v0","AcTools.CSP.ModuleLDM.AIProject.CarControls0.v0"]
    info_list = []
    control_list = []
    controller_list = []
    states_list = []

    df = pd.read_csv("Data/Kevin/dynamic.csv",converters={'position': pd.eval})
    df['position'] =df[["WorldPosition_X","WorldPosition_Y","WorldPosition_Z"]].apply(lambda r: np.array(r), axis=1)
    for i in range(nbr_car):
        control_string = template_name[1]
        info_string = template_name[0]
        final_control = control_string[:43]+str(i) +control_string[44:]
        final_info = info_string[:35]+str(i) +info_string[36:]
        info_list.append(SimInfo(final_info))
        control_list.append(SimControl(final_control))
        
    for i in range(nbr_car):
        states,_ =  info_list[i].read_states()
        states_list.append(states)
        controller_list.append(VehiclePIDController(info_list[i],states_list[i],control_list[i],df))

    PID_multiple(states_list,df,info_list,control_list,controller_list,sim,nbr_car)
        
def print_size_states(states):
    count = 0
    for i in states.values():
        if isinstance(i,list):
            count+= len(i)
        else:
            count+=1
    print(count)
def states_to_1d_vector(states):
    new_list = list(states.values())
    final_list = []
    for i in new_list:
        if isinstance(i,list):
            for j in i:
                final_list.append(j)
        elif isinstance(i,np.ndarray):
            for j in i:
                if isinstance(j,np.ndarray):
                    for k in j:
                        final_list.append(k)
                else:
                    final_list.append(j)
        else:
            final_list.append(i)
    #for i in final_list:
    #    if isinstance(i, )
    arr = np.array(final_list)
    arr = arr.astype(np.float32)
    return arr

def main_unit():
    info = SimInfo()
    control= SimControl()
    sim =SimStates()
    sim.change_states()
    sim.update_sim()
    states,_dict = info.read_states()
    states_to_1d_vector(states)
    df = pd.read_csv("Data/Kevin/dynamic.csv",converters={'position': pd.eval})
    df['position'] =df[["WorldPosition_X","WorldPosition_Y","WorldPosition_Z"]].apply(lambda r: np.array(r), axis=1)
    PID_unit(states,df,info,control,sim)


#main_multiple(2)
#main_unit()   
    
    
   
    
