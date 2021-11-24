import mmap
import functools
import ctypes
from ctypes import c_int32, c_float, c_wchar,c_bool, c_uint8, c_uint32
import time
import numpy as np
import struct
import pandas as pd
from threading import Thread
import os
from multiprocessing import shared_memory
from controller import VehiclePIDController
import subprocess
import matplotlib.pyplot as plt
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
                   'cg_height',
                   'slip_ratio',
                   'load',
                   'pressure',
                   'angular_velocity',
                   'wear',
                   'dirty_level',
                   'core_temperature',
                   'disc_temperature',
                   'slip',
                   'slip_angle_deg',
                   'nd_slip',
                   'wheel_look',
                   'abs_in_action',
                   'traction_control_in_action',
                   'lap_time_ms',
                   'best_lap_time_ms',
                   'drivetrain_torque',]


class AP_wheel(ctypes.Structure):
    _pack_ = 4
    _fields_ = [('wheel_position', c_float * 3),
        ('contact_point', c_float * 3),
        ('contact_normal', c_float * 3),
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
        ('wheel_look', c_float * 3),
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
    ]

class SimInfo:
    def __init__(self,file ="AcTools.CSP.ModuleLDM.AIProject.Car0.v0" ):

        self._acpmf_physics = mmap.mmap(0, ctypes.sizeof(SPageFilePhysics), file)
        self.physics = SPageFilePhysics.from_buffer(self._acpmf_physics)
        self.init_pos = self.physics.position
        self.init_dir = self.physics.look

    def read_states(self,info):
        states = dict.fromkeys(features_needed, 0)
        _struct = info.physics
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
            'slip_ratio', 
            'load',
            'pressure',
            'angular_velocity',
            'wear',
            'dirty_level',
            'core_temperature',
            'camber_rad',
            'disc_temperature',
            'slip',
            'slip_angle_deg', 
            'nd_slip',
            'wheel_look',]
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
        self._controls = mmap.mmap(0, ctypes.sizeof(SPageFileControls), file)
        self.controls =SPageFileControls.from_buffer(self._controls)
        self.controls.clutch=1
        self.treshold_up = 7500 
        self.treshold_dn = 5000
        self.changed_gear=False
        self.controls.autoclutch_on_change=True
        

    def init_file(self):
        p = subprocess.Popen('Drivemaster.AiProjectWriter.exe {}'.format(self.file))
        time.sleep(2)
        p.terminate()
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
        previous_gear= states['gear']
        if gas > 0:
            if states['gear'] == 1 :
                self.change_gear(0)
                
            if states['rpm'] > self.treshold_up and states['gear'] != 1 :
                self.change_gear(0) #increase

        if brake > 0:
            if states['rpm'] < self.treshold_dn and (states['gear'] not in [1,2]) :
                self.change_gear(1) #decrease
                
            if states['speedKmh'] < 1:
               if states['gear']== 0:
                    self.change_gear(0)
               if states['gear'] == 2: 
                   self.change_gear(1)

        self.update_control()

    def teleport(self,position= c_float * 3,dir=c_float * 3):
        self.controls.teleport_to = 2
        self.controls.teleport_pos = position
        self.controls.teleport_dir = dir
        self.update_control()
        time.sleep(0.5)

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
    def change_states(self):
        self.sim.disable_collisions = True

    def update_sim(self):
        new_controls =bytearray(self.sim)
        self._sim.seek(0)
        self._sim.write(new_controls)


def teleport(controller,control):
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
def PID_unit(states,df,info,control):
    controller = VehiclePIDController(states,control,df)
    
    #teleport(controller,control)
    i=0
    while states['best_lap_time_ms'] > 59400 or states['best_lap_time_ms'] == 0:
        states,_dict = info.read_states(info)
        
        input,steering =controller.run_step(states)

        i+=1
    control.change_controls(states,0,0)    
def main_multiple(nbr_car):
    sim =SimStates()
    sim.change_states()
    sim.update_sim()
    info_list = []
    control_list = []
    states_list = []
        

def main_unit():
    info = SimInfo()
    control= SimControl()
    states,_dict = info.read_states(info)
    df = pd.read_csv("Data/Kevin/dynamic.csv",converters={'position': pd.eval})
    df['position'] =df[["WorldPosition_X","WorldPosition_Y","WorldPosition_Z"]].apply(lambda r: np.array(r), axis=1)
    PID_unit(states,df,info,control)
main_multiple(1)
main_unit()   
    
    
   
    
