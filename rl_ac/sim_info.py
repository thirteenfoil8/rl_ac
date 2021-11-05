import mmap
import functools
import ctypes
from ctypes import c_int32, c_float, c_wchar,c_bool, c_uint8, c_uint32
import time
import numpy as np
import struct

AC_STATUS = c_int32
AC_OFF = 0
AC_REPLAY = 1
AC_LIVE = 2
AC_PAUSE = 3
AC_SESSION_TYPE = c_int32
AC_UNKNOWN = -1
AC_PRACTICE = 0
AC_QUALIFY = 1
AC_RACE = 2
AC_HOTLAP = 3
AC_TIME_ATTACK = 4
AC_DRIFT = 5
AC_DRAG = 6
AC_FLAG_TYPE = c_int32
AC_NO_FLAG = 0
AC_BLUE_FLAG = 1
AC_YELLOW_FLAG = 2
AC_BLACK_FLAG = 3
AC_WHITE_FLAG = 4
AC_CHECKERED_FLAG = 5
AC_PENALTY_FLAG = 6
pack =4
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
                   'wheel_position',
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

class SimInfo:
    def __init__(self):

        self._acpmf_physics = mmap.mmap(0, ctypes.sizeof(SPageFilePhysics), "AcTools.CSP.ModuleLDM.AIProject.Car0.v0")
        self.physics = SPageFilePhysics.from_buffer(self._acpmf_physics)

    def close(self):
        self._acpmf_physics.close()

    def __del__(self):
        self.close()

class SimControl:
    def __init__(self):


        self._controls = mmap.mmap(0, ctypes.sizeof(SPageFileControls), "AcTools.CSP.ModuleLDM.AIProject.CarControls0.v0")
        self.controls =SPageFileControls.from_buffer(self._controls)
    def change_controls(self):
        pass

    def close(self):
        self._controls.close()

    def __del__(self):
        self.close()

def getdict(struct):
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
def unified_wheels(_dict):
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

info = SimInfo()  
control=SimControl()

def read_states(info_flag = True):
    if info_flag:
        states = dict.fromkeys(features_needed, 0)
        _struct = info.physics
        _dict= getdict(_struct)
        unified_wheels(_dict)
    else:
        states = dict.fromkeys(controls_needed, 0)
        _struct = control.controls
        _dict= getdict(_struct)
    for key in _dict.keys():
        if key in states.keys():
            states[key] = _dict[key]
    return states,_dict


if __name__ == '__main__':
    while True:
        states,_ = read_states(info_flag = False)
        for i in states:
            print(i,states[i])


