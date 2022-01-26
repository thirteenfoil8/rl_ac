from collections import deque
import math
import sys
import numpy as np
import pandas as pd
from scipy import spatial
import time
from numpy.core.umath_tests import inner1d
from numpy import sin, cos, tan, arctan, pi
import pandas as pd

class VehiclePIDController():
    """
    VehiclePIDController is the combination of two PID controllers (lateral and longitudinal) to perform the
    low level control a vehicle from client side
    """

    def __init__(self,info, states,control,df,sim, args_lateral=None, args_longitudinal=None):
        """
        :param info: SimInfo class
        :param states: states of the car (dict)
        :param control: SimControl class
        :param df: Reference pandas DF
        :param sim: SimStates class
        :param args_lateral: dictionary of arguments to set the lateral PID controller using the following semantics:
                             K_P -- Proportional term
                             K_D -- Differential term
                             K_I -- Integral term
        :param args_longitudinal: dictionary of arguments to set the longitudinal PID controller using the following
        semantics:
                             K_P -- Proportional term
                             K_D -- Differential term
                             K_I -- Integral term
        """
        if not args_lateral:
            args_lateral = {'K_P':3, 'K_D': 0.5, 'K_I':  0}
        if not args_longitudinal:
            args_longitudinal = {'K_P': 0.3, 'K_D': 0, 'K_I': 10}
        self.info = info
        self._vehicle = states
        self.sim =sim
        self.dict = {}
        self._controls = control
        self.speed_multiplier = 1
        self.df =df
        self.data_flag = True
        self.data = []
        self.target = [0,0,0]
        self.last_lap_time_ms = 0
        self._lon_controller = PIDLongitudinalController(self._vehicle, **args_longitudinal)
        self._lat_controller = PIDLateralController(self._vehicle, **args_lateral)
    def __reduce__(self):
        deserializer = VehiclePIDController
        serialized_data = (self.info,self._vehicle,self._controls,self.df,self._lon_controller,self._lat_controller,)
        return deserializer, serialized_data

    def run_step(self):
        """
        Execute one step of control invoking both lateral and longitudinal PID controllers to reach a target waypoint
        at a given target_speed.

        Return:
            input (float): action gas/brake
            steering (float): action steering wheel
        """
        #
        self._vehicle,_ = self.info.read_states()
        waypoint,speed,yaw=self.find_nearest()
        self.data_collecting()
        
        speed= speed * self.speed_multiplier
        input = self._lon_controller.run_step(self._vehicle,speed)
        steering = self._lat_controller.run_step(self._vehicle,waypoint,yaw)
        control = self._controls
        return input,steering

    def data_collecting(self):
        
        if self._vehicle['best_lap_time_ms'] > 0 and self.data_flag:
            df = pd.DataFrame(self.data)
            self.sim.pause()
            df.to_excel("test.xlsx",index=False)
            self.sim.play()
            self.data_flag = False
        if self._vehicle['lap_time_ms'] < self.last_lap_time_ms:
            self.data=[]
            print('reset')
        self.last_lap_time_ms = self._vehicle['lap_time_ms']
        self.data.append(self._vehicle)
        


    def next_target(self):
        waypoint,speed= self.find_nearest()
        return waypoint,speed
    def angle_clockwise(self,A,B):
        dot = np.dot(A,B)
        inner = np.arccos(dot)
        
        det = A[0]*B[2]-A[2]*B[0]
        if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
            return inner
        else: # if the det > 0 then A is immediately clockwise of B
            return 2*np.pi-inner

    def horizon_selection(self,norms):
        """
        Function that gives the next horizon index of the reference (50 meter in front of the car)

        Returns:
            minimum_norm (float): next position index 
            gas_norm (float): next gas index 
            
        """
        horizon = 48
        horizon_gas= 10
        minimum_norm = norms.argmin()
        gas_norm = norms.argmin()
        turn= minimum_norm+horizon/8
        self.speed_multiplier = 1.01
        self._lat_controller._K_P = 0.5
        self._lat_controller._K_D = 0.1
        self._lat_controller._K_I = 10
        self._lat_controller.max_steer = 1
        if turn>= self.df.shape[0]:
            turn = turn -self.df.shape[0]
        #Small curve PID hyperparameter update
        if np.abs(self.df.LocalAngularVelocity_Y[turn]) > 0.03:
            minimum_norm =  turn
            self._lat_controller.max_steer = 1
            self.speed_multiplier = 1.00
            self._lat_controller._K_P = 2
            self._lat_controller._K_D = 0.5
            self._lat_controller._K_I = 0
            #Sharp curve PID hyperparameter update
            if np.abs(self.df.LocalAngularVelocity_Y[turn]) > 0.10:
                self.speed_multiplier = 1.00
                self._lat_controller._K_P = 3
                self._lat_controller._K_D = 0.4
        elif minimum_norm+horizon >= self.df.shape[0]:
            minimum_norm =  minimum_norm+horizon-self.df.shape[0]
        else:
            minimum_norm = minimum_norm + horizon
        if gas_norm +horizon_gas>= self.df.shape[0]:
            gas_norm = gas_norm+horizon_gas -self.df.shape[0]
        else: 
            gas_norm = gas_norm+horizon_gas
        return minimum_norm,gas_norm

    def find_nearest(self):
        """Use the horizon selection output to find the related values (based on idex)
            Returns:
            nodes[minimum_norm] (np.array(3* np.float32)): Next position to target
            self.df.Speed[gas_norm] (np.float32) : Next speed to achieve
            self.df.LocalAngularVelocity_Y[minimum_norm] (float32): Next angular velocity (curve sharpness)

        """
        x,y,z = self._vehicle['position']
        node = np.array([x,y,z])
        nodes = self.df.position
        deltas = self.df.position.apply(lambda r: r-node)
        norms =np.sqrt(inner1d(np.stack(deltas,axis = 0),np.stack(deltas,axis = 0)))
        minimum_norm,gas_norm =self.horizon_selection(norms)
        return nodes[minimum_norm],self.df.Speed[gas_norm],self.df.LocalAngularVelocity_Y[minimum_norm]
    
    


class PIDLongitudinalController():
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """

    def __init__(self, vehicle, K_P=1, K_D=0, K_I=0, dt=0.02):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param K_P: Proportional term
        :param K_D: Differential term
        :param K_I: Integral term
        :param dt: time differential in seconds
        """
        #KU = 0.8
        self._vehicle = vehicle
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        self._dt = dt
        self._e_buffer = deque(maxlen=30)
    def __reduce__(self):
        deserializer = PIDLongitudinalController
        serialized_data = (self._vehicle,)
        return deserializer, serialized_data

    def run_step(self,states, target_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.
        :param target_speed: target speed in Km/h
        :return: throttle control in the range [-1, 1]
        """
        self._vehicle = states
        current_speed = self._vehicle['speedKmh']

        if debug:
            print('Current speed = {}'.format(current_speed))

        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle of the vehicle based on the PID equations
        :param target_speed:  target speed in Km/h
        :param current_speed: current speed of the vehicle in Km/h
        :return: throttle control in the range [0, 1]
        """
        _e = (target_speed - current_speed)
        self._e_buffer.append(_e)

        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0
        return np.clip((self._K_P * _e) + (self._K_D * _de / self._dt) + (self._K_I * _ie * self._dt), -1, 1.0)


class PIDLateralController():
    """
    PIDLateralController implements lateral control using a PID.
    """

    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.001,current_time = None):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param K_P: Proportional term
        :param K_D: Differential term
        :param K_I: Integral term
        :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        self.max_steer = 1
        self.pterm= 0
        self.iterm = 0
        self.dterm = 0
        self._dt = dt
        self.last_error = 0
        self.sample_time = 0.00
        self.windup_guard = 100.0
        self.current_time = current_time if current_time is not None else time.time()
        self.last_time = self.current_time
    def __reduce__(self):
        deserializer = PIDLongitudinalController
        serialized_data = (self._vehicle,)
        return deserializer, serialized_data

    def run_step(self,states, waypoint,yaw):
        """
        Execute one step of lateral control to steer the vehicle towards a certain waypoin.
        :param waypoint: target waypoint
        :return: steering control in the range [-1, 1] where:
            -1 represent maximum steering to left
            +1 maximum steering to right
        """
        PID = True
        self._vehicle = states
        if PID:
            return self._pid_control(waypoint)
        else:
            x =self._vehicle['position'][0]
            y =self._vehicle['position'][2]
            yaw = self._vehicle['local_angular_velocity'][2]
            v = self._vehicle['speedKmh']
            x_target = waypoint[0]
            y_target = waypoint[2]
            yaw_target = yaw
            return self.path_following_controller(x, y, yaw, v, x_target, y_target, yaw_target)

    def angle_clockwise(self,A,B):
        dot = np.dot(A,B)
        inner = np.arccos(dot)
        
        det = A[0]*B[2]-A[2]*B[0]
        if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
            return inner
        else: # if the det > 0 then A is immediately clockwise of B
            return -inner


    def _pid_control(self, waypoint,current_time=None):
        """
        Estimate the steering angle of the vehicle based on the PID equations
        :param waypoint: target waypoint
        :param vehicle_transform: current transform of the vehicle
        :return: steering control in the range [-1, 1]
        """
        v_begin = np.array(self._vehicle['position'])
        look = np.array(self._vehicle['look'])
        ideal = np.array((waypoint-v_begin)/( np.sqrt(inner1d(waypoint-v_begin,waypoint-v_begin))))
        #pseudo code
        error = self.angle_clockwise(look,ideal)
        if error < np.pi/4 or error> -np.pi/4:
            self.current_time = current_time if current_time is not None else time.time()
            delta_time = self.current_time - self.last_time
            delta_error = error -self.last_error
            #if (delta_time >= self.sample_time):
            self.pterm = self._K_P *error
            self.iterm = error * delta_time
        
            if self.iterm < - self.windup_guard:
                self.iterm = - self.windup_guard
            if self.iterm > self.windup_guard:
                self.iterm = self.windup_guard
            self.dterm = 0
            if delta_time > 0:
                self.dterm = delta_error /  delta_time
            self.last_time = self.current_time
            self.last_error = error
            pid_term = -self.pterm - (self._K_I* self.iterm) - (self._K_D * self.dterm)
            return np.clip(pid_term, -self.max_steer, self.max_steer)
        else : 
            return 0