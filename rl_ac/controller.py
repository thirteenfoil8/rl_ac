from collections import deque
import math
import sys
import numpy as np
import pandas as pd
from scipy import spatial
import time
from numpy.core.umath_tests import inner1d
from numpy import sin, cos, tan, arctan, pi
# tune more in bang bang method !!!!
class VehiclePIDController():
    """
    VehiclePIDController is the combination of two PID controllers (lateral and longitudinal) to perform the
    low level control a vehicle from client side
    """

    def __init__(self, states,control,df, args_lateral=None, args_longitudinal=None):
        """
        :param vehicle: actor to apply to local planner logic onto
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
            
            #using ziegler-nichols, We have Ku = 0.3 Tu = 0.8
             #Then K_P =0.9 ,K_D = 0.075, K_I = 0.3
            args_lateral = {'K_P':3, 'K_D': 0.5, 'K_I':  0}
        if not args_longitudinal:
             #using ziegler-nichols, We have Ku = 1 Tu = 1
             #Then K_P =0.6 ,K_D = 0.075, K_I = 0.3
            args_longitudinal = {'K_P': 0.3, 'K_D': 0.00, 'K_I': 0.000}

        self._vehicle = states
        self._controls = control
        self.speed_multiplier = 1.1
        self.df =df
        self.target = [0,0,0]
        self._lon_controller = PIDLongitudinalController(self._vehicle, **args_longitudinal)
        self._lat_controller = PIDLateralController(self._vehicle, **args_lateral)

    def run_step(self, states):
        """
        Execute one step of control invoking both lateral and longitudinal PID controllers to reach a target waypoint
        at a given target_speed.
        :param target_speed: desired vehicle speed
        :param waypoint: target location encoded as a waypoint
        :return: distance (in meters) to the waypoint
        """
        self._vehicle = states
        
        
        
        
        waypoint,speed,yaw=self.find_nearest()
        
        speed= speed * self.speed_multiplier
        input = self._lon_controller.run_step(states,speed)
        steering = self._lat_controller.run_step(states,waypoint,yaw)
        control = self._controls
        control.change_controls(self._vehicle, input,steering)

        return input,steering

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


    def in_look(self,deltas):
        look = self._vehicle['look']
        
        deltas_normed=deltas/np.sqrt(inner1d(np.stack(deltas,axis = 0),np.stack(deltas,axis = 0)))
        
        angle = deltas_normed.apply(lambda r : self.angle_clockwise(look, r))
        angle_selected= angle[(angle < np.pi/4) | (angle > 7*np.pi/4)]
        return angle_selected.index
    def horizon_selection(self,norms):
        horizon = 48
        minimum_norm = norms.argmin()
        turn= minimum_norm+horizon/6
        self.speed_multiplier = 1.08
        # TODO Try PI + full steering available
        self._lat_controller._K_P = 0.5
        self._lat_controller._K_D = 0.1
        self._lat_controller._K_I = 0
        self._lat_controller.max_steer = 0.2
        if turn>= self.df.shape[0]:
            turn = turn -self.df.shape[0]
        if np.abs(self.df.LocalAngularVelocity_Y[turn]) > 0.03:
            minimum_norm =  turn
            self._lat_controller.max_steer = 1
            self.speed_multiplier = 1
            self._lat_controller._K_P = 2
            self._lat_controller._K_D = 0.5
            self._lat_controller._K_I = 0
            if np.abs(self.df.LocalAngularVelocity_Y[turn]) > 0.15:
                self.speed_multiplier = 1
                self._lat_controller._K_P = 4
                self._lat_controller._K_D = 0.4
        elif minimum_norm+horizon >= self.df.shape[0]:
            minimum_norm =  minimum_norm+horizon-self.df.shape[0]
        else:
            minimum_norm = minimum_norm + horizon
        return minimum_norm
    def find_nearest(self):
        x,y,z = self._vehicle['position']
        node = np.array([x,y,z])
        nodes = self.df.position
        deltas = self.df.position.apply(lambda r: r-node)
        norms =np.sqrt(inner1d(np.stack(deltas,axis = 0),np.stack(deltas,axis = 0)))
        minimum_norm =self.horizon_selection(norms)
        return nodes[minimum_norm],self.df.Speed[minimum_norm],self.df.LocalAngularVelocity_Y[minimum_norm]
    
    


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
        return np.clip((self._K_P * _e) + (self._K_D * _de / self._dt) + (self._K_I * _ie * self._dt), -0.9, 1.0)

def wrap2pi(angle):
    wrapped_angle = np.remainder(angle, 2*pi)
    if wrapped_angle > pi:
        wrapped2pi = -2*pi + wrapped_angle
    else:
        wrapped2pi = wrapped_angle
    return wrapped2pi

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

    def path_following_controller(self,x, y, yaw, v, x_target, y_target, yaw_target):
        ''' 
        E-Racing's Steering Controller
    
        INPUTS: vehicle states, position and steering targets.
        OUTPUTS: steering angle.
        '''
        # define controller hyperparameters
        l = 1
        g = v
        k = v
        Lb = 1
        b = 3*Lb/tan(1)
    
        # calculate sigma
        s = (y - y_target)*cos(yaw_target) - (x - x_target)*sin(yaw_target) + \
            b*wrap2pi(yaw-yaw_target)         
    
        # calculate the steering angle
        yaw_dot = -(Lb/b)*sin(yaw-yaw_target) - ((k*Lb)/(b*v))*(s/(l + abs(s)))
        delta = arctan(yaw_dot)
    
        return delta

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