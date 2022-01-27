import gym
from gym.spaces import Box
import numpy as np
import sim_info
from lidar import init_track_data,compute_lidar_distances
from gym.utils import seeding
from numpy.core.umath_tests import inner1d
import time
import random
import os
import pandas as pd
import math
from ctypes import c_float
from controller import VehiclePIDController
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter
from ray.rllib.models.preprocessors import get_preprocessor
import ray._private.utils

class ClipAction(gym.core.ActionWrapper):
    r"""Clip the continuous action within the valid bound."""

    def __init__(self, env):
        super().__init__(env)
        self.low = -1
        self.high = 1

    def action(self, action):
        
        if action[1] > self.last_action+0.04: # 9° 
            action[1] = self.last_action+0.04
        if action[1] < self.last_action-0.04:
            action[1] = self.last_action-0.04
        
        return np.array([action[0],np.clip(action[1], self.low, self.high)])

class AC_Env(gym.Env):
    """

    The main API methods that users of this class need to know are:

        step
        reset
        render
        close
        seed
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards

    The methods are accessed publicly as "step", "reset", etc...
    """
    def __init__(self,env_config={}):
        self.initialise_data()
        self.done = False
        self.info = {}
        self.reward_total= 0
        self.reward_step = 0
        self.n_obj=0
        self.out_of_road = 0
        self.start_speed = 0
        self.time_check = 0
        self.last_action = 0
        self.collision = 0
        self.max_steps= 300
        self.spline_start= 0
        self.offset_spline = 0
        self.wrong_action = 0
        self.count= 0
        self.eps=0
        self.time_section = 0
        
        
        
        self._vehicle =  sim_info.SimInfo()
        self.init_pos = c_float * 3
        self.init_dir = c_float * 3
        
        self.controls = sim_info.SimControl()
        self.sim =sim_info.SimStates()
        self.line=sim_info.LineControl()
        self.states,self._dict = self._vehicle.read_states() 
        
        
        self.obs1d = []
        self.sideleft_xy,self.sideright_xy,self.centerline_xy,self.normal_xyz,self.track_data = init_track_data()
        self.df =  pd.read_csv("Data/Kevin/dynamic.csv",converters={'position': pd.eval})
        self.df['position'] =self.df[["WorldPosition_X","WorldPosition_Y","WorldPosition_Z"]].apply(lambda r: np.array(r), axis=1)
        self.controller = VehiclePIDController(self._vehicle,self.states,self.controls,self.df,self.sim)
        # Set these in ALL subclasses
        self.action_space = gym.spaces.Box(
            low=-1.,
            high=1.,
            shape=(2,),
            dtype=np.float32)

        self.observation_space = gym.spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(84,), #6 car states + 61 lidar + 14curvatures + angle+dist+ self.last_action= 84 obs in float 32 --> 804 bytes per step
                dtype=np.float32)
        self.seed()
        self.reset()
    def teleport_conversion(self):

        FloatArr = c_float * 3
        

        #random_splines = np.arange(0,1,0.05)
        #spline = random.choice(random_splines)
        #spline= 0.0
        #idx=(np.abs(self.track_data[:,3]-spline)).argmin()
        #_waypoint = FloatArr()
        #_waypoint[0] = self.track_data[idx,0]
        #_waypoint[1] = self.track_data[idx,1]
        #_waypoint[2] =  self.track_data[idx,2]
        #_look = FloatArr()
        #_look[0] = -self.track_data[idx,16]
        #_look[1] = self.track_data[idx,17]
        #_look[2] = -self.track_data[idx,18]

        _waypoint = FloatArr()
        _waypoint[0] = 322.2773742675781
        _waypoint[1] = 192.01971435546875
        _waypoint[2] =  84.85726165771484
        _look = FloatArr()
        _look[0] =-0.9996861815452576
        _look[1] = 0.02443912997841835
        _look[2] = -0.005505245644599199


        self.init_pos =  _waypoint
        self.init_dir =  _look
    def initialise_data(self):
        sim_info.create_cars()
    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.

        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.
self._dict
        Returns:
            observation (object): the initial observation.
        """
        self.done = False
        self.teleport_conversion()
        self.controls.teleport(self.init_pos,self.init_dir)
        time.sleep(0.5)
        self.states,self._dict = self._vehicle.read_states()
        
        self.controls.change_controls(self._dict,0,0)
        
        #self.sim.reset()
        self.update_observations()
        self.reward_total = 0
        self.count= 0
        self.eps +=1
        self.wrong_action = 0
        self.n_obj = 0
        self.out_of_road= 0
        self.reward_step= 0
        self.start_speed=0
        self.last_action=0
        self.last_action_one=0
        self.max_steps= 300
        self.time_section = time.time()
        self.time_check=time.time()
        self.collision =self._dict['collision_counter']
        self.spline_start=self.truncate(self.states['spline_position'],2)
        self.offset_spline = self.states['spline_position']
        return self.obs1d
    def move(self,action):
        self.controls.change_controls(self.states,action[0],action[1])

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # [-1:1]  18° 
        #batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
        #writer = JsonWriter(
        #    os.path.join(ray._private.utils.get_user_temp_dir(), "demo-out"))
        #prev_action = [self.last_action,self.last_action_one]
        #prev_reward = self.reward_step
        #prev_obs = self.obs1d
        
        
        
        if self.wrong_action >20:
            self.done = True
        #if not self.start_speed:
        #    while self.states['speedKmh'] < 10:
        #        self.states,self._dict = self._vehicle.read_states()
        #        self.controls.change_controls(self._dict,1,0)
        #    self.start_speed=1

        #if time.time()-self.time_check >30 and self.states['speedKmh']<2:
        #    self.done=True

        if self.count > self.max_steps:
            self.done=True
        #Workflow:
        #   send action to the car
        #   Update observations using RAM Data
        #   Update the reward
        #   Store data to the buffer
        if not self.done:
            self.move(action)
            self.update_observations()
            self.update_reward_naive()
            self.count+= 1
            self.last_action = action[1]

        obs = self.obs1d
        reward = self.reward_step
        done = self.done
        info = {}
        # ─── RESET ITERATION VARIABLES ───────────────────────────────────
        #batch_builder.add_values(
        #    t=self.count,
        #    eps_id=self.eps,
        #    agent_index=0,
        #    obs=prev_obs,
        #    actions=action,
        #    action_prob=1.0,  
        #    action_logp=0.0,
        #    rewards=reward,
        #    prev_actions=prev_action,
        #    prev_rewards=prev_reward,
        #    dones=done,
        #    infos=info,
        #    new_obs=obs)
        #if self.done:
        #    writer.write(batch_builder.build_and_reset())
        return [obs,reward ,done,info]

    def update_observations(self):

        self.out_of_road=0

        self.states,self._dict = self._vehicle.read_states()
        dist,angle,_ = self.find_nearest()
       
        if self._dict['collision_counter']> self.collision:
            self.out_of_road = 1
            self.done=True
        for i in self._dict['dirty_level']:
            if i != 0:
                self.out_of_road = 1
                self.wrong_action+=1

        self.obs1d = np.array([])
        self.obs ={k:self.states[k] for k in ('velocity','acceleration') if k in self.states}
        self.obs1d = sim_info.states_to_1d_vector(self.obs)
        self.obs1d = self.obs1d =np.append(self.obs1d,np.array([self.last_action]).astype(np.float32))
        
        #Compute the distance with the border of the track
        lidar,_,_= compute_lidar_distances(self.states["look"],self.states["position"],self.sideleft_xy,self.sideright_xy)
        
        # if len(lidar)< 61, then the car is out of track --> add 0 to have data consistency
        if lidar.shape[0]< 61:
            self.out_of_road= 1
            self.wrong_action +=1
            self.obs1d =np.append(self.obs1d,lidar)
            self.obs1d =np.append(self.obs1d,np.zeros(61-lidar.shape[0]).astype(np.float32))
            self.done=True
        
        else:
            self.obs1d =np.append(self.obs1d,lidar)
        self.obs1d = np.append(self.obs1d,self.find_curvature().astype(np.float32))
        self.obs1d = np.append(self.obs1d,np.array([dist,angle]).astype(np.float32))
        

    def update_reward(self):
        dist,angle,speed = self.find_nearest()
        #reward_speed = np.abs(self.states['spline_position']-self.spline_before)
        reward_speed = 1-(np.abs(self.states['speedKmh']-speed)/(speed+1))
        
        # If distance with centerline > 10m --> negative reward

        if dist > 10:
            reward_ai_pos = 0
        # Else, the closer to 0m , the higher the reward 
        else:
            dist_normed= dist/10
            reward_ai_pos = 1-dist_normed
        #If the angle between the look angle of the car and the AI angle > np.pi/4 --> negative reward
        #if angle > np.pi/9 or angle < -np.pi/9:
        #    reward_ai_angle = 0

        ##Add positiv reward for angle diff small
        #if np.abs(angle)<np.pi/36:
        #    reward_ai_angle =  (1-np.abs(angle/(np.pi/36))) 
        reward_ai_angle = np.cos(angle) 
        #Maybe Add Reward based on the angle of the steer related to the curvature of the track
       
            
        # Reward step = distance achieved reward + distance to centerline reward + angle with centerline angle reward. If out of road, the reward becomes negativ
        self.reward_step= (2*reward_speed+ reward_ai_pos + 3*reward_ai_angle)/6
        if self.states['speedKmh'] < 1:
            self.reward_step=0
        self.reward_total += self.reward_step
    def update_reward_naive(self):

        

        dist_goal_before,dist_goal_after= self.find_progression()
        time_in_section= time.time()-self.time_section
        time_reward = time_in_section/10
        if time_in_section/10 >0.6:
            time_reward += self.n_obj
        
        self.reward_step= self.n_obj+ (1- (dist_goal_after/(dist_goal_after+dist_goal_before)))*(1- time_reward)

        #spline_kevin = (np.abs(self.df.NormalizedSplinePosition-self.states['spline_position'])).argmin()

        #speed = self.df.Speed.iloc[[spline_kevin]].values[0]
        #x = 1-(np.abs(self.states['speedKmh']-speed)/(speed+1))

        #speed_reward = -0.1*np.exp(30*np.power((2*x-1),4)-32)+0.6*np.exp(np.power(x,8))-0.6
        #self.reward_step +=speed_reward
        if self.states['speedKmh'] < 1:
            self.reward_step=0
        if self.out_of_road:
            self.reward_step-=20


        self.reward_total += self.reward_step

    def find_progression(self):

        spline_goal_before =  self.truncate(self.states['spline_position'],2)
        spline_goal_after = spline_goal_before+0.01
        idx_goal_before = (np.abs(self.track_data[:,3]-spline_goal_before)).argmin()
        idx_goal_after = (np.abs(self.track_data[:,3]-spline_goal_after)).argmin()
        pos_goal_before = self.track_data[idx_goal_before,0:3]
        pos_goal_after = self.track_data[idx_goal_after,0:3]
        self.line.conversion(pos_goal_before,pos_goal_after)
        dist_goal_before = np.linalg.norm(self.states['position']-pos_goal_before)
        dist_goal_after = np.linalg.norm(pos_goal_after-self.states['position'])
        if spline_goal_before != self.spline_start  :
            self.n_obj += 1
            self.spline_start=spline_goal_before
            self.count = 0
            self.time_section=time.time()

        return dist_goal_before,dist_goal_after

    def find_curvature(self):
        idx=(np.abs(self.track_data[:,3]-self.states['spline_position'])).argmin()
        #Return the curvature radius on the current location
        curvatures = np.array([])
        for i in range(14):
            if idx + i >= len(self.track_data[:,11]):
                new_idx = (idx + i)-len(self.track_data[:,11])
                curvatures = np.append(curvatures,1/(self.track_data[new_idx,11]*self.track_data[new_idx,7]))
            else:
                curvatures = np.append(curvatures,1/(self.track_data[idx + i,11]*self.track_data[idx + i,7]))

        return curvatures
    def find_nearest(self):
        x = self.centerline_xy[:,0]
        y = self.centerline_xy[:,1]
        pos = np.array([x,y])
        x_car,z_car,y_car = self.states['position']
        deltas = pos - np.array([x_car,-y_car]).reshape((2,1))
        norms =np.sqrt(inner1d(np.stack(deltas,axis = 1),np.stack(deltas,axis = 1)))
        minimum_norm = norms.argmin()
        norm = norms.min()
        minimum_norm += 6
        if minimum_norm >= norms.shape[0]: 
            minimum_norm = minimum_norm - norms.shape[0]
        
        look = self.states['look']
        ideal = self.normal_xyz[minimum_norm]
        error = self.angle_clockwise(look,[ideal[0],ideal[1],ideal[2]])


        spline_kevin = (np.abs(self.df.NormalizedSplinePosition-self.states['spline_position'])).argmin()
        speed = self.df.Speed.iloc[[spline_kevin]]

        #goal_spline =  self.truncate(self.states['spline_position'],1)
        #goal_position = 


        return norm,error,speed.values[0]

    def angle_clockwise(self,A,B):
        dot = np.dot(A,B)
        inner = np.arccos(dot)
        
        det = A[0]*B[2]-A[2]*B[0]
        if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
            return inner
        else: # if the det > 0 then A is immediately clockwise of B
            return -inner

    def truncate(self,number, digits) -> float:
        stepper = 10.0 ** digits
        return math.trunc(stepper * number) / stepper

    def controller_step(self):

        if self.wrong_action >1:
           self.done = True
        #if not self.start_speed:
        #    while self.states['speedKmh'] < 10:
        #        self.states,self._dict = self._vehicle.read_states()
        #        self.controls.change_controls(self._dict,1,0)
        #    self.start_speed=1

        #if time.time()-self.time_check >30 and self.states['speedKmh']<2:
        #    self.done=True

        if self.count > self.max_steps:
            self.done=True
        #Workflow:
        #   send action to the car
        #   Update observations using RAM Data
        #   Update the reward
        #   Store data to the buffer
        input,steering=self.controller.run_step()
        action = [input,steering]
        self.move(action)
        self.update_observations()
        self.update_reward_naive()
        self.count+= 1
        self.last_action_one = action[0]
        self.last_action = action[1]

        obs = self.obs1d
        reward = self.reward_step
        done = self.done
        info = {}
        # ─── RESET ITERATION VARIABLES ───────────────────────────────────
        self.reward_step = 0
        print(reward)
        return [obs,reward ,done,info,action]
    def store_expert_data(self):
        batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
        writer = JsonWriter(
            os.path.join(ray._private.utils.get_user_temp_dir(), "demo-out"))
        for eps_id in range(1):
            self.reset()
            self.sim.reset()
            self.controller.data_flag=True
            while self.controller.data_flag:
                
                prev_action = [self.last_action,self.last_action_one]
                prev_reward = self.reward_step
                prev_obs = self.obs1d
                obs,reward ,done,info,action = self.controller_step()
                if not self.controller.data_flag:
                    done = True
                batch_builder.add_values(
                    t=self.count,
                    eps_id=eps_id,
                    agent_index=0,
                    obs=prev_obs,
                    actions=action,
                    action_prob=1.0,  
                    action_logp=0.0,
                    rewards=reward,
                    prev_actions=prev_action,
                    prev_rewards=prev_reward,
                    dones=done,
                    infos=info,
                    new_obs=obs)
            writer.write(batch_builder.build_and_reset())

    

    
    def render(self, mode="human"):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with

        Example:

        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}

            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        print("test")
    def seed (self, seed=None):
        """Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def close(self):
        """Override close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass


    @property
    def unwrapped(self):
        """Completely unwrap this env.

        Returns:
            gym.Env: The base non-wrapped gym.Env instance
        """
        return self

    def __str__(self):
        if self.spec is None:
            return "<{} instance>".format(type(self).__name__)
        else:
            return "<{}<{}>>".format(type(self).__name__, self.spec.id)

    def __enter__(self):
        """Support with-statement for the environment."""
        return self

    def __exit__(self, *args):
        """Support with-statement for the environment."""
        self.close()
        # propagate exception
        return False