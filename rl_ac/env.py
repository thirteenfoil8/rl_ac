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
        self.k11 = 0.1
        self.k12 = 0.3
        self.k21 = 0.9
        self.k22 = 0.7

    def action(self, action):
        if self.progressiv_action:
            action[0] = action[0]*self.k12 + self.last_input*self.k22  
            action[1] = action[1]*self.k11 + self.last_steering*self.k21
        else:
        
            #if action[1] > self.last_steering+0.04: # 9° 
            #    action[1] = self.last_steering+0.04
            #if action[1] < self.last_steering-0.04:
            #    action[1] = self.last_steering-0.04
            pass
        

        
        
        return np.array([np.clip(action[0], self.low, self.high),np.clip(action[1], self.low, self.high)])

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
        self.parse_env_config(env_config)
        self.initialise_data()
        self.done = False
        self.info = {}
        self.reward_total= 0
        self.reward_step = 0
        self.n_obj=0
        self.out_of_road = 0
        self.time_check = 0
        self.last_steering = 0
        self.last_input=0
        self.collision = 0
        self.spline_start= 0
        self.wrong_action = 0
        self.count= 0
        self.obj_to_lap = 0
        self.eps=0
        self.data = []
        self.time_section = 0
        
        
        
        self._vehicle =  sim_info.SimInfo()
        self.init_pos =  self._vehicle.init_pos
        self.init_dir =  self._vehicle.init_dir
        self.tp_pos= c_float *3
        self.tp_dir= c_float *3
        
        self.controls = sim_info.SimControl()
        self.sim =sim_info.SimStates()
        self.line=sim_info.LineControl()
        self.states,self._dict = self._vehicle.read_states() 
        self.sim.speed(3)
        
        
        self.obs1d = []
        self.sideleft_xy,self.sideright_xy,self.centerline_xy,self.normal_xyz,self.track_data = init_track_data(track=self.track)
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
                shape=(84,), #6 car states + 61 lidar + 14curvatures + angle+dist+ self.last_actions= 85 obs in float 32 --> 804 bytes per step
                dtype=np.float32)
        self.seed()
        self.reset()
    def parse_env_config(self,env_config):
        keyword_dict = {
            # these are all available keyboards and valid values respectively
            # the first value in the list is the default value
            'max_steps'             : [500, 'any', int],
            'reward_speed_prop'     : [False,True],
            'random_tp'             :[False,True],
            'errors'                : [50, 'any', int],
            'track'                 :['vallelunga','any',str],
            'store_data'            :[True,False],  
            'normalize_obs'         :[False,True],
            "centralize_obs"      :[False,True],
            "progressiv_action"   :[False,True],
        }
        
        # ─── STEP 1 GET DEFAULT VALUE ────────────────────────────────────
        assign_dict = {}
        for keyword in keyword_dict:
            # asign default value form keyword_dict
            assign_dict[keyword] = keyword_dict[keyword][0]
            
        # ─── STEP 2 GET VALUE FROM env_config ─────────────────────────────
        for keyword in env_config:
            if keyword in keyword_dict:
                # possible keyword proceed with assigning
                if env_config[keyword] in keyword_dict[keyword]:
                    # valid value passed, assign
                    assign_dict[keyword] = env_config[keyword]
                elif 'any' in keyword_dict[keyword]:
                    # any value is allowed, assign if type matches
                    if isinstance(env_config[keyword],keyword_dict[keyword][2]):
                        print('type matches')
                        assign_dict[keyword] = env_config[keyword]
                    else:
                        print('error: wrong type. type needs to be: ', keyword_dict[keyword][2])
                else:
                    print('given keyword exists, but given value is illegal')
            else:
                print('passed keyword does not exist: ',keyword)
        # ─── ASSIGN DEFAULT VALUES ───────────────────────────────────────
        self.max_steps = assign_dict['max_steps']
        self.reward_speed_prop = assign_dict['reward_speed_prop']
        self.random_tp=assign_dict['random_tp']
        self.errors = assign_dict['errors']
        self.track = assign_dict['track']
        self.store_data = assign_dict['store_data']
        self.normalize_obs = assign_dict['normalize_obs']
        self.centralize_obs = assign_dict["centralize_obs"]
        self.progressiv_action = assign_dict["progressiv_action"]
    def teleport_conversion(self):
        # add offset for generalization
        FloatArr = c_float * 3
        
        if self.random_tp :
            random_splines = np.arange(0,1,0.05)
            spline = random.choice(random_splines)
            idx=(np.abs(self.track_data[:,3]-spline)).argmin()
            _waypoint = FloatArr()
            _waypoint[0] = self.track_data[idx,0]
            _waypoint[1] = self.track_data[idx,1]
            _waypoint[2] =  self.track_data[idx,2]
            _look = FloatArr()
            _look[0] = -self.track_data[idx,16]
            _look[1] = self.track_data[idx,17]
            _look[2] = -self.track_data[idx,18]
        else:
            _waypoint = FloatArr()
            _waypoint[0] = self.init_pos[0]
            _waypoint[1] = self.init_pos[1]
            _waypoint[2] =  self.init_pos[2]
            _look = FloatArr()
            _look[0] = -self.init_dir[0]
            _look[1] = self.init_dir[1]
            _look[2] = -self.init_dir[2]


        self.tp_pos =  _waypoint
        self.tp_dir =  _look
    def initialise_data(self):
        """Initialise ram files in order to be able to use the Drive Master API"""
        sim_info.create_cars()
    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.

        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.
        Returns:
            observation (object): the initial observation.
        """
        # Reset car states and Teleport to initial position
        self.done = False
        self.teleport_conversion()
        self.controls.teleport(self.tp_pos,self.tp_dir)
        time.sleep(0.5)
        self.states,self._dict = self._vehicle.read_states()
        self.controls.change_controls(self._dict,0,0)
        
        #self.sim.reset()

        # Reset Agent States 
        self.update_observations()
        self.reward_total = 0
        self.count= 0
        self.eps +=1
        self.wrong_action = 0
        self.n_obj = 0
        self.out_of_road= 0
        self.reward_step= 0
        self.last_steering=0
        self.last_input=0
        self.data=[]
        self.time_section = time.time()
        self.time_check=time.time()
        self.collision =self._dict['collision_counter']
        self.spline_start=self.truncate(self.states['spline_position'],2)
        self.spline_before = self.states['spline_position']
        self.obj_to_lap=100+ (100-self.truncate(self.spline_start*100,0)) 
        self.data_cleaning = False

        return self.obs1d
    def move(self,action):
        """ Run one timestep of the car dynamics. Write into RAM the next action 
        """
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

        
        
        # If the car is out of the track too long, then the episode is early terminated
        if self.store_data:
            self.data.append(self.states)
            self.data.append({"reward":self.reward_step})
        if self.wrong_action >self.errors:
            print('too many wrong actions')
            self.done = True
        if self.states['spline_position'] <self.spline_before and not self.data_cleaning:
            print('cleaning')
            self.data=[]
            self.data_cleaning=True
        if self.n_obj > self.obj_to_lap:
            if self.store_data:
                df = pd.DataFrame(self.data)
                self.sim.pause()
                df.to_excel("test.xlsx",index=False)
                self.sim.play()
                self.data=[]
            self.done=True

        # If the car is blocked in one segment, then the episode is early terminated
        if self.count > self.max_steps:
            print('too slow in segment ' + str(self.n_obj))
            self.done=True
        #Workflow:
        #   send action to the car
        #   Update observations using RAM Data
        #   Update the reward
        #   Store data to the buffer
        
        if not self.done:
            self.spline_before = self.states['spline_position']
            self.move(action)
            self.update_observations()
            self.update_reward_naive()
            self.count+= 1
            self.last_steering = action[1]
            self.last_input = action[0]
            

        obs = self.obs1d
        reward = self.reward_step
        done = self.done
        info = {}
        return [obs,reward ,done,info]

    def normalizeData(self,data):
        # for lidar 10 m max 0 min 
        
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def update_observations(self):
        """Update all the obersvations: Car states, Distance and angle differences with the reference,
        distance with the border of the track using the Lidar Sensors, 14 front points regarding the Curvature
        and the last action.

        

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # If the car is out of the road, this flag is set to 1
        self.out_of_road=0

        #Read states of the car
        self.states,self._dict = self._vehicle.read_states()
        dist,angle,_ = self.find_nearest()
       
        # If the car collides with a wall, then the episode is early terminated
        if self._dict['collision_counter']> self.collision:
            self.out_of_road = 1
            self.wrong_action+=1
        #If the car wheels are dirty, this means that the car is out of the track
        for i in self._dict['dirty_level']:
            if i >0:
                self.out_of_road = 1
                self.wrong_action+=1

        #Collect all the observations inside a 1d np array in float32 format
        self.obs1d = np.array([])
        self.obs ={k:self.states[k] for k in ('velocity','acceleration') if k in self.states}
        self.obs1d = sim_info.states_to_1d_vector(self.obs)
        self.obs1d = np.append(self.obs1d,np.array([self.last_steering]).astype(np.float32))
        #self.obs1d = np.append(self.obs1d,np.array([self.last_input]).astype(np.float32))
        
        #Compute the distance with the border of the track
        lidar,_,_= compute_lidar_distances(self.states["look"],self.states["position"],self.sideleft_xy,self.sideright_xy,self.centralize_obs)
        if lidar.shape[0]!=0:
            if self.normalize_obs:
                lidar=self.normalizeData(lidar)
        # if len(lidar)< 61, then the car is out of track --> add 0 to have data consistency
        if lidar.shape[0]< 61:
            self.out_of_road= 1
            self.wrong_action +=1
            self.obs1d =np.append(self.obs1d,lidar)
            self.obs1d =np.append(self.obs1d,-np.ones(61-lidar.shape[0]).astype(np.float32))
        else:
            self.obs1d =np.append(self.obs1d,lidar)
        self.obs1d = np.append(self.obs1d,self.find_curvature().astype(np.float32))
        self.obs1d = np.append(self.obs1d,np.array([dist,angle]).astype(np.float32))

        #Sometimes, some RAM data are nan (too fast data collecting) --> replace nan values by 0
        self.obs1d=np.nan_to_num(self.obs1d)
        

    def update_reward(self):
        """First try of reward designing based on the car states only.
            This was not the good way to proceed as the car doesn't learn how to progress on the track
        """
        dist,angle,speed = self.find_nearest()

        #Speed reward is computed as the faster, the bigger the reward is
        reward_speed = 1-(np.abs(self.states['speedKmh']-speed)/(speed+1))
        
        # Distance with the reference reward. The closer to 0, the biggest the reward
        if dist > 10:
            reward_ai_pos = 0
        else:
            dist_normed= dist/10
            reward_ai_pos = 1-dist_normed

        # relative angle with the reference reward. The closer to 0, the biggest the reward
        reward_ai_angle = np.cos(angle) 
       
            
        #The reward step is a weighted average of the three reward above
        self.reward_step= self.states['speedKmh']*(reward_ai_pos + reward_ai_angle)

        if self.out_of_road:
            self.reward_step = -10

        #The reward is not taken into account if the agent speed is smaller than 0
        #This way, the agent understand quickly that it's better to accelerate when extreme low speed
        if self.states['speedKmh'] < 1:
            self.reward_step = -0.1
        self.reward_total += self.reward_step
    def update_reward_naive(self):
        """Second try of reward designing based on the car progression along the track.
            This function first look how much the car has progressed along the track and then
            gives a bigger reward if it did it fast. In addition, if the car is stucked inside a track segment,
            The reward becomes smaller and smaller
        """
        
        #Find the progression along one segment and the time inside it
        dist_goal_before,dist_goal_after= self.find_progression()
        time_in_section= time.time()-self.time_section
        time_reward = time_in_section/10

        #If the car stayed too long in one segment, then the reward is decreasing
        #if time_reward >0.3:
        #    time_reward += self.n_obj
        
        # The nearer to the segment end/goal, the biggest the reward 
        self.reward_step= self.n_obj+ (1- (dist_goal_after/(dist_goal_after+dist_goal_before)))*(1- time_reward)

        #The reward_speed term has been added to accentuate the fact that it's better to drive as fast as possible
        if self.reward_speed_prop:
            spline_kevin = (np.abs(self.df.NormalizedSplinePosition-self.states['spline_position'])).argmin()

            speed = self.df.Speed.iloc[[spline_kevin]].values[0]
            x = self.states['speedKmh']/speed

            speed_reward = -0.1*np.exp(30*np.power((2*x-1),4)-32)+0.6*np.exp(np.power(x,2))-0.5
            self.reward_step +=speed_reward

        

        #When out of the track, the reward becomes negative to indicate that it's not a good way to drive
        if self.out_of_road:
            self.reward_step -= 100

        #if self.states['speedKmh'] <1 and time_reward > 0.2:
        #    self.reward_step = min([-0.1,self.reward_step])
        self.reward_total += self.reward_step

    def find_progression(self):
        """This function is looking in depth where the car is in order to compute the past segment goal and
        the new one in order to have an idea of the progression.

        

        Returns:
            dist_goal_before (float): Agent's distance with the past goal
            dist_goal_after (float): Agent's distance with the next goal
        """
        #Find agent past and next goal
        spline_goal_before =  self.truncate(self.states['spline_position'],2)
        spline_goal_after = spline_goal_before+0.01
        idx_goal_before = (np.abs(self.track_data[:,3]-spline_goal_before)).argmin()
        idx_goal_after = (np.abs(self.track_data[:,3]-spline_goal_after)).argmin()
        pos_goal_before = self.track_data[idx_goal_before,0:3]
        pos_goal_after = self.track_data[idx_goal_after,0:3]

        #self.line.conversion(pos_goal_before,pos_goal_after)

        #Compute the distance betwen the car and the 2 goals
        dist_goal_before = np.linalg.norm(self.states['position']-pos_goal_before)
        dist_goal_after = np.linalg.norm(pos_goal_after-self.states['position'])

        #Self.n_obj is incremented if the car has reached the new segment.
        if spline_goal_before != self.spline_start  :
            self.n_obj += 1
            self.spline_start=spline_goal_before
            self.count = 0
            self.time_section=time.time()

        return dist_goal_before,dist_goal_after

    def find_curvature(self):
        """This function is looking in depth where the car is in order to compute the past segment goal and
        the new one in order to have an idea of the progression.

        

        Returns:
            curvatures (np.array([14]), dtype:float): Track's next 14 radius of curvature in front of the car
        """
        # Find the position of the car
        idx=(np.abs(self.track_data[:,3]-self.states['spline_position'])).argmin()
        
        curvatures = np.array([])
        #Compute the next 14 radius of curvature based on the car's position along the track

        for i in range(14):
            if idx + i >= len(self.track_data[:,11]):
                new_idx = (idx + i)-len(self.track_data[:,11])
                #try Ln(1/x) instead of 1/x
                if self.centralize_obs:
                    curvatures = np.append(curvatures,np.log(1/(self.track_data[new_idx,7])))
                else:
                    curvatures = np.append(curvatures,1/(self.track_data[new_idx,11]*self.track_data[new_idx,7]))
            else:
                if self.centralize_obs:
                    curvatures = np.append(curvatures,np.log(1/(self.track_data[idx + i,7])))
                else:
                    curvatures = np.append(curvatures,1/(self.track_data[idx + i,11]*self.track_data[idx + i,7]))
        if self.normalize_obs:
            return self.normalizeData(curvatures)
        else:
            return curvatures
    def find_nearest(self):
        """This function is looking in depth where the car is in order to compute the difference between its 
        position, its angle and its speed difference with the reference.

        

        Returns:
            norm (float): Difference of position with respect to the reference
            error (float): Difference of angle with respect to the reference
            speed (float): Difference of speed with respect to the reference

        """

        #Find the reference data regarding the car's position
        x = self.centerline_xy[:,0]
        y = self.centerline_xy[:,1]
        pos = np.array([x,y])
        x_car,z_car,y_car = self.states['position']
        deltas = pos - np.array([x_car,-y_car]).reshape((2,1))
        norms =np.sqrt(inner1d(np.stack(deltas,axis = 1),np.stack(deltas,axis = 1)))
        minimum_norm = norms.argmin()
        norm = norms.min()
        #minimum_norm += 6
        if minimum_norm >= norms.shape[0]: 
            minimum_norm = minimum_norm - norms.shape[0]
        
        look = self.states['look']
        ideal = self.normal_xyz[minimum_norm]
        error = self.angle_clockwise(look,[ideal[0],ideal[1],ideal[2]])


        spline_kevin = (np.abs(self.df.NormalizedSplinePosition-self.states['spline_position'])).argmin()
        speed = self.df.Speed.iloc[[spline_kevin]]

        if self.centralize_obs:
            norm = -np.log(norm)
            error = -np.log(error)


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
        """This function is only used in case of driving the car using the PID controller 
        for a Imitation learning Algorithm.

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)

        """

        if self.wrong_action >1:
           self.done = True


        if self.count > self.max_steps:
            self.done=True
        #Workflow:
        #   Compute action using the PID Controller
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
        self.last_input = action[0]
        self.last_steering = action[1]

        obs = self.obs1d
        reward = self.reward_step
        done = self.done
        info = {}
        # ─── RESET ITERATION VARIABLES ───────────────────────────────────
        self.reward_step = 0

        return [obs,reward ,done,info,action]
    def store_expert_data(self):
        """This function is only used in case of driving the car using the PID controller 
        to store the data for a Imitation learning Algorithm.

        """

        batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
        writer = JsonWriter(
            os.path.join(ray._private.utils.get_user_temp_dir(), "demo-out"))
        for eps_id in range(1):
            self.reset()
            self.controller.data_flag=True
            #While the lap is not finished, Store all states and actions inside a buffer
            while self.controller.data_flag:
                prev_action = [self.last_steering,self.last_input]
                prev_reward = self.reward_step
                prev_obs = self.obs1d
                obs,reward ,done,info,action = self.controller_step()
                print(self._dict)
                print("\n \n \n ############################")
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
            #Write the buffer inside a Json file

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