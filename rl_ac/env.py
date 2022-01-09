import gym
import numpy as np
import sim_info
from lidar import init_track_data,compute_lidar_distances
from gym.utils import seeding
from numpy.core.umath_tests import inner1d
import time
import random
from ctypes import c_float


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
        self.spline_before = 0
        self.out_of_road = 1
        self.start_speed = 0
        self.time_check = 0
        self.last_action = 0
        self.collision = 0
        self.max_steps= 6000
        # add max_steps if spline_position is bigger than modulo 10%
        self.wrong_action = 0
        self.count= 0
        
        
        self._vehicle =  sim_info.SimInfo()
        self.init_pos = c_float * 3
        self.init_dir = c_float * 3
        
        self.controls = sim_info.SimControl()
        self.sim =sim_info.SimStates()
        self.states,_ = self._vehicle.read_states() 
        
        self.obs1d = []
        self.sideleft_xy,self.sideright_xy,self.centerline_xy,self.normal_xyz,self.track_data = init_track_data()

        # Set these in ALL subclasses
        self.action_space = gym.spaces.Box(
            low=-1.,
            high=1.,
            shape=(2,),
            dtype=np.float32)

        self.observation_space = gym.spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(62,), #50 without Lidar + 61 lidar + curvature= 112 obs in float 32 --> 804 bytes per step
                dtype=np.float32)
        self.seed()
        self.reset()
    def teleport_conversion(self):

        FloatArr = c_float * 3
        

        #random_splines = np.arange(0,1,0.05)
        #spline = random.choice(random_splines)
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
        _look[0] = -0.9996861815452576
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

        Returns:
            observation (object): the initial observation.
        """
        self.done = False
        self.teleport_conversion()
        for i in range(5):
            self.controls.teleport(self.init_pos,self.init_dir)
            time.sleep(0.2)
        self.states,dict = self._vehicle.read_states()
        
        self.controls.change_controls(self.states,0,0)
        
        #self.sim.reset()
        self.update_observations()
        self.reward_total = 0
        self.count= 0
        self.wrong_action = 0
        self.out_of_road= 1
        self.reward_step= 0
        self.start_speed=0
        self.time_check=time.time()
        self.collision =dict['collision_counter']
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
        self.spline_before = self.states['spline_position']
        # reward negativ on huge delta step. over 1 degree - > negative reward proportional to delta 
        # reward positif between 0-1 ° 
        if action[1] > self.last_action+0.04: # 18° 
            action[1] = self.last_action+0.04
        if action[1] < self.last_action-0.04:
            action[1] = self.last_action-0.04
        if self.wrong_action >20:
            self.done = True
        if not self.start_speed:
            while self.states['speedKmh'] < 10:
                self.states,dict = self._vehicle.read_states()
                self.controls.change_controls(self.states,1,0)
            self.start_speed=1

        if time.time()-self.time_check >20 and self.states['speedKmh']<2:
            self.done=True

        if self.count > self.max_steps:
            self.done=True
        if not self.done:
            self.move(action)
            self.update_observations()
            self.update_reward()
            self.count+= 1
            self.last_action = action[1]

        obs = self.obs1d
        reward = self.reward_step
        done = self.done
        info = {}
        # ─── RESET ITERATION VARIABLES ───────────────────────────────────
        self.reward_step = 0
        return [obs,reward ,done,info]

    def update_observations(self):

        self.out_of_road=1
        self.states,dict = self._vehicle.read_states()
       
        if dict['collision_counter']> self.collision:
            self.out_of_road = -100
            self.done=True

        self.obs1d = np.array([])
        #self.obs1d = sim_info.states_to_1d_vector(self.states)
        
        #Compute the distance with the border of the track
        lidar= compute_lidar_distances(self.states["look"],self.states["position"],self.sideleft_xy,self.sideright_xy)
        
        # if len(lidar)< 61, then the car is out of track --> add 0 to have data consistency
        if lidar.shape[0]< 61:
            self.out_of_road= -100
            self.wrong_action +=1
            self.obs1d =np.append(self.obs1d,lidar)
            self.obs1d =np.append(self.obs1d,np.zeros(61-lidar.shape[0]).astype(np.float32))
        
        else:
            self.obs1d =np.append(self.obs1d,lidar)
        self.obs1d = np.append(self.obs1d,self.find_curvature().astype(np.float32))
        

    def update_reward(self):
        # Sliding 
        #Maybe penalize pwd on gas
        dist,angle = self.find_nearest()
        reward_unit = self.states['spline_position']-self.spline_before
        reward_ai_angle = 0
        # If distance with centerline > 10m --> negative reward

        if dist > 10:
            reward_ai_pos = 0
        # Else, the closer to 0m , the higher the reward 
        else:
            dist_normed= dist/10
            reward_ai_pos = (1-dist_normed)
        #If the angle between the look angle of the car and the AI angle > np.pi/4 --> negative reward
        if angle > np.pi/9 or angle < -np.pi/9:
            reward_ai_angle = 0

        #Add positiv reward for angle diff small
        if np.abs(angle)<np.pi/36:
            reward_ai_angle =  10*(1-np.abs(angle/(np.pi/36)))

        #Maybe Add Reward based on the angle of the steer related to the curvature of the track

        # Reward step = distance achieved reward + distance to centerline reward + angle with centerline angle reward. If out of road, the reward becomes negativ
        self.reward_step= reward_unit*(self.out_of_road +reward_ai_pos + reward_ai_angle)
        self.reward_total += self.reward_step

    def find_curvature(self):
        idx=(np.abs(self.track_data[:,3]-self.states['spline_position'])).argmin()
        #Return the curvature radius on the current location
        return (1/(self.track_data[idx,12]*self.track_data[idx,8]))
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
        return norm,error

    def angle_clockwise(self,A,B):
        dot = np.dot(A,B)
        inner = np.arccos(dot)
        
        det = A[0]*B[2]-A[2]*B[0]
        if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
            return inner
        else: # if the det > 0 then A is immediately clockwise of B
            return -inner

    

    
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
        print(s.format(self.states, self.reward_step, self.info))
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