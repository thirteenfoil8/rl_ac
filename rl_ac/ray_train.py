from ray import tune
import ray
from ray.rllib import agents
from ray.rllib.agents.trainer import Trainer
from env import AC_Env,ClipAction
from ray.tune.registry import register_env
import time 
print('train')

ray.init(log_to_driver=False,object_store_memory=10**9)
def env_creator(env_config):
    return ClipAction(AC_Env(env_config))

register_env('Model',  env_creator)



tune.run(
    "SAC", # reinforced learning agent
    name = "SAC",
    # to resume training from a checkpoint, set the path accordingly:
    #resume = True, # you can resume from checkpoint
    #restore =r'.\ray_results\SAC\SAC_Model_43487_00000_0_2022-03-08_09-53-45\checkpoint_013800\checkpoint-13800',
    checkpoint_freq = 200,
    checkpoint_at_end = True,
    local_dir = r'./ray_results/',
    config={
        "env": 'Model',
        
        "num_gpus": 1,
        "num_workers": 1,
        "framework": "tf",
        #"framework": "tf2",
        #"eager_tracing": True,
        
        "lr": 0.0001,
        "env_config":{
            "max_steps": 300,
            "reward_speed_prop":False,
            "random_tp":False,
            "errors":50,
            "track":"vallelunga",
            "store_data":False,
            "normalize_obs":False,
            "centralize_obs":False,
            "progressiv_action":False,
            },
        ### Uncomment this if you use SAC
        "store_buffer_in_checkpoints": True,
        "prioritized_replay": True,
        "policy_model": {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        },

        "Q_model": {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
        },

        
        },
    stop = {
        "timesteps_total": 5_000_000,
        },
    )