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
    #restore =r'.\ray_results\SAC\SAC_evaluate_new_lidar\checkpoint_031400\checkpoint-31400',
    checkpoint_freq = 200,
    checkpoint_at_end = True,
    local_dir = r'./ray_results/',
    config={
        "env": 'Model',
        
        "num_gpus": 1,
        "num_workers": 1,
        "num_cpus_per_worker": 1,
        "framework": "tf",
        #"framework": "tf2",
        #"eager_tracing": True,
        #"train_batch_size": 128,
        #"timesteps_per_iteration": 256,
        
        "lr": 0.001,
        "env_config":{
            "max_steps": 300,
            "reward_speed_prop":False,
            "random_tp":False,
            "errors":50,
            "track":"vallelunga",
            },
        ### Uncomment this if you use SAC
        "store_buffer_in_checkpoints": True,
        "prioritized_replay": True,
        "_deterministic_loss":True,
        "policy_model": {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "tanh",
        },

        "Q_model": {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "tanh",
        },

        
        },
    stop = {
        "timesteps_total": 5_000_000,
        },
    )