from ray import tune
import ray
from ray.rllib import agents
from ray.rllib.agents.trainer import Trainer
from env import AC_Env,ClipAction
from ray.tune.registry import register_env
import time 
print('train')

ray.init(log_to_driver=False)
def env_creator(env_config):
    return ClipAction(AC_Env(1))

register_env('Model',  env_creator)


tune.run(
    "SAC", # reinforced learning agent
    name = "SAC",
    # to resume training from a checkpoint, set the path accordingly:
    #resume = True, # you can resume from checkpoint
    #restore = r'.\ray_results\PPO\PPO_Model_16fc1_00000_0_2022-01-13_18-20-23\checkpoint_000420\checkpoint-420',
    checkpoint_freq = 20,
    checkpoint_at_end = True,
    local_dir = r'./ray_results/',
    config={
        "env": 'Model',
        "num_workers": 3,
        "num_gpus": 1,
        "num_cpus_per_worker": 3,
        "framework": "tf",
        "lr": 0.0001,
        #"store_buffer_in_checkpoints": True,
        #"learning_starts": 1000,
        #"train_batch_size": 1000,
        #"input": "C:/Users/flori/AppData/Local/Temp/demo-out",
        #"evaluation_config":{
        #    "input": "sampler",
        #    },
        
        #"train_batch_size": 4096,
        
        
        "policy_model": {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        },
        },
    stop = {
        "timesteps_total": 5_000_000,
        },
    )