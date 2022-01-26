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
    return ClipAction(AC_Env(env_config))

register_env('Model',  env_creator)


tune.run(
    "SAC", # reinforced learning agent
    name = "SAC",
    # to resume training from a checkpoint, set the path accordingly:
    #resume = True, # you can resume from checkpoint
    #restore = r'.\ray_results\SAC\SAC_Model_c777a_00000_0_2022-01-24_18-12-16\checkpoint_000340\checkpoint-340',
    checkpoint_freq = 20,
    checkpoint_at_end = True,
    local_dir = r'./ray_results/',
    config={
        "env": 'Model',
        "store_buffer_in_checkpoints": True,
        "num_gpus": 0.5,
        "num_workers": 1,
        "num_cpus_per_worker": 12,
        "framework": "tf",
        #"train_batch_size": 128,
        "timesteps_per_iteration": 256,
        #"prioritized_replay": True,
        "lr": 0.001,
        "env_config":{
            "max_steps": 300,
            "reward_speed_prop":True,
            "random_tp":False,
            },

        "policy_model": {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        },
        },
    stop = {
        "timesteps_total": 5_000_000,
        },
    )