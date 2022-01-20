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
    "A3C", # reinforced learning agent
    name = "A3C",
    # to resume training from a checkpoint, set the path accordingly:
    #resume = True, # you can resume from checkpoint
    #restore = r'.\ray_results\PPO\PPO_Model_16fc1_00000_0_2022-01-13_18-20-23\checkpoint_000420\checkpoint-420',
    checkpoint_freq = 20,
    checkpoint_at_end = True,
    local_dir = r'./ray_results/',
    config={
        "env": 'Model',
        "num_workers": 1,
        "num_gpus": 1,
        "num_cpus_per_worker": 6,
        "framework": "torch",
        "lr": 0.001,
        #"rollout_fragment_length": 200,
        #"train_batch_size": 4096,
        "batch_mode": "complete_episodes",
        "model": {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                #"_use_default_native_models": False,
                #"use_lstm": True,
                },
        #"prioritized_replay": False,
        
        #"train_batch_size": 64,
        #"store_buffer_in_checkpoints": True,
        #"policy_model": {
        #    "fcnet_hiddens": [64, 64],
        #    "fcnet_activation": "relu",
        #    "post_fcnet_hiddens": [],
        #    "post_fcnet_activation": None,
        #    "custom_model": None,  # Use this to define a custom policy model.
        #    "custom_model_config": {},
        #},
        },
    stop = {
        "timesteps_total": 5_000_000,
        },
    )