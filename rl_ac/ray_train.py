
from ray import tune
import ray
from ray.rllib import agents
from ray.rllib.agents.trainer import Trainer
from env import AC_Env
import time 
print('train')

ray.init(log_to_driver=False)

trainer = agents.ppo.PPOTrainer

tune.run(
    "DDPG", # reinforced learning agent
    name = "DDPG",
    # to resume training from a checkpoint, set the path accordingly:
    #resume = True, # you can resume from checkpoint
    #restore = r'.\ray_results\PPO\PPO_AC_Env_747c1_00000_0_2022-01-04_14-01-39\checkpoint_000160\checkpoint-160',
    checkpoint_freq = 20,
    checkpoint_at_end = True,
    local_dir = r'./ray_results/',
    config={
        "env": AC_Env,
        "num_workers": 2,
        "num_gpus": 1,
        "num_cpus_per_worker": 3,
        "framework": "torch",
        "model": {
                "_use_default_native_models": False,
                "use_lstm": True,
                },
        
        },
    stop = {
        "timesteps_total": 5_000_000,
        },
    )