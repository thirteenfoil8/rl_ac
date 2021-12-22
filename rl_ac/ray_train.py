
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
    "PPO", # reinforced learning agent
    name = "PPO",
    # to resume training from a checkpoint, set the path accordingly:
    #resume = True, # you can resume from checkpoint
    #restore = r'.\ray_results\A3C\A3C_RocketMeister10_94e86_00000_0_2021-11-02_07-53-27checkpoint_000060\checkpoint-60',
    checkpoint_freq = 100,
    checkpoint_at_end = True,
    local_dir = r'./ray_results/',
    config={
        "env": AC_Env,
        "num_workers": 1,
        "num_gpus": 1,
        "num_cpus_per_worker": 8,
        "framework": "tf",
        #"model": {
        #        "_use_default_native_models": False,
        #        "use_attention": True,
        #        },
        #"env_config":{
        #    "max_steps": 1000,
        #    "export_frames": False,
        #    "export_states": False,
        #    # "reward_mode": "continuous",
        #    # "env_flipped": True,
        #    # "env_flipmode": True,
        #    },
        
        },
    stop = {
        "timesteps_total": 5_000_000,
        },
    )