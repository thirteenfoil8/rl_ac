import gym, json
from ray.rllib import evaluate
from ray.tune.registry import register_env
import os 
from env import AC_Env,ClipAction


def env_creator(env_config):
    return ClipAction(AC_Env(env_config))

register_env('Model',  env_creator)



# path to checkpoint
checkpoint_path = r'ray_results\SAC\SAC_prog_speed\checkpoint_013440\checkpoint-13440'
#ppo   r'.\ray_results\PPO_Vehicule\PPO_RocketMeister10_15db7_00000_0_2021-11-01_08-05-25\checkpoint_000440\checkpoint-440'

string = ' '.join([
    checkpoint_path,
    '--run',
    'SAC',
    '--env',
    'Model',
    '--episodes',
    '5',
    '--steps', 
    '1000000'
])
config = {
    "env_config":{
            "max_steps": 500,
            "reward_speed_prop":False,
            "random_tp":True,
            },
}
config_json = json.dumps(config)
parser = evaluate.create_parser()
args = parser.parse_args(string.split() + ['--config', config_json])

evaluate.run(args, parser)
