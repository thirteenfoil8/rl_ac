import gym, json
from ray.rllib import evaluate
from ray.tune.registry import register_env
import os 
from env import AC_Env,ClipAction


def env_creator(env_config):
    return ClipAction(AC_Env(env_config))

register_env('Model',  env_creator)



# path to checkpoint
checkpoint_path = r'.\ray_results\SAC\SAC_Model_66510_00000_0_2022-01-31_12-10-29\checkpoint_031420\checkpoint-31420'

string = ' '.join([
    checkpoint_path,
    '--run',
    'SAC',
    '--env',
    'Model',
    '--episodes',
    '10',
    '--steps', 
    '1000000'
])
config = {
    "env_config":{
            "max_steps": 500,
            "reward_speed_prop":False,
            "random_tp":False,
            "errors":200,
            "track":"vallelunga",
            },
}
config_json = json.dumps(config)
parser = evaluate.create_parser()
args = parser.parse_args(string.split() + ['--config', config_json])

evaluate.run(args, parser)
