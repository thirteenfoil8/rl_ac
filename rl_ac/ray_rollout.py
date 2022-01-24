import gym, json
from ray.rllib import rollout
from ray.tune.registry import register_env
import os 
from env import AC_Env,ClipAction


def env_creator(env_config):
    return ClipAction(AC_Env(1))

register_env('Model',  env_creator)



# path to checkpoint
checkpoint_path = r'ray_results\SAC\SAC_prog_speed\checkpoint_013840\checkpoint-13840'
#ppo   r'.\ray_results\PPO_Vehicule\PPO_RocketMeister10_15db7_00000_0_2021-11-01_08-05-25\checkpoint_000440\checkpoint-440'

string = ' '.join([
    checkpoint_path,
    '--run',
    'SAC',
    '--env',
    'Model',
    '--episodes',
    '1',
    '--video-dir',
    r'.\media',
    # '--no-render',
])

parser = rollout.create_parser()
args = parser.parse_args(string.split() )

# ──────────────────────────────────────────────────────────────────────────
# if you want to automate this, by calling rollout.run() multiple times, you
# uncomment the following lines too. They need to called before calling
# rollout.run() a second, third, etc. time
# ray.shutdown()
# tune.register_env("rocketgame", lambda c: MultiEnv(c))
# from ray.rllib import _register_all
# _register_all()
# ──────────────────────────────────────────────────────────────────────────

rollout.run(args, parser)
#os.system('rllib rollout \
#    ray_results/Training2/PPO_RocketMeister10_8de4a_00000_0_2021-10-26_08-40-05/checkpoint_001250/checkpoint-1250 \
#    --run PPO --env rocketmeister --steps 10000')