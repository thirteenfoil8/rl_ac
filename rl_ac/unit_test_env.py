from env import AC_Env
env_config={
            "max_steps": 500,
            "reward_speed_prop":False,
            "random_tp":False,
            }
env= AC_Env(env_config)
#env.find_nearest()
#env.reset()
#env.update_reward_naive()

env.store_expert_data()