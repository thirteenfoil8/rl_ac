from env import AC_Env
env= AC_Env(1)
env.find_nearest()
env.reset()
env.update_reward_naive()
_print('test')