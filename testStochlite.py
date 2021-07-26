# ### Stochlite Testing/Data Analysis Code
# Written by Tejas Rane (May, 2021)

import sys, os
import argparse
import gym
import pybullet as p
import numpy as np
import stoch_gym.envs.stochlite_pybullet_env as e



if (__name__ == "__main__"):
	policy_dir = "logdir_name"
	policy = np.load("experiments/"+policy_dir+"/iterations/best_policy.npy") # Loading the best policy PolicyDir
	print(policy)

	iter = 1
	episode_length = 1000
	env = e.StochliteEnv(render=True, end_steps = episode_length*iter, on_rack=False,
						 gait = 'trot')

	obs = env.reset()

	p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
	for i in range(iter):
		print("iter num: ",i)
		t_r = 0
		obs = env.reset()
		for i_step in range(episode_length):
			state = obs
			state[7] = 0.4
			# print("state: ", state)
			# print("GetBaseLinearVelocity: ",env.GetBaseLinearVelocity())
			action = policy.dot(state)
			obs, r, _, angle = env.step(action)
	
	