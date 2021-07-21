# import gym
# import pybullet, pybullet_envs

# env = gym.make('InvertedPendulumBulletEnv-v0')
# env.render('human')

from agent import Agent
from plot_utils import plot_learning_curve
import gym
import pybullet_envs
import pybullet
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
import gym_sloped_terrain.envs.stoch2_pybullet_env as e
#Registering new environments
from gym.envs.registration import registry, register, make, spec


if __name__=='__main__':
    # env = gym.make('InvertedPendulumBulletEnv-v0')

    register(id='Stoch2-v0',
           entry_point='gym_sloped_terrain.envs.stoch2_pybullet_env:Stoch2Env', 
           kwargs = {'gait' : 'trot', 'render': True, 'action_dim': 20, 'stairs': 0} )
  
#     env = gym.make('InvertedPendulumBulletEnv-v0')
#     env = gym.make('HalfCheetahBulletEnv-v0')
    env = gym.make('Stoch2-v0')
    env.render('human')
    observation = env.reset()
    for _ in range(1000):
        # env.render()
        
        # env.render()

        action = env.action_space.sample()
        print("action: ",action)
        observation, reward, done, info = env.step(action)
        print("observation, reward, done, info: ",observation, reward, done, info)
        if done:
            observation = env.reset()
    # env.close()