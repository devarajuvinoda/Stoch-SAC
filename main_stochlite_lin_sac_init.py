import numpy as np
import gym
import pybullet_envs
import pybullet
from agent2 import Agent
from plot_utils import plot_learning_curve
import matplotlib.pyplot as plt
from gym import wrappers
import stoch_gym.envs.stochlite_pybullet_env as e
#Registering new environments
from gym.envs.registration import registry, register, make, spec

# give tuned actions for diversified dataset when working with slopes, tuned actions wont affect much in flat ground walking
tuned_actions_Stochlite = np.array([[-0.5, -0.5, -0.5, -0.5, 
                                        0.0, 0.0, 0.0, 0.0,
                                        -0.02, -0.02, -0.02, -0.02,
                                        0.4, 0.0, 0.0],

                                    [-0.4, -0.4, -0.4, -0.4, 
                                        0.0, 0.0, 0.0, 0.0,
                                        -0.02, -0.02, -0.02, -0.02,
                                        0.3, 0.0, 0.0],
                                        
                                    [-0.4, -0.4, -0.4, -0.4, 
                                        0.0, 0.0, 0.0, 0.0,
                                        -0.02, -0.02, -0.02, -0.02,
                                        1, 0.0, 0.0]])


if (__name__ == '__main__'):

    # NUmber of steps per episode
    num_of_steps = 100

    # list that tracks the states and actions
    # states = []
    # actions = []

    # env = sl.StochliteEnv(render=True, wedge = False, stairs = False,on_rack=False, gait = 'trot')
    register(id='Stochlite-v0',
           entry_point='stoch_gym.envs.stochlite_pybullet_env:StochliteEnv', 
           kwargs = {'gait' : 'trot', 'render': False, 'action_dim': 15, 'stairs': 0} )

    #     env = gym.make('InvertedPendulumBulletEnv-v0')
    #     env = gym.make('HalfCheetahBulletEnv-v0')
    env = gym.make('Stochlite-v0')
    agent = Agent(env=env, input_dims=env.observation_space.shape,
                    n_actions=env.action_space.shape[0])
    print("env.action_space.high: ",env.action_space.high)
    n_episodes = 5001
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

    file_name = 'stochlite_plot_linear_with_demo_2.png'
    figure_file = 'plots/' + file_name 
    best_score = env.reward_range[0]
    score_history = []
    load_chkpt = False


    if load_chkpt:
        agent.load_models(1350)
        env.render(mode='human')
#     env.render(mode='human')
    
    if not load_chkpt:
        for i in range(3):	
            observation = env.reset()
            done = False
            score = 0
            for ii in np.arange(0,2000):
                next_observation, reward, done, info = env.step(tuned_actions_Stochlite[i])
                score += reward
                agent.store_tuples(observation, tuned_actions_Stochlite[i], reward, next_observation, done)
            print("Returns of the experiment:",score)

        for i in range(10000):
            agent.learn()
    #         if i%100 == 0:
    #             action = agent.choose_action(observation)
    #             print(action)

    for i in range(n_episodes):		
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            score += reward 
            if i>100:
                agent.store_tuples(observation, action, reward, next_observation, done)

            if not load_chkpt:
                agent.learn()

            observation = next_observation 

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        env.close()
        if avg_score > best_score:
        	best_score = avg_score
        	if not load_chkpt:
        		agent.save_models("best")

        if i%50==0 and not load_chkpt:
            agent.save_models(i)
            x = [j+1 for j in range(i+1)]
            plot_learning_curve(x, score_history, figure_file)

        print('episode: ',i, ' score: %.2f'%score, ' avg_score: %.2f'%avg_score)


    if not load_chkpt:
        x = [i+1 for i in range(n_episodes)]
        plot_learning_curve(x, score_history, figure_file)