from agent2 import Agent
from plot_utils import plot_learning_curve
import gym
import pybullet_envs
import pybullet
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
import stoch_gym.envs.stochlite_pybullet_env as e
#Registering new environments
from gym.envs.registration import registry, register, make, spec


if __name__=='__main__':
    # env = gym.make('InvertedPendulumBulletEnv-v0')

    register(id='Stochlite-v0',
           entry_point='stoch_gym.envs.stochlite_pybullet_env:StochliteEnv', 
           kwargs = {'gait' : 'trot', 'render': True, 'action_dim': 15, 'stairs': 0} )

    #     env = gym.make('InvertedPendulumBulletEnv-v0')
    #     env = gym.make('HalfCheetahBulletEnv-v0')
    env = gym.make('Stochlite-v0')
    agent = Agent(env=env, input_dims=env.observation_space.shape,
                    n_actions=env.action_space.shape[0])
    n_episodes = 5001
    # env = wrappers.Monitor(env, 'tmp/video', 
    # 		vide_callable=lambda episode_id: episode_id%5==0, force=True)
    # env = wrappers.Monitor(env, 'tmp/video', 
    # 		video_callable=lambda episode_id: True, force=True)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
    file_name = 'stochlite_plot.png'
    figure_file = 'plots/' + file_name 
    best_score = env.reward_range[0]
    score_history = []
    load_chkpt = True

    if load_chkpt:
        agent.load_models(5000)
        env.render(mode='human')
    # env.render(mode='human')

    for i in range(n_episodes):		
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            score += reward 
            agent.store_tuples(observation, action, reward, next_observation, done)

            if not load_chkpt:
                agent.learn()

            observation = next_observation 

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        env.close()
        # if avg_score > best_score:
        # 	best_score = avg_score
        # 	if not load_chkpt:
        # 		agent.save_models()

        if i%50==0 and not load_chkpt:
            agent.save_models(i)
            x = [j+1 for j in range(i+1)]
            plot_learning_curve(x, score_history, figure_file)

        print('episode: ',i, ' score: %.2f'%score, ' avg_score: %.2f'%avg_score)


    if not load_chkpt:
        x = [i+1 for i in range(n_episodes)]
        plot_learning_curve(x, score_history, figure_file)