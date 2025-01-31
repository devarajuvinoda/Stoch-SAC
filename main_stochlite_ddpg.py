import gym
import numpy as np
from agent_ddpg import Agent
from plot_utils import plot_learning_curve
import pybullet_envs
import pybullet
import matplotlib.pyplot as plt
from gym import wrappers
import stoch_gym.envs.stochlite_pybullet_env as e
#Registering new environments
from gym.envs.registration import registry, register, make, spec
import tensorflow as tf
from utils.logger import DataLog

# give tuned actions for diversified dataset when working with slopes, tuned actions wont affect much in flat ground walking
# [(X-shifts of FL, FR, BL, BR), (Y-shifts of FL, FR, BL, BR),
#                       (Z-shifts of FL, FR, BL, BR), (Augmented cmd_vel Vx, Vx, Wz)]

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



if __name__ == '__main__':
    #env = gym.make('LunarLanderContinuous-v2')
    #env = gym.make('Pendulum-v0')
    # env = gym.make('BipedalWalker-v2')
    register(id='Stochlite-v0',
           entry_point='stoch_gym.envs.stochlite_pybullet_env:StochliteEnv', 
           kwargs = {'gait' : 'trot', 'render': False, 'action_dim': 15, 'stairs': 0} )

    #     env = gym.make('InvertedPendulumBulletEnv-v0')
    #     env = gym.make('HalfCheetahBulletEnv-v0')
    env = gym.make('Stochlite-v0')
    agent = Agent(input_dims=env.observation_space.shape, 
            alpha=0.001, beta=0.002, env=env,
            gamma=0.99, n_actions=env.action_space.shape[0], 
            max_size=1000000, tau=0.005, 
            fc1=400, fc2=300, batch_size=64, noise=0.1)

    # agent = Agent(alpha=0.001, beta=0.001,
    #         input_dims=env.observation_space.shape, tau=0.005,
    #         env=env, batch_size=100, layer1_size=400, layer2_size=300,
    #         n_actions=env.action_space.shape[0])
    
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

    n_games = 3001
    figure_file = 'plots/' + 'stochlite_ddpg_from_demo_' + str(n_games) + '_games.png'

    best_score = env.reward_range[0]
    score_history = []

    load_chkpt = False


    if load_chkpt:
        agent.load_models("440")
        env.render(mode='human')
#     env.render(mode='human')

    if not load_chkpt:
#         for i in range(10):  
#             observation = env.reset()
#             done = False
#             score = 0
#             for ii in np.arange(0,1000):
#                 next_observation, reward, done, info = env.step(tuned_actions_Stochlite[i%3])
#                 score += reward
#                 agent.store_tuples(observation, tuned_actions_Stochlite[i%3], reward, next_observation, done)
#                 # demo_states.append(next_observation)
#                 # demo_actions.append(tuned_actions_Stochlite[i])
#                 observation = next_observation
#             print("Returns of the experiment:",score)

#         state_memory,action_memory,reward_memory,new_state_memory,terminal_memory=agent.memory.get_buffer()

#         np.save("../state_memory.npy", state_memory)
#         np.save("../action_memory.npy", action_memory)
#         np.save("../reward_memory.npy", reward_memory)
#         np.save("../new_state_memory.npy", new_state_memory)
#         np.save("../terminal_memory.npy", terminal_memory)    

        state_memory = np.load('../state_memory.npy')
        action_memory = np.load('../action_memory.npy')
        reward_memory = np.load('../reward_memory.npy')
        new_state_memory = np.load('../new_state_memory.npy')
        terminal_memory = np.load('../terminal_memory.npy')
        agent.memory.mem_cntr = state_memory.shape[0]
        # print(state_memory.shape, action_memory.shape,reward_memory.shape, new_state_memory.shape, terminal_memory.shape)
        agent.memory.set_buffer(state_memory, action_memory, 
            reward_memory, new_state_memory, terminal_memory)  
              
    # for i in range(5000):
    #     agent.learn()
    #     if i%100 == 0:
    #         action = agent.choose_action(observation)
    #         print(action)

        
        
        for i in range(12000):
            # print("i: ",i)
            agent.learn()
            # with tf.GradientTape() as tape:
            #     states, actions, _, _, _ = \
            #         agent.memory.sample_buffer(batch_size=100)
            #     states = tf.convert_to_tensor(states, dtype=tf.float32)
            #     actions = tf.convert_to_tensor(actions, dtype=tf.float32)
                
            #     new_actions = agent.actor(states)

            #     actor_loss = tf.keras.metrics.mean_squared_error(actions, new_actions)
            # actor_gradient = tape.gradient(actor_loss, agent.actor.trainable_variables)
            # agent.actor.optimizer.apply_gradients(
            #                 zip(actor_gradient, agent.actor.trainable_variables))

    
    for i in range(n_games):
        observation = env.reset()
        # observation,_,_,_ = env.step(tuned_actions_Stochlite[0])
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)            
            observation_, reward, done, info = env.step(action)
            # if i > 100:
            #     agent.store_tuples(observation, action, reward, observation_, done)
            
            if not load_chkpt:
                agent.learn()
#             print("action: ",action)
#             print("reward: ",reward)
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not load_chkpt:
            if avg_score > best_score:
                best_score = avg_score
                agent.save_models("best")
            if i%40==0:
                agent.save_models(i)
            x = [j+1 for j in range(i+1)]
            plot_learning_curve(x, score_history, figure_file)

        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)

    if not load_chkpt:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
