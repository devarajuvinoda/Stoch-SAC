import gym
import numpy as np
from agent_td3 import Agent
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

'''
initial policy:  [-1., 0.3426804, 1., -0.25243604, 
    0.98132079, 0.51421884,  0.22117967, -1., 
    -0.18949583, 0.25500144, -0.45802699, 0.43516349,
     -0.58359505, 0.81684707, 0.67272081]


'''
 
def get_mean(arr):
#     mx = max(arr)
#     mn = max(arr)
#     if mn==mx:
#         mx += 1
#     arr = (2*(arr-mn)/(mx-mn)) - 1
    arr = np.array(arr)
    return np.mean(arr)

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
    
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
    
    n_games = 2001
    file_name = 'stochlite_plot_sac_demo.png'
    figure_file = 'plots/' + file_name 
    best_score = env.reward_range[0]
    score_history = []
    
    np.random.seed(100)    
    init_policy = np.clip(np.random.randn(1,15), -1, 1)[0]
    print("initial policy: ",init_policy)
    epsilon = 0.01
    for i in range(201):        
        new_delta_policy = []
        pos_delta_score = []
        neg_delta_score = []
        zero_delta_score = []
        update_vals = []

        for j in range(20):
            delta = np.random.randint(-1,2, size=15)
            action = init_policy + delta*epsilon
            action = np.clip(action, -1, 1)
            observation = env.reset()
            done = False
            score = 0 
            for k in range(250):
                observation_, reward, done, info = env.step(action)
#                 print("action: ",action)
#                 print("reward: ",reward)
                observation = observation_
                score += reward

            score_history.append(score)
            avg_score = np.mean(score_history[-10:])  
            
            for k in range(15):
                if delta[k]<0:
                    neg_delta_score.append(score)
                if delta[k]>0:
                    pos_delta_score.append(score)
                else:
                    zero_delta_score.append(score)
        x = [k+1 for k in range((i+1)*20)]
        plot_learning_curve(x, score_history, figure_file)

        for j in range(15):
            neg_mean = get_mean(neg_delta_score)
            pos_mean = get_mean(pos_delta_score)
            zero_mean = get_mean(zero_delta_score)

            update_par = 0
            if not (zero_mean > neg_mean and zero_mean > pos_mean):
                update_par =  pos_mean - neg_mean

            update_vals.append(update_par)
#         print(update_vals, update_vals.shape)
        update_vals = np.array(update_vals)
        update_vals = 0.01 * (update_vals/np.linalg.norm(update_vals))
#         print(update_vals, update_vals.shape)
        init_policy += update_vals
#         init_policy = np.clip(init_policy, -1, 1)

        print("learned policy: ", init_policy)
    while not done:      
            observation_, reward, done, info = env.step(init_policy)
#             print("action: ",action)
#             print("reward: ",reward)

    x = [i+1 for i in range(n_games*20)]
    plot_learning_curve(x, score_history, figure_file)
