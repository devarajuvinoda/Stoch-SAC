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
    
    n_games = 201
    file_name = 'stochlite_plot_simple_pol.png'
    figure_file = 'plots/' + file_name 
    best_score = env.reward_range[0]
    score_history = []
    
    np.random.seed(n_games)    
    init_policy = np.clip(np.random.randn(1,15), -1, 1)[0]
    print("initial policy: ",init_policy)
    epsilon = 0.05
    load_chkpt = False
    if not load_chkpt:
        for i in range(n_games):        
            pos_delta_score = np.zeros(15)
            neg_delta_score = np.zeros(15)
            zero_delta_score = np.zeros(15)
            pos_delta_cnt = np.zeros(15)
            neg_delta_cnt = np.zeros(15)
            zero_delta_cnt = np.zeros(15)
            update_vals = []

            for j in range(20):
                delta = np.random.randint(-1,2, size=15)
                action = init_policy + delta*epsilon
                action = np.clip(action, -1.0, 1.0)
                observation = env.reset()
                done = False
                score = 0 
                itr = 0
                while not done:
                    itr += 1
                    observation_, reward, done, info = env.step(action)
#                     print(i, " : ", j, " : ", itr)
#                     print("action: ",action)
#                     print("reward: ",reward)
                    observation = observation_
                    score += reward

                if itr > 10:
                    score_history.append(score)
                    avg_score = np.mean(score_history[-10:])  
                    if best_score < avg_score:
                        best_score = avg_score
                        # with open('./tmp/saved_action_arr.npy', 'wb') as f:
                        np.save('./tmp/saved_action_arr.npy', action)
                    for k in range(15):
                        if delta[k]<0:
                            neg_delta_score[k] += score
                            neg_delta_cnt[k] += 1
                        if delta[k]>0:
                            pos_delta_score[k] += score
                            pos_delta_score[k] += 1
                        else:
                            zero_delta_score[k] += score
                            zero_delta_cnt[k] += 1
                            
            x = [k+1 for k in range((i+1)*20)]
            plot_learning_curve(x, score_history, figure_file)
            
            for j in range(15):
                neg_mean = neg_delta_score[j]/max(1, neg_delta_cnt[j])
                pos_mean = pos_delta_score[j]/max(1, pos_delta_cnt[j])
                zero_mean = zero_delta_score[j]/max(1, zero_delta_cnt[j])

                update_par = 0
                if not (zero_mean > neg_mean and zero_mean > pos_mean):
                    update_par =  pos_mean - neg_mean

                update_vals.append(update_par)
    #         print(update_vals, update_vals.shape)
            update_vals = np.array(update_vals)
            update_vals = 0.1 * (update_vals/np.linalg.norm(update_vals))
    #         print(update_vals, update_vals.shape)
            init_policy += update_vals
    #         init_policy = np.clip(init_policy, -1, 1)

            print("learned policy: ", init_policy)
        x = [i+1 for i in range(n_games*20)]
        plot_learning_curve(x, score_history, figure_file)

    else:
        # with open('./tmp/saved_action_arr.npy', 'rb') as f:
        action = np.load('./tmp/saved_action_arr.npy')
        print(action)
        done = False
        while not done:      
                observation_, reward, done, info = env.step(action)
    #             print("action: ",action)
    #             print("reward: ",reward)

