import gym
import numpy as np
from plot_utils import plot_learning_curve
import pybullet_envs
import matplotlib.pyplot as plt
from gym import wrappers
import os

def get_mean(arr):
    mx = max(arr)
    mn = max(arr)
    if mn==mx:
        mx += 1
    arr = (2*(arr-mn)/(mx-mn)) - 1
    return np.mean(arr)

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


work_dir = mkdir('exp', 'simple_pol')
directory = mkdir(work_dir, 'monitor')

if __name__ == '__main__':
#     env = gym.make('LunarLanderContinuous-v2')
    #env = gym.make('Pendulum-v0')
    # env = gym.make('BipedalWalker-v2')

#     env = gym.make('InvertedPendulumBulletEnv-v0')
    env = gym.make('HalfCheetahBulletEnv-v0')
    env._max_episode_steps = 81
    env = gym.wrappers.Monitor(env, directory, video_callable=lambda episode_id: episode_id%20==0, force = True)

    file_name = 'HalfCheetahBulletEnv_simple_pol.png'
    figure_file = 'plots/' + file_name 
    best_score = env.reward_range[0]
    score_history = []
    
    print("env.action_space.high: ",env.action_space.high)
    print("env.action_space.low: ",env.action_space.low)
    
    np.random.seed(100)    
    init_policy = np.clip(np.random.randn(1,env.action_space.shape[0]), -1, 1)[0]
    print("initial policy: ",init_policy)
    epsilon = 0.01
    n_games = 201
    for i in range(n_games):
        delta = np.random.randint(-1,2, size=20)
        new_delta_policy = []
        pos_delta_score = np.array([])
        neg_delta_score = np.array([])
        zero_delta_score = np.array([])
        update_vals = np.array([])
        mx, mn = 10000, -10000
        for j in range(20):
            action = init_policy + delta[j]*epsilon
            action = np.clip(action, -1, 1)
            observation = env.reset()
            done = False
            score = 0 
#             step = 0
            while not done:
                observation_, reward, done, info = env.step(action)
#                 print("action: ",action)
#                 print("reward: ",reward)
#                 print("step: ",step)
#                 step += 1
                observation = observation_
                score += reward

            score_history.append(score)
            avg_score = np.mean(score_history[-10:])  
            print("avg_score: %.2f "%avg_score, " episode: ",i,", ",j, "th policy rollout: ")
            if delta[j]<0:
                neg_delta_score = np.append(neg_delta_score, score)
            if delta[j]>0:
                pos_delta_score = np.append(pos_delta_score, score)
            else:
                zero_delta_score = np.append(zero_delta_score, score)

        for j in range(len(init_policy)):
            neg_mean = get_mean(neg_delta_score)
            pos_mean = get_mean(pos_delta_score)
            zero_mean = get_mean(zero_delta_score)

            update_par = 0
            if zero_mean <= neg_mean or zero_mean <= pos_mean:
                update_par =  pos_mean - neg_mean

            update_vals = np.append(update_vals, update_par)
#         print(update_vals, update_vals.shape)
        update_vals = 0.1 * update_vals/len(update_vals)
#         print(update_vals, update_vals.shape)
        init_policy += update_vals
        init_policy = np.clip(init_policy, -1, 1)
        print("learned policy: ", init_policy)
        x = [j+1 for j in range((i+1)*20)]
        plot_learning_curve(x, score_history, figure_file)
        
    print("learned policy: ", init_policy)
    while not done:      
            observation_, reward, done, info = env.step(init_policy)
            print("action: ",action)
            print("reward: ",reward)

    x = [i+1 for i in range(n_games*20)]
    plot_learning_curve(x, score_history, figure_file)
