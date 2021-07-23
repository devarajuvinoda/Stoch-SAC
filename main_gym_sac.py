import pybullet_envs
import gym
import numpy as np
from agent2 import Agent
from plot_utils import plot_learning_curve
from gym import wrappers

def run_model(env_name, folder_name):
        # env = gym.make('HalfCheetahBulletEnv-v0')
    print("env_name: ", env_name)
    env = gym.make(env_name)
    agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0])
    n_episodes = 1001
    # uncomment this line and do a mkdir tmp && mkdir tmp/video if you want to
    # record video of the agent playing the game.
    env = wrappers.Monitor(env, folder_name, video_callable=lambda episode_id: episode_id%20==0, force=True)
    filename = env_name

    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_chkpt = True

    if load_chkpt:
        agent.load_models(1000)
        env.render(mode='human')

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

#         if avg_score > best_score:
#             best_score = avg_score
#             if not load_chkpt:
#                 agent.save_models()

        if i%5==0 and not load_chkpt:
            agent.save_models(i)

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_chkpt:
        x = [i+1 for i in range(n_episodes)]
        plot_learning_curve(x, score_history, figure_file)


if __name__ == '__main__':
    run_model('InvertedPendulumBulletEnv-v0', 'tmp/video_iv')
    run_model('Walker2DBulletEnv-v0', 'tmp/video_wal')    
    run_model('HumanoidBulletEnv-v0', 'tmp/video_huma')
    
    # run_model('HalfCheetahBulletEnv-v0', 'tmp/video_hc')
    # run_model('AntBulletEnv-v0', 'tmp/video_ant')
