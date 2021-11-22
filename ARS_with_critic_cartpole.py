# ref: https://github.com/fschur/Evolutionary-Reinforcement-Learning-for-OpenAI-Gym
# ref: https://keras.io/examples/rl/actor_critic_cartpole/

import numpy as np
import random
import gym
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers 

"""
Implementation of Augmented-Random-Search with Critic.
"""
seed = 42
gamma = 0.99  # Discount factor for past rewards
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

num_inputs = 4
num_actions = 2
# num_hidden = 128
num_hidden = 64

inputs = layers.Input(shape=(num_inputs,))
hidden1 = layers.Dense(num_hidden, activation='relu')(inputs)
hidden2 = layers.Dense(num_hidden, activation='relu')(hidden1)
critic = layers.Dense(1)(hidden2)

model = keras.Model(inputs=inputs, outputs=[critic])
optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
critic_value_history = []
rewards_history = []
history_values = []
history_returns = []
running_reward = 0
episode_count = 0

def numpy_softmax(x, axis=None):
    x += eps
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def plot_learning_curve(x, scores, figure_file):
    moving_avg = np.zeros(len(scores))

    for i in range(len(moving_avg)):
        moving_avg[i] = np.mean(scores[max(0, i-100) : (i+1)])

    plt.plot(x, moving_avg)
    plt.title('score v/s #episodes')
    plt.savefig(figure_file)
    plt.close(figure_file)

def update_critic(policies_minus, policies_plus, direction, env, version, H, render, train, action_mode, mu, sigma, best_idx):
    """
    Performs one episode in the environment for every given weights/gene.
    Returns the rewards and for version V2 the encountered states.
    """
    rewards = []
    states = []
    critic_values = []

    for idx in range(1):
        gene = policies_plus
        # if direction[idx]<0:
        #     gene = policies_minus[idx]
        done = False
        state = env.reset()
        rewards_sum = 0
        critic_value_sum = 0
        step = 0
        with tf.GradientTape() as tape:
            while not done and step <= H:
                step += 1
                if version == "V2":
                    # In V2 the states/oberservations are normalized and saved
                    if train:
                        states.append(state)
                    state = np.matmul(np.diag(np.sqrt(sigma)), (state - mu))

                if action_mode == "continuous":
                    action = np.matmul(gene, state)
                else:
                    values = np.matmul(gene, state)
                    probs = numpy_softmax(values)
                    # print('probs:',probs)
                    action = np.random.choice(len(values), p=probs)

                if render:
                    env.render()

                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, 0)
                critic_value = model(state)
                # print('critic_value[0, 0]:',critic_value[0, 0].numpy())
                critic_value_history.append(critic_value[0, 0])

                state, reward, done, _ = env.step(action)
                rewards_sum += reward
                critic_value_sum += critic_value[0, 0]
                rewards_history.append(reward)

                if done:
                    break

            # Update running reward to check condition for solving
            # running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

            returns = []
            discounted_sum = 0
            for r in rewards_history[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)

            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()

            history = zip(critic_value_history, returns)
            critic_losses = []
            for value, ret in history:
                # At this point in history, the critic estimated that we would get a
                # total reward = `value` in the future. 

                # The critic must be updated so that it predicts a better estimate of
                # the future rewards.
                critic_losses.append(
                    huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                )

            # Backpropagation
            loss_value = sum(critic_losses)
            # loss_value = sum(critic_losses)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Clear the loss and reward history
            critic_value_history.clear()
            rewards_history.clear()


def compute_rewards_v1(genes, env, version, H, render, train, action_mode, mu, sigma):
    """
    Performs one episode in the environment for every given weights/gene.
    Returns the rewards and for version V2 the encountered states.
    """
    rewards = []
    states = []
    critic_values = []
    del history_values[:]
    del history_returns[:]

    for gene in genes:
        done = False
        state = env.reset()
        rewards_sum = 0
        critic_value_sum = 0
        step = 0
        while not done and step <= H:
            step += 1

            if version == "V2":
                # In V2 the states/oberservations are normalized and saved
                if train:
                    states.append(state)
                state = np.matmul(np.diag(np.sqrt(sigma)), (state - mu))

            if action_mode == "continuous":
                action = np.matmul(gene, state)
            else:

                values = np.matmul(gene, state)
                probs = numpy_softmax(values)
                # print('probs:',probs)
                action = np.random.choice(len(values), p=probs)

            if render:
                env.render()

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)
            critic_value = model(state)
            # print('critic_value[0, 0]:',critic_value[0, 0].numpy())
            critic_value_history.append(critic_value[0, 0].numpy())

            state, reward, done, _ = env.step(action)
            rewards_sum += reward
            critic_value_sum += critic_value[0, 0].numpy()
            rewards_history.append(reward)

            if done:
                break

        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history_returns.append(returns)
        # print('critic_value_history:',critic_value_history)
        history_values.append(critic_value_history.copy())
        # print('history_values:',history_values)
        # Clear the loss and reward history
        critic_value_history.clear()
        rewards_history.clear()

        rewards.append(rewards_sum)
        critic_values.append(critic_value_sum)
    rewards = np.array(rewards)
    critic_values = np.array(critic_values)

    if train:
        rewards = rewards / np.std(rewards)
        critic_values = (critic_values -np.mean(critic_values))/ (np.std(critic_values)+eps)
        return rewards, states, critic_values
    else:
        return rewards

def main(step_size=0.5, nu=0.5, H=np.inf, num_individuals=50, num_selected=50, measure_step=100, num_episodes=15000,
         render=False, env_name="CartPole-v1", version="V2", measure_repeat=100, seed=42):

    assert (version == "V1" or version == "V2"), "Possible versions are: V1, V2"
    assert (num_selected <= num_individuals), "'num_selected' must be smaller or equal 'num_individuals'"

    env = gym.make(env_name)
    state = env.reset()

    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)
    
    score_plt = []
    x_plt = []
    # check whether the environment has a continuous or discrete action space.
    if type(env.action_space) == gym.spaces.Discrete:
        action_mode = "discrete"
    elif type(env.action_space) == gym.spaces.Box:
        action_mode = "continuous"
    else:
        raise Exception("action space is not known")

    # Get number of actions for the discrete case and action dimension for the continuous case.
    if action_mode == "continuous":
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    state_dim = env.observation_space.shape[0]

    # Performance is measured with respect to the amount of episodes taken in the environment.
    # In every iteration "num_indiviuals" many episodes are performed. For that the variable
    # "measure_step" is adjusted.
    num_iter = num_episodes/(2*num_individuals)
    assert (float.is_integer(num_iter)), "'num_episodes' needs to be a multiple of 2*'num_individuals'"
    num_iter = int(num_iter)
    measure_step_adjusted = measure_step/(2*num_individuals)
    assert (float.is_integer(measure_step_adjusted)), "'measure_step' needs to be a multiple of 2*'num_individuals'"
    measure_step_adjusted = int(measure_step_adjusted)

    current_policy = np.zeros((action_dim, state_dim))
    current_policy_rewards = np.zeros((action_dim, state_dim))

    if version == "V2":
        mu = np.zeros(state_dim)
        sigma = np.ones(state_dim)
        total_num_states = 0
    else:
        mu = None
        sigma = None

    performance = []
    
    for iteration in range(num_iter):
            # Mutates the linear policy weights randomly in both directions.
            translations = np.random.normal(0, 1, (num_individuals, action_dim, state_dim))
            policies_plus = current_policy + nu*translations
            policies_minus = current_policy - nu*translations

            # compute the performance of the mutated policies
            rewards_plus, states_plus, critic_values_plus = compute_rewards_v1(policies_plus, env, version, H, render=False, train=True, action_mode=action_mode, mu=mu, sigma=sigma)
            rewards_minus, states_minus, critic_values_minus = compute_rewards_v1(policies_minus, env, version, H, render=False, train=True, action_mode=action_mode, mu=mu, sigma=sigma)

            # update the policy with the most important mutated policies (measured by relative reward)
            relative_rewards = rewards_plus - rewards_minus
            relative_critic_values = critic_values_plus - critic_values_minus
            
            best_idx_rewards = np.array(np.argpartition(relative_rewards, -num_selected)[-num_selected:])
            best_idx = np.array(np.argpartition(relative_critic_values, -num_selected)[-num_selected:])
            
            best_relative_rewards = relative_rewards[best_idx_rewards]
            best_relative_critic_values = relative_critic_values[best_idx]
            
            std_rewards = np.std(best_relative_rewards)
            std = np.std(best_relative_critic_values)
            
            direction_rewards = (np.expand_dims(np.expand_dims(best_relative_rewards, axis=1), axis=1)*translations[best_idx]).sum(0)            
            direction = (np.expand_dims(np.expand_dims(best_relative_critic_values, axis=1), axis=1)*translations[best_idx]).sum(0)
            print(np.sum((direction<0)==(direction_rewards<0)))
#             print('direction:',(step_size/(num_selected*std))*direction)
#             print('current_policy:',current_policy)
            current_policy += (step_size/(num_selected*std))*direction
            current_policy_rewards += (step_size/(num_selected*std_rewards))*direction_rewards
#             print('current_policy:',current_policy.shape)
#             print('current_policy:',current_policy)
            
            update_critic(current_policy, current_policy_rewards, direction, env, version, H, render=False, train=False, action_mode=action_mode, mu=mu, sigma=sigma, best_idx=best_idx)

            # computes running mean and standard deviation of the encountered states/observations
            if version == "V2":
                states = np.array(states_minus + states_plus)
                num_new_states = len(states)
                num_old = total_num_states
                total_num_states += num_new_states
                mean = states.mean(0)
                sqrt_diff = (mean-mu)**2
                mu = (num_new_states*mean + num_old*mu)/total_num_states
                sigma += states.std(0) + sqrt_diff*num_new_states*num_old  / total_num_states

            # After "measure_step" many episodes performed in the environment the performance is measured.
            if iteration % measure_step_adjusted == 0:
                # print("current_policy:",current_policy)
                mean_rewards = np.mean([compute_rewards_v1([current_policy], env, version, render=(i==1)*render, H=np.inf, train=False, action_mode=action_mode, mu=mu, sigma=sigma) for i in range(measure_repeat)])
                performance.append([iteration*2*num_individuals, mean_rewards])
                print("Episode: ", performance[-1][0])
                print("rewards: ", performance[-1][1])
                score_plt.append(performance[-1][1])
                x_plt.append(performance[-1][0])
    np.save("../current_policy_2.npy", current_policy)
    plot_learning_curve(x_plt, score_plt, 'ars_cartpole_22.png')

if __name__ == '__main__':
    main()