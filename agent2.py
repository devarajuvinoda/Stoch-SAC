import numpy as np 
import os 
import tensorflow as tf 
from tensorflow.keras.optimizers import Adam 
from replay_buffer import ReplayBuffer  
from neural_networks import ActorNetwork, CriticNetwork, ValueNetwork

class Agent:

	def __init__(self, policy_lr=0.0003, q_fun_lr=0.0003, alpha_lr=0.0003,gamma=0.99, 
			env=None, input_dims=[8], n_actions=2, max_size=100000, tau=0.005, 
			layer1_dims=256, layer2_dims=256, batch_size=256, reward_scale=2, target_entropy=-20):

		self.gamma = gamma 
		self.tau = tau 
		self.batch_size = batch_size
		self.n_actions = n_actions 
		self.memory = ReplayBuffer(max_size, input_dims, self.n_actions)

		self.actor = ActorNetwork(max_action=env.action_space.high, layer1_dims=layer1_dims, 
								layer2_dims=layer2_dims, n_actions=n_actions, name='actor')
		self.critic_1 = CriticNetwork(n_actions=n_actions, layer1_dims=layer1_dims, 
									layer2_dims=layer2_dims, name='critic1')
		self.critic_2 = CriticNetwork(n_actions=n_actions, layer1_dims=layer1_dims, 
									layer2_dims=layer2_dims, name='critic2')
		self.target_1 = CriticNetwork(n_actions=n_actions, layer1_dims=layer1_dims, 
									layer2_dims=layer2_dims, name='target1')
		self.target_2 = CriticNetwork(n_actions=n_actions, layer1_dims=layer1_dims, 
									layer2_dims=layer2_dims, name='target2')
		
		self.actor.compile(optimizer=Adam(learning_rate=policy_lr))
		self.critic_1.compile(optimizer=Adam(learning_rate=q_fun_lr))
		self.critic_2.compile(optimizer=Adam(learning_rate=q_fun_lr))
		self.target_1.compile(optimizer=Adam(learning_rate=q_fun_lr))
		self.target_1.compile(optimizer=Adam(learning_rate=q_fun_lr))

		self.scale = reward_scale 
		self.target_entropy = target_entropy
		self.log_alpha = tf.Variable(0.0)
		self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.exp)
		self.alpha_optimizer = tf.optimizers.Adam(
		    self.alpha_lr, name='alpha_optimizer')
		self.update_network_parameters(tau=1)


	def update_network_parameters(self, target_value, q_value, tau=None):
		if tau is None:
			tau = self.tau 

		weights = []
		targets = target_value.weights
		for i, weight in enumerate(q_value.weights):
			weights.append(weight * tau + targets[i] * (1.0-tau))

		target_value.set_weights(weights)


	def choose_action(self, observation):
		state = tf.convert_to_tensor([observation])
		actions, _ = self.actor.sample_normal(state)

		return actions[0]

	def store_tuples(self, curr_state, action, reward, next_state, done):
		self.memory.store_transitions(curr_state, action, reward, next_state, done)


	def learn(self):
		if self.memory.mem_cnt < self.batch_size:
			return 

		curr_state, action, reward, next_state, done = self.memory.sample_transitions(
																		self.batch_size)
		curr_states = tf.convert_to_tensor(curr_state, dtype=tf.float32)
		actions = tf.convert_to_tensor(action, dtype=tf.float32)
		rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
		next_states = tf.convert_to_tensor(next_state, dtype=tf.float32)

		with tf.GradientTape(persistent=True) as tape:
			new_policy_actions, log_probs = self.actor.sample_normal(curr_states)
			log_probs = tf.squeeze(log_probs, 1)

			target_q1 = tf.squeeze(self.target_1(curr_states, new_policy_actions),1)
			target_q2 = tf.squeeze(self.target_2(curr_states, new_policy_actions),1)

			entropy_scale = tf.convert_to_tensor(self.alpha)
			target_val = tf.squeeze(
					tf.math.minimum(target_q1, target_q2),1) - entropy_scale * log_probs
			target_q = reward + (1-done)* self.gamma * target_val

			q1_old_policy = tf.squeeze(self.critic_1(curr_states, actions), 1)
			q2_old_policy = tf.squeeze(self.critic_2(curr_states, actions), 1)
			critic_1_loss = 0.5 * tf.keras.losses.MeanSquaredError(q1_old_policy, target_q)
			critic_2_loss = 0.5 * tf.keras.losses.MeanSquaredError(q2_old_policy, target_q)

		critic_1_network_gradients = tape.gradient(critic_1_loss, 
													self.critic_1.trainable_variables)
		critic_2_network_gradients = tape.gradient(critic_2_loss, 
													self.critic_2.trainable_variables)
		self.critic_1.optimizer.apply_gradients(zip(
						critic_1_network_gradients, self.critic_1.trainable_variables))
		self.critic_2.optimizer.apply_gradients(zip(
						critic_2_network_gradients, self.critic_2.trainable_variables))


		with tf.GradientTape() as tape:
			new_policy_actions, log_probs = self.actor.sample_normal(curr_states)
			log_probs = tf.squeeze(log_probs, 1)
			q1_new_policy = self.critic_1(curr_states, new_policy_actions)
			q2_new_policy = self.critic_2(curr_states, new_policy_actions)
			critic_value = tf.squeeze(tf.math.minimum(
										q1_new_policy, q2_new_policy), 1)

			entropy_scale = tf.convert_to_tensor(self.alpha)
			actor_loss = entropy_scale * log_probs - critic_value 
			actor_loss = tf.math.reduce_mean(actor_loss)
		actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_weights)
		self.actor.optimizer.apply_gradients(zip(
								actor_network_gradient, self.actor.trainable_variables))

		with tf.GradientTape() as tape:
			new_policy_actions, log_probs = self.actor.sample_normal(curr_states)
			log_probs = tf.squeeze(log_probs, 1)
			alpha_losses = -1.0 * (self.alpha * tf.stop_gradient(log_probs + 
																self.target_entropy))
			alpha_loss = tf.nn.compute_average_loss(alpha_losses)

		alpha_gradients = tape.gradient(alpha_loss, [self.log_alpha])
		self.alpha_optimizer.apply_gradients(zip(alpha_gradients, [self.log_alpha]))

		update_network_parameters(target_value=self.target_1, q_value=self.critic_1)
		update_network_parameters(target_value=self.target_2, q_value=self.critic_2)


	def save_models(self, episode_num):
	    print('saving models.........')
	    self.actor.save_weights(self.actor.chkpt_dir+str(episode_num))
	    self.critic_1.save_weights(self.critic_1.chkpt_dir+str(episode_num))
	    self.critic_2.save_weights(self.critic_2.chkpt_dir+str(episode_num))
	    self.value.save_weights(self.value.chkpt_dir+str(episode_num))
	    self.target_value.save_weights(self.target_value.chkpt_dir+str(episode_num))


	def load_models(self, episode_num):
	    print('loading models........')
	    self.actor.load_weights(self.actor.chkpt_dir+str(episode_num))
	    self.critic_1.load_weights(self.critic_1.chkpt_dir+str(episode_num))
	    self.critic_2.load_weights(self.critic_2.chkpt_dir+str(episode_num))
	    self.value.load_weights(self.value.chkpt_dir+str(episode_num))
	    self.target_value.load_weights(self.target_value.chkpt_dir+str(episode_num))
