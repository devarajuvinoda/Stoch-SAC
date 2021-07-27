import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense 
import os 
import numpy as np  


class ActorNetwork(tf.keras.Model):
	
	def __init__(self, max_action, layer1_dims = 256, layer2_dims = 256, 
			n_actions=2, name='actor', chkpt_dir='tmp/sac_stochlite_lin'):

		super().__init__() 
		self.layer1_dims = layer1_dims
		self.layer2_dims = layer2_dims
		self.n_actions = n_actions 
		self.chkpt_dir = os.path.join(chkpt_dir, name+'_sac')
		self.max_action = max_action
		self.noise = 1e-6 
		
		self.layer1 = Dense(self.layer1_dims, activation=None)
		self.layer2 = Dense(self.layer2_dims, activation=None)
		self.mu_layer = Dense(self.n_actions, activation=None)
		self.sigma_layer = Dense(self.n_actions, activation=None)


	def call(self, state):

		layer1_op = self.layer1(state)
		layer2_op = self.layer2(layer1_op)
		mu = self.mu_layer(layer2_op)
		sigma = self.sigma_layer(layer2_op)
		sigma = tf.clip_by_value(sigma, self.noise, 1)

		return mu, sigma 

	def sample_normal(self, state):
		mu, sigma = self.call(state)
		pdf = tfp.distributions.Normal(mu, sigma)
		actions = pdf.sample()

		action = tf.math.tanh(actions) * self.max_action 
		log_probs = pdf.log_prob(actions)
		log_probs -= tf.math.log(1 - tf.math.pow(action, 2) + self.noise)
		log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

		return action, log_probs


class CriticNetwork(tf.keras.Model):

	def __init__(self, n_actions, layer1_dims=256, layer2_dims=256,
			name='critic', chkpt_dir='tmp/sac_stochlite_lin'):

		super().__init__()

		self.layer1_dims = layer1_dims
		self.layer2_dims = layer2_dims
		self.n_actions = n_actions
		self.chkpt_dir = os.path.join(chkpt_dir, name+'_sac')
		
		self.layer1 = Dense(self.layer1_dims, activation='relu')
		self.layer2 = Dense(self.layer2_dims, activation='relu')
		self.layer3 = Dense(1, activation=None)

	def call(self, state, action):

		layer1_op = self.layer1(tf.concat([state, action], axis=1))
		layer2_op = self.layer2(layer1_op)
		q_value = self.layer3(layer2_op)

		return q_value 


class ValueNetwork(tf.keras.Model):

	def __init__(self, layer1_dims=256, layer2_dims=256, name='value',
			chkpt_dir='tmp/sac_stochlite_lin'):

		super().__init__()

		self.layer1_dims = layer1_dims
		self.layer2_dims = layer2_dims
		self.chkpt_dir = os.path.join(chkpt_dir, name+'_sac')

		self.layer1 = Dense(self.layer1_dims, activation='relu')
		self.layer2 = Dense(self.layer2_dims, activation='relu')
		self.layer3 = Dense(1, activation=None)

	def call(self, state):

		layer1_op = self.layer1(state)
		layer2_op = self.layer2(layer1_op)
		v_value = self.layer3(layer2_op)

		return v_value




