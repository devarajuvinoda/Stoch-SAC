# implementation of replay buffer class
# replay buffer contains collection of experience tuples (S, A, R, S')
# tuples are added to buffer gradually as the agent interacts with the environment

import numpy as np

class ReplayBuffer:
	def __init__(self, max_size, input_shape, n_actions):
		self.mem_size = max_size
		self.mem_cnt = 0

		self.curr_state_mem = np.zeros((self.mem_size, *input_shape))
		self.action_mem = np.zeros((self.mem_size, n_actions))
		self.reward_mem = np.zeros(self.mem_size)
		self.next_state_mem = np.zeros((self.mem_size, *input_shape))
		self.done_flag = np.zeros(self.mem_size, dtype = bool)

	def store_transitions(self, curr_state, action, reward, next_state, done):
		self.mem_cnt += 1
		id = self.mem_cnt % self.mem_size

		self.curr_state_mem[id] = curr_state 
		self.action_mem[id] = action
		self.reward_mem[id] = reward 
		self.next_state_mem[id] = next_state 
		self.done_flag[id] = done 

	def sample_transitions(self, batch_size):
		max_avail_mem = min(self.mem_cnt, self.mem_size)

		# Generate a uniform random sample from 'np.arange(max_avail_mem)' of size 'batch_size'
		batch_ids = np.random.choice(max_avail_mem, batch_size)

		curr_states = self.curr_state_mem[batch_ids]
		actions = self.action_mem[batch_ids]
		rewards = self.reward_mem[batch_ids]
		next_states = self.next_state_mem[batch_ids]
		dones = self.done_flag[batch_ids]

		return curr_states, actions, rewards, next_states, dones