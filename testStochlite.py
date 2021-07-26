# ### Stochlite Testing/Data Analysis Code
# Written by Tejas Rane (May, 2021)

import sys, os
import gym_sloped_terrain.envs.stochlite_pybullet_env as e
import utils.joystick as j
import argparse
from fabulous.color import blue,green,red,bold
import gym
import pybullet as p
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use( 'tkagg' )

PI = np.pi


if (__name__ == "__main__"):
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	
	parser.add_argument('--PolicyDir', help='directory of the policy to be tested', type=str, default='23July3')
	parser.add_argument('--FrontMass', help='mass to be added in the first', type=float, default=0)
	parser.add_argument('--BackMass', help='mass to be added in the back', type=float, default=0)
	parser.add_argument('--FrictionCoeff', help='foot friction value to be set', type=float, default=0.8)
	parser.add_argument('--WedgeIncline', help='wedge incline degree of the wedge', type=int, default=0)
	parser.add_argument('--WedgeOrientation', help='wedge orientation degree of the wedge', type=float, default=0)
	parser.add_argument('--MotorStrength', help='maximum motor Strength to be applied', type=float, default=7.0)
	parser.add_argument('--RandomTest', help='flag to sample test values randomly ', type=bool, default=False)
	parser.add_argument('--seed', help='seed for the random sampling', type=float, default=100)
	parser.add_argument('--EpisodeLength', help='number of gait steps of a episode', type=int, default=1000)
	parser.add_argument('--PerturbForce', help='perturbation force to applied perpendicular to the heading direction of the robot', type=float, default=0.0)
	parser.add_argument('--Downhill', help='should robot walk downhill?', type=bool, default=False)
	parser.add_argument('--Stairs', help='test on staircase', type=bool, default=False)
	parser.add_argument('--AddImuNoise', help='flag to add noise in IMU readings', type=bool, default=False)
	parser.add_argument('--VideoTitle', help='Title of the vioeo if recording', type=str, default='testVideo')
	parser.add_argument('--InputFunction', help='Command velocity function (st, mst, r, mr, sin, sip, rand)', type=str, default='rand')

	args = parser.parse_args()

	# Policy Selection and Tuning
	# policy = np.load("utils/zero_policy.npy") # Loading a zero policy
	# policy[12][7] = 1 # cmd Vx to aug Vx
	# policy[13][8] = 1 # cmd Vy to aug Vy
	# policy[14][9] = 1 # cmd Wz to aug Wz
	# policy = np.load("initial_policies/HT_IPSL_14Jn2.npy") # Loading an initial policy
	policy = np.load("experiments/"+args.PolicyDir+"/iterations/best_policy.npy") # Loading the best policy PolicyDir
	# for i in range(4): 
	# 	policy[i][3] = 0.01
	# 	policy[i+3][2] = 0.01
	# policy[13][8] = 1
	# policy[14][9] = 1
	# policy = np.load("experiments/"+args.PolicyDir+"/iterations/policy_21.npy") # Loading the selected policy PolicyDir
	# for i in range(0,12): policy[i][7:] = 0 # Policy Mask - shift actions made independent of cmd vel
	# policy = np.array([[0, 0.13, 0, 0.01, 0, 0, 0.52, 0, 0, 0],
	# 				   [0, 0.13, 0, 0.01, 0, 0, 0.52, 0, 0, 0],
	# 				   [0, 0.13, 0, 0.01, 0, 0, 0.52, 0, 0, 0],
	# 				   [0, 0.13, 0, 0.01, 0, 0, 0.52, 0, 0, 0],
	# 				   [-0.538, 0, 0.01, 0, 0, -2.152, 0, 0, 0, 0],
	# 				   [-0.538, 0, 0.01, 0, 0, -2.152, 0, 0, 0, 0],
	# 				   [-0.538, 0, 0.01, 0, 0, -2.152, 0, 0, 0, 0],
	# 				   [-0.538, 0, 0.01, 0, 0, -2.152, 0, 0, 0, 0],
	# 				   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	# 				   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	# 				   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	# 				   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	# 				   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	# 				   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	# 				   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) # Hand-tuned policy
	print(policy)

	WedgePresent = True
	iter = 1

	if(args.WedgeIncline == 0 or args.Stairs):
		WedgePresent = False
	elif(args.WedgeIncline <0):
		args.WedgeIncline = -1*args.WedgeIncline
		args.Downhill = True
	env = e.StochliteEnv(render=True, end_steps = args.EpisodeLength*iter, wedge=WedgePresent, stairs = args.Stairs, downhill= args.Downhill, seed_value=args.seed,
				      on_rack=False, gait = 'trot',IMU_Noise=args.AddImuNoise)

	# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

	# cam = p.getDebugVisualizerCamera()
	# print('camera', cam)
	# p.resetDebugVisualizerCamera(distance(in m), yaw(in deg), pitch(in deg), lookat(coords in m))
	# p.resetDebugVisualizerCamera(1.0, PI, PI/2, [1, 0, 0])

	#For recording videos, records from debug visualizer
	# p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, 'videos/'+args.VideoTitle+'.mp4')

	if(args.RandomTest):
		env.Set_Randomization(default=False)
	else:
		env.incline_deg = args.WedgeIncline
		env.incline_ori = math.radians(args.WedgeOrientation)
		env.SetFootFriction(args.FrictionCoeff)
		# env.SetLinkMass(0,args.FrontMass)
		# env.SetLinkMass(11,args.BackMass)
		env.clips = args.MotorStrength
		env.pertub_steps = 300
		env.y_f = args.PerturbForce
	obs = env.reset()

	# p.resetDebugVisualizerCamera(1.0, 90, PI/2, [1, 0, 1])

	print (
	bold(blue("\nTest Parameters:\n")),
	green('\nWedge Inclination:'),red(env.incline_deg),
	green('\nWedge Orientation:'),red(math.degrees(env.incline_ori)),
	green('\nCoeff. of friction:'),red(env.friction),
	# green('\nMass of the front half of the body:'),red(env.FrontMass),
	# green('\nMass of the rear half of the body:'),red(env.BackMass),
	green('\nMotor saturation torque:'),red(env.clips))

	# Simulation with fixed commands
	p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
	plot_data = []
	for i in range(iter):
		js = j.Joystick(args.EpisodeLength, 1)
		commands = js.get_commands(args.InputFunction)
		t_r = 0
		obs = env.reset()
		for i_step in range(args.EpisodeLength):
			# Fixed Actions
			# action = np.array( [-0.17, -0.17, -0.17, -0.17,
			#                     0.0, 0.0, 0.0, 0.0,
			# 					0.0, 0.0, 0.0, 0.0,
			# 					commands[i_step][0], commands[i_step][1], commands[i_step][2]])
			env.commands = commands[i_step]
			state = np.concatenate((obs, commands[i_step])).ravel()
			# print(state)
			# state[2:5] = 0 #Making roll_vel, pitch_vel, yaw_vel 0 
			action = policy.dot(state)
			# print('Action before env', action)
			# plot_data.append(action)
			# env.updateCommands(i_step, simstep)
			obs, r, _, angle = env.step(action)
			# env.render()
			# t_r += r
			plot_data.append(r)
			# print("Plot data", plot_data)
	
	# p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4, 'videos/'+args.VideoTitle+'.mp4')

	# Simulation with Joystick Emulation
	# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
	# cmd_vx = p.addUserDebugParameter('Joystick Vx', -1.0, 1.0, 0)
	# cmd_vy = p.addUserDebugParameter('Joystick Vy', -1.0, 1.0, 0)
	# cmd_wz = p.addUserDebugParameter('Joystick Wz', -1.0, 1.0, 0)
	# commands = [0, 0, 0]
	# plot_data = []

	# while (True):
	# 	commands[0] = p.readUserDebugParameter(cmd_vx)
	# 	commands[1] = p.readUserDebugParameter(cmd_vy)
	# 	commands[2] = p.readUserDebugParameter(cmd_wz)
		
	# 	state = np.concatenate((obs, commands)).ravel()
	# 	# state[2:5] = 0 #Making roll_vel, pitch_vel, yaw_vel 0 
	# 	action = policy.dot(state)
	# 	# print('Z shifts', action[8:12])
	# 	# plot_data.append(action)
	# 	# env.updateCommands(i_step, simstep)
	# 	obs, r, _, angle = env.step(action)
	# 	# env.render()
	# 	# t_r +=r
	# 	# plot_data.append(r)
	# 	# print("Plot data", plot_data)
	
	# p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4, 'videos/'+args.VideoTitle+'.mp4')

	# Reward Function Analysis - Plotting
	roll_r = [p[0] for p in plot_data]
	pitch_r = [p[1] for p in plot_data]
	height_r = [p[2] for p in plot_data]
	roll_vel_r = [p[3] for p in plot_data]
	pitch_vel_r = [p[4] for p in plot_data]
	cmd_lin_r = [p[5] for p in plot_data]
	cmd_ang_r = [p[6] for p in plot_data]
	sx_r = [p[7] for p in plot_data]
	sy_r = [p[8] for p in plot_data]
	power_r = [p[9] for p in plot_data]
	roll = [p[10]* 180/PI for p in plot_data] 
	pitch = [p[11]* 180/PI for p in plot_data] 
	cum_r = [p[12] for p in plot_data]

	for i in range(len(cum_r)): t_r = t_r + cum_r[i]
	print("Total Reward:", t_r)

	plt.plot(roll_r, label = "roll reward")
	plt.plot(pitch_r, label = "pitch reward")
	plt.plot(height_r, label = "height reward")
	plt.plot(roll_vel_r, label = "roll vel reward")
	plt.plot(pitch_vel_r, label = "pitch vel reward")
	plt.plot(cmd_lin_r, label = "cmd_lin reward")
	plt.plot(cmd_ang_r, label = "cmd_ang reward")
	# plt.plot(sx_r, label = "step_x reward")
	# plt.plot(sy_r, label = "step_y reward")
	plt.plot(power_r, label = "power consumed reward")
	plt.plot(roll, label = "SP roll")
	plt.plot(pitch, label = "SP pitch")
	plt.plot(cum_r, label = "Cumulative reward")
	plt.legend()
	plt.show()

	# PLotting cmd_vel vs robot vel
	# cmd_xvel = [p[0] for p in plot_data]
	# cmd_yvel = [p[1] for p in plot_data]
	# cmd_angvel = [p[2] for p in plot_data]
	# x_vel = [p[3] for p in plot_data]
	# y_vel = [p[4] for p in plot_data]
	# ang_vel = [p[5] for p in plot_data]
	# t_r = [p[6] for p in plot_data]
	# plt.figure(1)
	# plt.plot(cmd_xvel, label = "command vel x")
	# plt.plot(cmd_yvel, label = "command vel y")
	# plt.plot(cmd_angvel, label = "command vel ang")
	# plt.plot(x_vel, label = "x vel")
	# plt.plot(y_vel, label = "y vel")
	# plt.plot(ang_vel, label = "ang vel")
	# plt.legend()
	# plt.show()

	# Writing to policy header file
	# folder = input('Enter folder name, ')
	# file = input('Enter file name, ')
	# os.chdir('/home/stoch-lab/stochrl_linear_policies/linear_policy_vel/')
	# if os.path.isdir(folder) == False: os.mkdir(folder)
	# os.chdir(folder)
	# f = open(file+'.h', 'w')
	# f.write('const double policy[15][10] = {')
	# for i in range(15):
	# 	for j in range(10):
	# 		if j == 0: f.write('{')
	# 		f.write(str(policy[i][j]))
	# 		if j < 9: f.write(', ')
	# 	if i == 14: f.write('}};')
	# 	else: f.write('},\n')

	# Creating Initial Policy
	# file = input('Enter file name, ')
	# os.chdir('/home/stoch-lab/SlopedTerrainLinearPolicy/initial_policies/')
	# np.save(file, policy)

	# Creating Policy Mask
	# file = input('Enter file name, ')
	# os.chdir('/home/stoch-lab/SlopedTerrainLinearPolicy/utils/')
	# mask = np.zeros(policy.shape)
	# dir = 0
	# for i in range(15):
	# 	for j in range(10):
	# 		if policy[i][j] != 0.0:
	# 			mask[i][j] = 1
	# 			dir += 1
	# 		else:
	# 			mask[i][j] = 0
	# for k in range(4): 
	# 	mask[k][3] = 0
	# 	mask[k+4][2] = 0
	# 	dir -= 2
	# print('Directions, ', dir)
	# np.save(file, mask)
