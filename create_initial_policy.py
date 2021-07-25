import numpy as np
import stoch_gym.envs.stoch2_pybullet_env as s
import stoch_gym.envs.HyQ_pybullet_env as h
import stoch_gym.envs.Laikago_pybullet_env as l
import stoch_gym.envs.stochlite_pybullet_env as sl
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

'''
convention for action values has changed
action[:4] -> step_length fr fl br bl
action[4:8] -> steer angle 
action[8:12] -> x_shift fr fl br bl
action[12:16] -> y_shift fr fl br bl
action[16:20] -> z_shift fr fl br bl

tuned_actions_Laikago = np.array([ [0.8,0.8,0.8,0.8,
									0.0,0.0,0.0,0.0,
									-1.0,-1.0,-1.0,-1.0,
									-0.8,-0.8,-0.8,-0.8,
									0.0, 0.0, 0.0, 0.0],

									[1.0,1.0,1.0,1.0,
									0.0,0.0,0.0,0.0,
									-1.0,-1.0,-1.0,-1.0,
									-0.5,-0.5,-0.5,-0.5,
									0.8, 0.8, 0.8, 0.8],

									[0.5, 0.5, 0.5, 0.5,
									0.0, 0.0, 0.0, 0.0,
									-1.0, -1.0, -1.0, -1.0,
									-0.5, -0.5, -0.5, -0.5,
									0.0, 0.0, 0.0, 0.0]])

tuned_actions_Stoch2= np.array([[0.5,0.5,0.5,0.5,
                        0,0,0,0,
                       -1,-1,-1,-1,
                       -1,-1,-1,-1,
                        0, 0, 0, 0],
					  [0.5,0.5,0.5,0.5,
                        0,0,0,0,
                       -1,-1,-1,-1,
                        0,0,0,0,
                        1, 1, 1, 1],
					  [0.5,0.5,0.5,0.5,
                        0,0,0,0,
                       -1,-1,-1,-1,
                       -0.5,-0.5,-0.5,-0.5,
                        0.5, 0.5, 0.5, 0.5]
					  ])

tuned_actions_HyQ=   np.array([[0.0,0.0,0.0,0.0,
                             -1.0,-1.0,-1.0,-1.0,
                             -1.0,-1.0,-1.0,-1.0,
                              0.0, 0.0, 0.0, 0.0],

					          [0.5,0.5,0.5,0.5,
                               0.0,0.0,0.0,0.0,
                              -1.0,-1.0,-1.0,-1.0,
                              -0.5,-0.5,-0.5,-0.5,
                               0.5, 0.5, 0.5, 0.5]])
'''
# give tuned actions for diversified dataset when working with slopes, tuned actions wont affect much in flat ground walking

tuned_actions_Stochlite = np.array([[-1.0, -1.0, -1.0, -1.0, 
										0.0, 0.0, 0.0, 0.0,
	                        			-0.02, -0.02, -0.02, -0.02,
		                    			0.3, 0.3, 0.3, 0.3,
		                    			0.0, 0.0, 0.0, 0.0],

									[-1.0, -1.0, -1.0, -1.0, 
										0.0, 0.0, 0.0, 0.0,
	                        			-0.04, -0.04, -0.04, -0.04,
		                    			0.3, 0.3, 0.3, 0.3,
		                    			0.0, 0.0, 0.0, 0.0],
										
									[-1.0, -1.0, -1.0, -1.0, 
										0.0, 0.0, 0.0, 0.0,
	                        			-0.05, -0.05, -0.05, -0.05,
		                    			0.3, 0.3, 0.3, 0.3,
		                    			0.0, 0.0, 0.0, 0.0]])


if (__name__ == '__main__'):

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--robotName',help='the robot to be trained for',type=str,default='Stochlite')
	parser.add_argument('--policyName', help='file name of the initial policy', type=str, default='IP_')
	args = parser.parse_args()	


	if(args.policyName == 'IP_'):
		args.policyName += args.robotName
	# NUmber of steps per episode
	num_of_steps = 1000

	# list that tracks the states and actions
	states = []
	actions = []
	do_supervised_learning = True

	if(args.robotName == 'Stochlite'):
		idx1 = [0, 2, 3]
		idx2 = [0]
		idx3 = [2]
		experiment_counter = 0
		env = sl.StochliteEnv(render=False, wedge = False, stairs = False,on_rack=False, gait = 'trot')
		for i in idx1:
			for j in idx2:
				for k in idx3:
					t_r = 0

					env.Set_Randomization(default=True,idx1 = i,idx2=j,idx3=k,idx0=0,idx11=0)
					cstate = env.reset()
					roll = 0
					pitch = 0
					
					for ii in np.arange(0,num_of_steps):
# 						env.updateCommands(ii, num_of_steps)
						cstate, r, _, info = env.step(tuned_actions_Stochlite[experiment_counter])
						t_r +=r
						states.append(cstate)
						actions.append(tuned_actions_Stochlite[experiment_counter])
					# experiment_counter = experiment_counter +1
					print("Returns of the experiment:",t_r)

	if(do_supervised_learning):
		model = LinearRegression(fit_intercept = False)
		states = np.array(states)
		actions = np.array(actions)

		#train
		print("Shape_X_Labels:",states.shape,"Shape_Y_Labels:",actions.shape)
		model.fit(states,actions)
		action_pred= model.predict(states)
		
		#test
		print('Mean squared error:', mean_squared_error(actions, action_pred))
		res = np.array(model.coef_)

		# res = res * 0.1
		# res[3:] = res[3:] * 0 # changing policy matix to smaller values

		np.save("./initial_policies/"+args.policyName+".npy", res)

		print(res)

	'''
	Conventions for all robots below this needs to be changed.

    Conventions used in StochLite -> Right Hand Rule, x: forwards, z: upwards

	if(args.robotName == 'Stoch2'):
		idx1 = [3]
		idx2 = [0,3,2]
		idx3 = [1]
		experiment_counter = 0
		env = s.Stoch2Env(render=True, wedge = True, stairs = False,on_rack=False, gait = 'trot')
		for i in idx1:
			for j in idx2:
				for k in idx3:
					t_r = 0

					env.Set_Randomization(default=True,idx1 = i,idx2=j,idx3=k,idx0=0,idx11=0)
					cstate = env.reset()
					roll = 0
					pitch = 0
					
					for ii in np.arange(0,num_of_steps):
						cstate, r, _, info = env.step(tuned_actions_Stoch2[experiment_counter])
						t_r +=r
						states.append(cstate)
						actions.append(tuned_actions_Stoch2[experiment_counter])
					experiment_counter = experiment_counter +1
					print("Returns of the experiment:",t_r)

	if(args.robotName == 'Laikago'):
		#for Laikago	
		idx1 = [3,0]
		idx2 = [0,3]
		env = l.LaikagoEnv(render=True, wedge = True, stairs = False,on_rack=False)

		experiment_counter = 0
	
		for i in idx1:
			for j in idx2:
				if(i == 0 and j==3):
					break
				t_r = 0
				env.randomize_only_inclines(default=True, idx1=i, idx2=j)

				cstate = env.reset()
				roll = 0
				pitch = 0

				for ii in np.arange(0,num_of_steps):
					cstate, r, _, info = env.step(tuned_actions_Laikago[experiment_counter])
					t_r +=r
					states.append(cstate)
					actions.append(tuned_actions_Laikago[experiment_counter])
				experiment_counter = experiment_counter +1
				print("Returns of the experiment:",t_r)

	if(args.robotName == 'HyQ'):
		#for HyQ
		idx1 = [2]
		idx2 = [0,3]
		env = h.HyQEnv(render=True, wedge = True, stairs = False,on_rack=False)

		experiment_counter = 0
	
		for i in idx1:
			for j in idx2:
				if(i == 0 and j==3):
					break
				t_r = 0
				env.randomize_only_inclines(default=True, idx1=i, idx2=j)

				cstate = env.reset()
				roll = 0
				pitch = 0

				for ii in np.arange(0,num_of_steps):
					cstate, r, _, info = env.step(tuned_actions_Laikago[experiment_counter])
					t_r +=r
					states.append(cstate)
					actions.append(tuned_actions_Laikago[experiment_counter])
				experiment_counter = experiment_counter +1
				print("Returns of the experiment:",t_r)
	'''