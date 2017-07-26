import torch
from A3C_model import Net
import gym
from torch.autograd import Variable
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

def discount(x, gamma):
	disList = torch.zeros(len(x))
	disList[len(x)-1] = x[-1]
	for i in reversed(range(len(x)-1)):
		disList[i] = gamma*disList[i+1] + x[i]
	return disList

# Helper function
# pre process frame
# resize to 42x42 and convert to grayscale (tensor) 
def processFrame(frame):
	# remove irrelevant parts of the frame
	# such as the score board at the top
	frame = frame[34:34+160, :160]
	frame = cv2.resize(frame, (80, 80))
	frame = cv2.resize(frame, (42, 42))
	frame = frame.mean(2)
	frame = frame.astype(np.float32)
	frame *= (1.0 / 255.0)
	#frame = np.ascontiguousarray(frame, dtype=np.float32) / 255.0 # convert  to float 0 - 1
	frame = np.reshape(frame, [1, 42, 42])
	frame = torch.from_numpy(frame) # convert to tensor
	return Variable(frame)

# sample action from multinomial distribution
def sampleAction(policy):
	sample = (policy[0] > 0).float()
	policy = policy[0]*sample
	return torch.multinomial(policy, 1).data.numpy()

def updateGlobalGradients(localNet, globalNet):
	for localParam, globalParam in zip(localNet.parameters(), globalNet.parameters()):
		if globalParam.grad is not None:
			return
		globalParam._grad = localParam.grad

# Calculates loss with Generalized Advantage Estimation (GAE)
# with lambda = 0, which means low variance but high bias
# GAE(gamma, lambda) = GAE(0.99, 0) = r_t + 0.99 * V(s_(t + 1)) - V(s_t)
# r_t is reward and V is value for a specfic frame (state)
# see equation (20) and (23), here https://arxiv.org/pdf/1506.02438.pdf
# R is the bootstrap value
# 
# Compute loss with loop
# def computeLoss(rewards, values, actions, policies, R):
# 	policy_loss = 0
# 	value_loss = 0
# 	discountAdvantage = torch.zeros(1,1)
# 	# this can be done more efficient by using vectorization
# 	for i in reversed(range(len(rewards))):
# 		# value loss
# 		R = rewards[i] + 0.99 * R
# 		value_loss += 0.5 * (R - values[i]).pow(2)
# 		# entropy loss
# 		entropy = -torch.sum(torch.log(policies[i])*policies[i])
# 		# GAE
# 		advantage = rewards[i] +  0.99*values[i+1].data - values[i].data
# 		discountAdvantage = 0.99 * discountAdvantage + advantage					
# 		# policy loss
# 		policy = policies[i]
# 		policy_loss = policy_loss - torch.log(policy[actions[i]])*Variable(discountAdvantage) - 0.01 * entropy
# 	return 0.5 * value_loss + policy_loss

# Compute loss with vectorization (faster)
def computeLoss(rewards, values, actions, policies, policy_action, R):
	policy_loss = 0
	value_loss = 0

	# Stack vectors
	values = torch.stack(values).squeeze(1)
	values_plus = torch.cat((values.data, R[0].data)) # concat bootstrap value R
	policy_action = torch.stack(policy_action).squeeze(1)
	policies = torch.stack(policies)

	# Rewards
	rewards = torch.FloatTensor(rewards)
	rewards_plus = torch.cat((rewards, R[0].data))
	discountedRewards = discount(rewards_plus, 0.99)[:-1]

	# GAE
	advantage = rewards +  0.99 * values_plus[1:] - values_plus[:-1]
	discountAdvantage = discount(advantage, 0.99)

	# Values loss
	value_loss = 0.5 * torch.sum((Variable(discountedRewards) - values).pow(2), 0)

	# entropy loss
	entropy = -torch.sum(torch.log(policies)*policies)
	
	# policy loss
	policy_loss = - torch.sum(torch.log(policy_action)*Variable(discountAdvantage))
	
	return (0.5 * value_loss + policy_loss - 0.01*entropy)


class Worker(object):
	"""docstring for Worker"""
	def __init__(self, rank, env, globalNet, args, globalCounter, actionSpace, optimizer = None):
		super(Worker, self).__init__()
		self.rank = rank + 1
		self.env = env
		self.globalNet = globalNet
		self.globalCounter = globalCounter
		self.actionSpace = actionSpace
		self.optimizer = optimizer
		
		torch.manual_seed(self.rank) # manual RNG seed
		
		# start train loop
		train(self.rank, self.env, self.globalNet, args, self.globalCounter, self.actionSpace, self.optimizer)
		

# Will contain the training loop
def train(rank, env, globalNet, args, globalCounter, actionSpace, optimizer):
	env.seed(rank)
	localNet = Net(1, actionSpace)

	if optimizer == None:
		optimizer = optim.Adam(globalNet.parameters(), lr = 0.0001)
		print('Creating Adam optimizer', rank)

	local_t = 0
	T_max = args.max_episode_length
	t_max = args.num_steps
	done = True # True in the beginning to initialize everything
	
	while True:
		rewards = []
		values = []
		actions = [] 
		policies = []
		policy_action = []

		# start
		localNet.load_state_dict(globalNet.state_dict()) # local copy of the global network
		t_start = local_t # episode counter

		if done:
			frame = env.reset() # reset if done
			score = 0
			hx, cx = localNet.local_hidden_state_init()
		else:
			hx = Variable(hx.data) # bootstrap from previous values
			cx = Variable(cx.data)

		while True:
			# perform a_t according to policy
			procFrame = processFrame(frame)
			policy, value, (hx, cx) = localNet((procFrame.unsqueeze(0), (hx, cx)))
			a_t = sampleAction(policy)
			
			# Render the step 
			if args.render:
				env.render()
				
			# perform action, get reward and next state
			frame, reward, done, _ = env.step(a_t)

			score += reward
			rewards.append(reward)
			values.append(value[0])
			actions.append(a_t[0])
			policies.append(policy[0])
			policy_action.append(policy[0, a_t[0]])


			local_t += 1
			globalCounter.increment()

			if done or ((local_t - t_start) == t_max):
				break

		if done:
			R = torch.zeros(1,1)
			print('Worker: %d Step: %d Score: %d' % (rank, globalCounter.value(), score)) 
		else:
			procFrame = processFrame(frame)
			_, value, _ = localNet((procFrame.unsqueeze(0), (hx, cx))) # bootstrap from last state
			R = value.data

		R = Variable(R)
		#values.append(value)

		# compute total loss
		loss = computeLoss(rewards, values, actions, policies, policy_action, R)

		# reset gradients
		optimizer.zero_grad()

		# compute gradients
		loss.backward()
		torch.nn.utils.clip_grad_norm(localNet.parameters(), 40)

		#update global network with gradients
		updateGlobalGradients(localNet, globalNet)
		optimizer.step()

		# stop training
		if globalCounter.value() >= T_max:
			break
	



		