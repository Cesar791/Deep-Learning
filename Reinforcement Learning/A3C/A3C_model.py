import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn.init as init

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))#.expand_as(out))
    return out

# custom weight initialization
def weightinit(m):
	classname = m.__class__.__name__
	if isinstance(m, nn.Conv2d):
		init.xavier_normal(m.weight.data) # xavier initialization
	elif isinstance(m, nn.Linear):
		init.xavier_normal(m.weight.data) # xavier initialization

class Net(nn.Module):
	"""docstring for Net"""
	def __init__(self, input_dim, action_space):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(input_dim, 32, 3, stride = 2, padding = 1)
		self.conv2 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
		self.conv3 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
		self.conv4 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
		self.lstm = nn.LSTMCell(288, 256)
		self.policy = nn.Linear(256, action_space)
		self.value = nn.Linear(256, 1)

		self.apply(weightinit)
		self.policy.weight.data = normalized_columns_initializer(
            self.policy.weight.data, 0.01)
		self.policy.bias.data.fill_(0)
		self.value.weight.data = normalized_columns_initializer(
            self.value.weight.data, 1.0)
		self.value.bias.data.fill_(0)

		self.lstm.bias_ih.data.fill_(0)
		self.lstm.bias_hh.data.fill_(0)

	def forward(self, inputs):
		x, (hx, cx) = inputs
		x = F.elu(self.conv1(x))
		x = F.elu(self.conv2(x))
		x = F.elu(self.conv3(x))
		x = F.elu(self.conv4(x))
		x = x.view(-1, 288)
		hx, cx = self.lstm(x, (hx, cx))
		x = hx
		return F.softmax(self.policy(x)), self.value(x), (hx, cx)

	def local_hidden_state_init(self):
		return Variable(torch.zeros(1, 256)), Variable(torch.zeros(1, 256))
