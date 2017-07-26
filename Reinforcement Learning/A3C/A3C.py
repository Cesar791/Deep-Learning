import torch
import torch.multiprocessing as mp
from A3C_worker import Worker
from A3C_model import Net
import gym
import os
import argparse
from shared_optim import SharedAdam


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--workers', metavar='W', type=int, default = mp.cpu_count(),
                    help='Number of concurrent workers (default: 2)')
parser.add_argument('--max-episode-length', type=int, default=8e6,
					help='Number of episodes per worker (default: 8e6)')
parser.add_argument('--num-steps', metavar='t',type = int, default=20,
					help='Number of steps (t_max) before backpropagation (default: 20)')
parser.add_argument('--env-name', metavar='E', default='PongDeterministic-v4',
					help = 'Environment to train on (default: PongDeterministic-v4)')
parser.add_argument('--render', action = 'store_true',
					help = 'Render the training process (default: false)')
parser.add_argument('--beta1', metavar='b1', type = float, default = 0.9,
					help = 'Argument beta1 in Adam optimizer (Default = 0.9)')
parser.add_argument('--beta2', metavar='b2', type = float, default = 0.999,
					help = 'Argument beta2 in Adam optimizer (Default = 0.999)')

# Global counter 
class Counter(object):
    def __init__(self, initval=0):
        self.val = mp.Value('i', initval)
        self.lock = mp.Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value


# start main (training) program and initialize all workers
def main():
	os.environ['OMP_NUM_THREADS'] = '1'
	args = parser.parse_args()
	numProcesses = args.workers
	env = args.env_name
	processes = []
	envs = [gym.make(args.env_name) for i in range(numProcesses)]


	# Global network
	globalNetwork = Net(1, envs[0].action_space.n)
	actionSpace = envs[0].action_space.n
	globalNetwork.share_memory()
	print(globalNetwork)

	# Init Global counter
	globalCounter = Counter(0)

	# Init Shared Adam
	print('Initialize shared Adam optimizer')
	optimizer = SharedAdam(globalNetwork.parameters(), lr = 0.0001)
	optimizer.share_memory()
	

	for rank in range(numProcesses):
		p = mp.Process(target = Worker, args = (rank, envs[rank], globalNetwork, args, globalCounter, actionSpace, optimizer))
		p.start()
		processes.append(p)
	for p in processes:
		p.join()


if __name__ == '__main__':
	main()