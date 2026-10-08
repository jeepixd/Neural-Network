from pickle import TRUE
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

env = gym.make("CartPole-v1")

#setting up matlab
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

# To ensure reproducibility during training, you can fix the random seeds
# by uncommenting the lines below. This makes the results consistent across
# runs, which is helpful for debugging or comparing different approaches.
#
# That said, allowing randomness can be beneficial in practice, as it lets
# the model explore different training trajectories.


# seed = 42
# random.seed(seed)
# torch.manual_seed(seed)
# env.reset(seed=seed)
# env.action_space.seed(seed)
# env.observation_space.seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)
Transition = namedtuple("Transition",
                        ("state", "action", "next_state", "reward"))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1= nn.Linear(n_observations, 128)
        self.layer2= nn.Linear(128, 128)
        self.layer3= nn.Linear(128, n_actions)

       # Called with either one element to determine next action, or a batch
       #  during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


#batch_size is the number of transitions sampled from the replay buffer
BATCH_SIZE = 128
#GAMMA is the discount factor as mentioned in the previous section 
GAMMA = 0.99
#EPS_START is the starting value of epsilon
EPS_START = 0.9
#EPS_END is the final value of epsilon
EPS_END = 0.01
#EPS_DECAY is the rate at which epsilon decays, HIGER MEANS SLOWER DECAY
EPS_DECAY = 2500
#TAU IS THE UPDATED RATE OF THE TARGET NETWORK 
TAU = 0.005
#LR IS THE LEARNING RATE 
LR = 3e-4

#GET NUMBER OF ACTION FROM GYM ACTION SPACE 
n_actions = env.action_space.n

#get number of state observations 
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr= LR, amsgrad =TRUE)
memory = ReplayMemory(10000)

steps_done =0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1 * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)

    else:
        return torch.tensor([[env.action_space.sample()]], device = device, dtype =torch.long)

episode_durations = []

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


#training loop

