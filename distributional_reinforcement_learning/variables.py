import torch

from random import shuffle
from collections import deque

replay_size = 200
batch_size = 50
lr = 0.0001
eps = 0.20
eps_min = 0.05
priority_level = 5
update_freq = 25

vmax = 10.0
vmin = -10.0
nsup = 51
aspace = 3

dz = (vmax - vmin) / (nsup-1)
support = torch.linspace(vmin,vmax,nsup)

replay = deque(maxlen=replay_size) 

#Initialize DQN parameter vector
tot_params = 128*100 + 25*100 + aspace*25*51  
theta = torch.randn(tot_params)/10. 
theta.requires_grad=True
theta_2 = theta.detach().clone() 

losses = []
cum_rewards = [] 
renders = []
game_wins = []                      # games won tracker

