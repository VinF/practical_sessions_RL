import torch
import gym
import numpy as np

from random import shuffle

from collections import deque

from auxiliary import preproc_state
from dist_dqn import dist_dqn


def trainer(env, aspace, nsup, vmin, vmax, gamma, episodes, replay_size = 200, batch_size = 50, lr = 0.0001, eps = 0.20, eps_min = 0.05, priority_level = 5, update_freq = 25):
    
    """
    This function carries out the training procedure for the game Atari Freeway. It initialises the network, makes predictions, updates (using backpropagation) according to the received rewards and computes the losses. 
    
    env (OpenAI object OpenAI Gym): instantiated environment object
    aspace (int): Size of the action space.
    support (np.array): vector of supports.
    vmin (int): Minimum support.
    vmax (int): Maximum support.
    episodes (int): Number of episodes to run during training. 
    replay_size (int): Size when replayed. 
    batch_size (int): Size of the batch. 
    lr (float): Learning rate.
    eps (float): Epislon greedy at start.
    eps_min (float): Minimal epsilon greedy.
    priority_level (int): Priority level. 
    update_freq (int): Update frequency.
    
    Output:
    - theta (torch.Tensor): Network parameters.
    - losses (torch.Tesnor): Losses over time.
    
    """
    
    env = gym.make('Freeway-ram-v0') 
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
    state = preproc_state(env.reset())  # state preprocessing 
    game_wins = []                      # games won tracker

    for i in range(episodes):

        # Get initial prediction
        pred = dist_dqn(state,theta,aspace=aspace)

        # epsilon greedy action selection, random 
        if i < replay_size or np.random.rand(1) < eps: 
            action = np.random.randint(aspace)

        # if not random, select according to function
        else:
            action = get_action_dqn(pred.unsqueeze(dim=0).detach(),support).item()
        #env.reset()

        # take selected action 
        state2, reward, done, info = env.step(action) 
        state2 = preproc_state(state2)

        # examine reward structure of action
        if reward == 1: cum_rewards.append(1) 
        reward = 10 if reward == 1 else reward 
        reward = -10 if done else reward 
        reward = -1 if reward == 0 else reward 

        # store experience
        exp = (state,action,reward,state2) 
        replay.append(exp)

        # prioritised replay if reward == 10, i.e. if agent won
        if reward == 10: 
            for e in range(priority_level):
                replay.append(exp)
                game_wins.append(1)

        shuffle(replay)
        state = state2

        # if replay == 200, update values 
        if len(replay) == replay_size: 
            indx = np.random.randint(low=0,high=len(replay),size=batch_size)
            exps = [replay[j] for j in indx]
            state_batch = torch.stack([ex[0] for ex in exps],dim=1).squeeze()
            action_batch = torch.Tensor([ex[1] for ex in exps])
            reward_batch = torch.Tensor([ex[2] for ex in exps])
            state2_batch = torch.stack([ex[3] for ex in exps],dim=1).squeeze()
            pred_batch = dist_dqn(state_batch.detach(),theta,aspace=aspace)
            pred2_batch = dist_dqn(state2_batch.detach(),theta_2,aspace=aspace)
            target_dist = get_target_dist(pred2_batch,action_batch,reward_batch, \
                                         support, lim=(vmin,vmax),gamma=gamma)
            loss = lossfn(pred_batch,target_dist.detach())
            losses.append(loss.item())
            loss.backward()
            with torch.no_grad(): 
                theta -= lr * theta.grad
            theta.requires_grad = True

        if i % update_freq == 0: 
            theta_2 = theta.detach().clone()

        if i > 100 and eps > eps_min: 
            dec = 1./np.log2(i)
            dec /= 1e3
            eps -= dec

        # if game is done
        if done: 
            state = preproc_state(env.reset())
            done = False

        if i % 200 == 0:
            print(f"Episodes completed: {i}.")

            if len(game_wins) > 0:
                print(f"{str(len(game_wins))} games won!")
                
                
    return theta, losses

def get_target_dist(dist_batch,action_batch,reward_batch,support,lim=(-10,10),gamma=0.8):

    """Given a mini-batch of Q-value distribution predictions,
    this function produces the target distribution used in the loss
    function to update the DQN.
    
    Args: 
        dist_batch (torch.Tensor): Ax51xB where A is the action space size and B is batch size.
        action_batch (torch.Tensor): dim=B vector of integers in {0,1,2} of actions.
        reward_batch (torch.Tensor): dim=B vector of floats.
        lim (tuple): 2-tuple, indicates the lower and upper bound of the support.
        gamma (float): Discount factor. 

    Returns:
        target_dist_batch.
    """

    nsup = support.shape[0]
    vmin,vmax = lim[0],lim[1]
    dz = (vmax-vmin)/(nsup-1.)
    
    target_dist_batch = dist_batch.clone()

    for i in range(dist_batch.shape[0]): #loop through batch dimension
        dist_full = dist_batch[i]
        action = int(action_batch[i].item())
        dist = dist_full[action]
        r = reward_batch[i]
        if r != -1: #terminal state, target is degenerate distribuation at reward
            target_dist = torch.zeros(nsup)
            bj = np.round((r-vmin)/dz)
            bj = int(np.clip(bj,0,nsup-1))
            target_dist[bj] = 1.
        else:
            target_dist = update_dist(r,support,dist,vmin = lim[0],vmax=lim[1],gamma=gamma)
        target_dist_batch[i,action,:] = target_dist
        
    return target_dist_batch

def get_action_dqn(dist,support):
    """This function returns an integer action in [0,1,2].
    
    Args: 
        dist (torch.Tensor): Input is a Ax51xB discrete distribution over Q-values for each action where `A` is the action-space size, 
        and `B` is the batch dimension.
        support (torch.Tensor): Support vector.

    Returns:
        actions (int): vector of integers in {0,1,2}, dimension dist.shape[0] (batch size)
    """

    actions = []
    for b in range(dist.shape[0]): #loop through batch dimension
        expectations = [support @ dist[b,a,:] for a in range(dist.shape[1])]
        action = int(np.argmax(expectations))
        actions.append(action)
    actions = torch.Tensor(actions).int()
    return actions

def lossfn(x,y):
    
    """Function computes the Cross-Entropy Loss between prediction distribution `x` and target distribution `y`
    
    Args: 
        x (torch.Tensor): B x A x 51 tensor where `B` is batch dimension, `A` is action dimension
        y (torch.Tensor): A B x A x 51 tensor that caputres the target distribution. 

    Returns:
        loss (float): The loss between target and predicted distributions.
    """

    loss = torch.Tensor([0.])
    loss.requires_grad=True
    for i in range(x.shape[0]): #loop through batch dimension
        loss_ = -1 *  torch.log(x[i].flatten(start_dim=0)) @ y[i].flatten(start_dim=0)
        loss = loss + loss_
    return loss

def update_dist(observed_reward,support,probabilities, vmin, vmax, gamma=0.8):
    """Function that updates the input distribution in accordance with observed reward.

    Args:
        observed_reward (int): Reward observed after taking some action.
        support (torch.Tensor): Support vector.
        probabilities (torch.Tensor): Probability vector. 
        vmin (float): Lower bound of the support.
        vmax (float): Upper bound of the support.
        gamma (float): Discount factor. 

    Returns:
        posterior.
    """
    nsup = probabilities.shape[0]
    dz = (vmax-vmin)/(nsup-1.)
    bj = np.round((observed_reward-vmin)/dz) 
    bj = int(np.clip(bj,0,nsup-1)) 
    posterior = probabilities.clone()
    j = 1
    for i in range(bj,1,-1): 
        posterior[i] += np.power(gamma,j) * posterior[i-1]
        j += 1
    j = 1
    for i in range(bj,nsup-1,1): 
        posterior[i] += np.power(gamma,j) * posterior[i+1]
        j += 1
    posterior /= posterior.sum() 
    return posterior