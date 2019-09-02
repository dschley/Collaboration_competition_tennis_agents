import random
from collections import namedtuple, deque
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import torch.optim as optim

seed = 42069
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

d4 = None

####################
# Hyper parameters:#
####################

# OUNoise hyperparams
mu = 0
theta = 0.075 #0.15
sigma = 0.075 #0.2

# Memory hyperparams
buffer_size = 100000
batch_size = 16
good_memories_enabled = True
good_memory_batch_representation = 0.25
good_memory_experience_lead = 20 # got some decent results around episode 3500 without this

# Update hyperparams
gamma = 0.99
tau = 0.01
alr = 1e-4
clr = 1e-3

'''
TODO: look into going back to single actor network/memory shared by all agents with new changes added to memory (dinky prioritized experience)
Also look into using real PER instead of dinky self made one
'''
class Agent():
    def __init__(self, state_size, action_size, numAgents):
        global d4
        if d4 == None:
            d4 = D4PGBrain(state_size, action_size, numAgents)
            
        self.d4 = d4  
        

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        action = self.d4.actor_local(state)
        action = action.detach().numpy()
        action = action + self.d4.noise_state
        action = action.clip(-1.,1.)

        return action
        
        
    def step(self, state, state_full, action, reward, next_state, next_state_full, done):
        self.d4.memory.add(state, state_full, action, reward, next_state, next_state_full, done)
        
    def save_actor_state(self, agentNum):
        torch.save(self.actor_local.state_dict(), 'actor_local_' + str(agentNum) + '.pth')
        torch.save(self.actor_target.state_dict(), 'actor_target_' + str(agentNum) + '.pth')
        
    def learn(self):
        self.d4.learn()

    
class MultiAgent():
    def __init__(self, state_size, action_size):
        self.d4 = D4PGBrain(state_size, action_size)
        
    def act(self, states):
        states = torch.FloatTensor(states)
        
        actions = self.d4.actor_local(states)
        actions = actions.detach().numpy()
        actions = actions + self.d4.noise_state
        
        return actions
    
    def step(self, states, actions, rewards, next_states, dones):
        self.d4.memory.addAll(states, actions, rewards, next_states, dones)

class D4PGBrain():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, state_size, action_size, numAgents):
        self.state_size = state_size
        self.action_size = action_size

        self.critic_local = Critic(self.state_size, self.action_size, numAgents).to(device)
        self.critic_target = Critic(self.state_size, self.action_size, numAgents).to(device)
        
        self.critic_target.load_state_dict(self.critic_local.state_dict())

        
        self.noise = OUNoise(self.action_size, mu, theta, sigma)
        self.noise_state = self.noise.sample()
        
        
        self.critic_loss_func = nn.MSELoss()
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=clr)
        
        
        self.memory = ReplayBuffer(buffer_size, batch_size)
                        
        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=alr)
        
        print('D4Brain created!')
        
    def save_state(self):
        torch.save(self.critic_local.state_dict(), 'critic_local.pth')
        torch.save(self.critic_target.state_dict(), 'critic_target.pth')
        torch.save(self.actor_local.state_dict(), 'actor_local.pth')
        torch.save(self.actor_target.state_dict(), 'actor_target.pth')
        
        
    def next_timestep(self):
        self.noise_state = self.noise.sample()
        
    def new_episode(self):
        self.noise.reset()
        
    def learn(self):
        experiences = self.memory.sample(batch_size)
        
        if experiences == None:
            print("not enough expeeeerience")
            return
        
        states, states_full, actions, rewards, next_states, next_states_full, _ = map(list, zip(*experiences))
        
        states = torch.FloatTensor(states)
        
        states_full = torch.FloatTensor(states_full)

        actions = torch.reshape(torch.FloatTensor(actions),(16,2,2))
                
        rewards = torch.FloatTensor(rewards).view(states.shape[0],1)
        
        next_states = torch.FloatTensor(next_states)
        
        next_states_full = torch.FloatTensor(next_states_full)

        
        
        
        #new critic loss with full obs
        Qvals = self.critic_local(states_full, actions)
        
        next_actions_primary = self.actor_target(next_states_full[:,0,:]).detach()
        next_actions_primary = torch.reshape(next_actions_primary, (next_actions_primary.shape[0],1,-1))
        next_actions_secondary = self.actor_target(next_states_full[:,1,:]).detach()
        next_actions_secondary = torch.reshape(next_actions_secondary, (next_actions_secondary.shape[0],1,-1))
        next_actions_full = torch.cat([next_actions_primary,next_actions_secondary],1)
        next_Q = self.critic_target(next_states_full, next_actions_full)
        Qprime = rewards + gamma * next_Q
        critic_loss = self.critic_loss_func(Qvals, Qprime.detach())
        
        '''
        # Old (solo agent) Critic loss
        Qvals = self.critic_local(states, actions)
        
        next_actions = self.actor_target(next_states)
        next_Q = self.critic_target(next_states, next_actions.detach())
        Qprime = rewards + gamma * next_Q
        critic_loss = self.critic_loss_func(Qvals, Qprime.detach())
        '''
        
        # New Actor loss
        actions_primary = self.actor_local(states_full[:,0,:])
        actions_primary = torch.reshape(actions_primary, (actions_primary.shape[0],1,-1))
        actions_secondary = self.actor_local(states_full[:,1,:]).detach()
        actions_secondary = torch.reshape(actions_secondary, (actions_secondary.shape[0],1,-1))

        actions_full = torch.cat([actions_primary,actions_secondary],1)
        actor_loss = -self.critic_local(states_full, actions_full).mean()
        
        '''
        # Old (solo agent) Actor loss
        actor_loss = -self.critic_local(states, self.actor_local(states)).mean()
        '''
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.soft_update(self.actor_local, self.actor_target)
        self.soft_update(self.critic_local, self.critic_target)
        
    def soft_update(self, local_model, target_model):
        for target_param, param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(param.data * tau + target_param.data * (1 - tau))

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, numAgents):
        """Initialize parameters and build model.
        
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        super(Critic, self).__init__()
#         self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        

        self.fc1 = nn.Linear(numAgents * (state_size + action_size), 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        
        torch.nn.init.kaiming_uniform(self.fc1.weight)
        torch.nn.init.kaiming_uniform(self.fc2.weight)
        torch.nn.init.kaiming_uniform(self.fc3.weight)
        # kaiming (he-et-al) is probably best for relu activations (and probably linear outputs)
        # while xavier is better for tanh and logistical activations

        
        '''
        last project's implementation:
        state input:
        fc:32:relu
        fc:64:relu
        
        action input:
        fc:32:relu
        fc:64:relu
        
        merge individual networks and follow up with:
        relu
        fc:1:natural
        '''

    def forward(self, state_full, actions):
        '''
        state_action = np.concatenate((state, action))
        state_action = torch.from_numpy(state_action).float().unsqueeze(0).to(device)
        '''
        
        state_action = torch.cat([state_full, actions], 2)
        state_action = torch.reshape(state_action,(state_action.shape[0],state_action.shape[2]*2))

        x = F.relu(self.fc1(state_action))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
        

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        super(Actor, self).__init__()
#         self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, action_size)
        
#         torch.nn.init.xavier_uniform(self.fc1.weight)
#         torch.nn.init.xavier_uniform(self.fc2.weight)
#         torch.nn.init.xavier_uniform(self.fc3.weight)
#         torch.nn.init.xavier_uniform(self.fc4.weight)
        
        torch.nn.init.kaiming_uniform(self.fc1.weight)
        torch.nn.init.kaiming_uniform(self.fc2.weight)
        torch.nn.init.kaiming_uniform(self.fc3.weight)
        torch.nn.init.xavier_uniform(self.fc4.weight)
        # kaiming (he-et-al) is probably best for relu activations (and probably linear outputs)
        # while xavier is better for tanh and logistical activations
        
        '''
        last project's implementation:
        fc:32:relu
        fc:64:relu
        fc:32:relu
        fc:out:sigmoid (which was multiplied by action range and added to action min)
        '''
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.tanh(self.fc4(x))
    

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, action_size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(action_size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
    
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "state_full", "action", "reward", "next_state", "next_state_full", "done"])
        
        # My own dinky version of "prioritized" experience replay.
        # basically, have a separate queue that only gets added to if the memory's reward is better than the average reward in the queue
        # when sampling memories, sample a small set from good memories then the rest from "general recent" memories
        # to prevent overfitting and overusing these good memories, random memories can enter the good memory queue every time the regular buffer does a "full refresh"
        # additionally, it would be helpful to keep learning from memories leading to this good action and reward so add some of the leading memories to the good memory buffer as well
        self.good_memories = deque(maxlen=(buffer_size))
        self.good_memory_avg = 0.
        self.memory_num = 0
        

    def add(self, state, state_full, action, reward, next_state, next_state_full, done):
        """Add a new experience to memory."""
        self.memory_num += 1
        e = self.experience(state, state_full, action, reward, next_state, next_state_full, done)
        self.memory.append(e)
        
        if good_memories_enabled and (reward >= self.good_memory_avg or self.memory_num % self.buffer_size == 0):
            if good_memory_experience_lead > 0 and reward >= self.good_memory_avg:
                self.good_memories.extend(list(self.memory)[-min(len(self.memory),good_memory_experience_lead):])
            else:
                self.good_memories.append(e)
            self.good_memory_avg = np.average([m.reward for m in self.good_memories])
        
    def addAll(self, states, actions, rewards, next_states, dones):
        exps = [self.experience(state, action, reward, next_state, done) for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones)]
        for exp in exps:
            self.memory.append(exp)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        if len(self.memory) >= self.batch_size: 
            if good_memories_enabled:
                good_memory_sample = random.sample(self.good_memories, k=min(len(self.good_memories), int(self.batch_size * good_memory_batch_representation)))
                random_memory_sample = random.sample(self.memory, k=(self.batch_size - len(good_memory_sample)))
                return good_memory_sample + random_memory_sample
            else:
                return random.sample(self.memory, k=self.batch_size)


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)