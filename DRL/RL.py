import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

cuda = True if torch.cuda.is_available() else False
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action): #discrete_features,
        super(Actor, self).__init__()
        # self.discrete_features = discrete_features
        self.max_action = max_action

        self.l1 = nn.Linear(state_dim, 25)
        self.l2 = nn.Linear(25, 25)
        self.l3 = nn.Linear(25, action_dim)
        # self.l4 = nn.Linear(50, 50)
        # self.l5 = nn.Linear(50, 50)
        # self.l6 = nn.Linear(50, 50)
        # self.l7 = nn.Linear(50, 50)
        # self.l8 = nn.Linear(50, action_dim)


    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        # a = F.relu(self.l3(a))
        # a = F.relu(self.l4(a))
        # a = F.relu(self.l5(a))
        # a = F.relu(self.l6(a))
        # a = F.relu(self.l7(a))
        continuous_actions = self.max_action * F.relu(self.l3(a))
        return continuous_actions


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, discrete_ranges):
        super(Critic, self).__init__()
        self.discrete_ranges = discrete_ranges

        self.l1 = nn.Linear(state_dim+action_dim, 25)
        self.l2 = nn.Linear(25, 25)
        self.l3 = nn.Linear(25, 1)

        self.l4 = nn.Linear(state_dim+action_dim, 25)
        self.l5 = nn.Linear(25, 25)
        self.l6 = nn.Linear(25, 1)

        self.discrete_q = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(state_dim + action_dim, 25),
                nn.ReLU(),
                nn.Linear(25, 25),
                nn.ReLU(),
                nn.Linear(25, num_actions),
            ) for name, num_actions in discrete_ranges.items()
        })

    def forward(self, state, continuous_action):
        sa = torch.cat([state, continuous_action], 1)

        q1_cont = F.relu(self.l1(sa))
        q1_cont = F.relu(self.l2(q1_cont))
        q1_cont = self.l3(q1_cont)

        q2_cont = F.relu(self.l4(sa))
        q2_cont = F.relu(self.l5(q2_cont))
        q2_cont = self.l6(q2_cont)

        discrete_q_values = {
            name: self.discrete_q[name](sa)
            for name in self.discrete_ranges.keys()
        }

        return q1_cont, q2_cont, discrete_q_values

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1




class TD3(object):
    def __init__(self, state_dim, action_dim, discrete_features, max_action, discount=0.8, tau=0.005, policy_noise=0.2,
                 noise_clip=0.5, policy_freq=1):

        self.discrete_features = discrete_features

        self.actor = Actor(state_dim, action_dim, max_action).cuda() if cuda else (
            Actor(state_dim, action_dim, max_action))
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.00001)

        self.critic = Critic(state_dim, action_dim, discrete_features).cuda() if cuda else (
            Critic(state_dim, action_dim, discrete_features))
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.00001)

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0
        self.replay_buffer = utils.ReplayBuffer()


    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).cuda() if cuda else torch.FloatTensor(state)
        continuous_actions = self.actor(state_tensor)
        sa = torch.cat([state_tensor, (continuous_actions.cuda() if cuda else continuous_actions)], 1)
        discrete_actions = {}

        for name, num_actions in self.discrete_features.items():
            if random.random() < self.epsilon:
                discrete_actions[name] = random.randrange(num_actions)
            else:
                q_values = self.critic.discrete_q[name](sa)
                discrete_actions[name] = q_values.argmax().item()
        return continuous_actions, discrete_actions

    def train(self):
        self.total_it += 1

        state, continuous_action, discrete_action, next_state, reward, done, target = self.replay_buffer.sample()
        state = torch.FloatTensor(state).cuda() if cuda else torch.FloatTensor(state)
        continuous_action = torch.FloatTensor(continuous_action).cuda() if cuda else torch.FloatTensor(continuous_action)
        next_state = torch.FloatTensor(next_state).cuda() if cuda else torch.FloatTensor(next_state)
        reward = torch.FloatTensor(reward).cuda() if cuda else torch.FloatTensor(reward)
        done = torch.FloatTensor(done).reshape(-1, 1).cuda() if cuda else torch.FloatTensor(done).reshape(-1, 1)
        target = torch.FloatTensor(target).cuda() if cuda else torch.FloatTensor(target)

        with torch.no_grad():
            noise = (torch.randn_like(continuous_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            next_continuous_action = (self.actor_target(next_state) + noise).clamp(0, self.max_action)

            target_Q1, target_Q2, target_discrete_Q = self.critic_target(next_state, next_continuous_action)
            target_Q = torch.min(target_Q1, target_Q2)

            target_Q = reward + (1 - done) * 0.99 * target_Q

            discrete_targets = {}
            for name in self.discrete_features.keys():
                next_q_values = target_discrete_Q[name]
                next_q_value = next_q_values.max(dim=1, keepdim=True)[0]
                discrete_targets[name] = reward + (1 - done) * 0.99 * next_q_value

        current_Q1, current_Q2, current_discrete_Q = self.critic(state, continuous_action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        for name in self.discrete_features.keys():
            critic_loss += F.mse_loss(current_discrete_Q[name], discrete_targets[name])

        print("critic loss", critic_loss)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            # continuous_actions = self.actor(state)
            # actor_loss = -self.critic.q1(torch.cat([state, continuous_actions], dim=1)).mean()
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            print("actor_loss", actor_loss)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def store_transition(self, state, continuous_action, discrete_actions, next_state, reward, done, target):
        self.replay_buffer.add((state, continuous_action, discrete_actions, next_state, reward, done, target))
