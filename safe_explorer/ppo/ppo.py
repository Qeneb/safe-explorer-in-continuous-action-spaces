from datetime import datetime
from collections import namedtuple
from itertools import count

import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from safe_explorer.core.tensorboard import TensorBoard

# Parameters
gamma = 0.99
render = False
seed = 1
log_interval = 10


torch.cuda.set_device("cuda:3")
torch.manual_seed(seed)

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])


class Actor(nn.Module):
    def __init__(self, num_state, num_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.action_head = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, num_state):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.state_value = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value


class PPO:
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 1000
    batch_size = 32

    def __init__(self,
                 env,
                 num_state,
                 num_action,
                 action_modifier=None):
        super(PPO, self).__init__()
        self.env = env
        self.actor_net = Actor(num_state, num_action)
        self.critic_net = Critic(num_state)
        # Safe layer
        self.action_modifier = action_modifier

        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.writer = TensorBoard.get_writer()

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3)
        if not os.path.exists('../param'):
            os.makedirs('../param/net_param')
            os.makedirs('../param/img')

    def select_action(self, observation, state, c):

        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()

        # Notice: Data structure of action
        if self.action_modifier:
            modified_action = self.action_modifier(observation, action, c)
        else:
            modified_action = action.item()

        return modified_action, action_prob[:, action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(), '../param/net_param/actor_net' + str(time.time())[:10], +'.pkl')
        torch.save(self.critic_net.state_dict(), '../param/net_param/critic_net' + str(time.time())[:10], +'.pkl')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_ep):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        # reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        # next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)
        # print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 == 0:
                    print('I_ep {} ，train {} times'.format(i_ep, self.training_step))
                # with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index]).gather(1, action[index])  # new policy

                ratio = (action_prob / old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:]  # clear experience

    def train(self):

        start_time = time.time()

        print("==========================================================")
        print("Initializing PPO training...")
        print("----------------------------------------------------------")
        print(f"Start time: {datetime.fromtimestamp(start_time)}")
        print("==========================================================")

        for i_epoch in range(1000):
            observation = self.env.reset()

            agent_position = observation['agent_position']
            target_position = observation['target_postion']
            state = np.hstack((agent_position, target_position))

            c = self.env.get_constraint_values()
            # if render: self.env.render() TODO: render safe-explorer

            for t in count():
                action, action_prob = self.select_action(observation, state, c)
                next_observation, reward, done, _ = self.env.step(action)

                agent_position = next_observation['agent_position']
                target_position = next_observation['target_postion']
                next_state = np.hstack((agent_position, target_position))

                trans = Transition(state, action, action_prob, reward, next_state)
                # if render: self.env.render() TODO: render safe-explorer
                self.store_transition(trans)

                observation = next_observation
                state = next_state
                c = self.env.get_constraint_values()

                # done: dead or finished
                if done:
                    if len(self.buffer) >= self.batch_size:
                        self.update(i_epoch)
                    self.writer.add_scalar('liveTime/livestep', t, global_step=i_epoch)
                    break

        print("==========================================================")
        print(f"Finished PPO training. Time spent: {(time.time() - start_time) // 1} secs")
        print("==========================================================")