import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger().setLevel(logging.DEBUG)


class PolicyModel(nn.Module):

    def __init__(self, num_states, num_actions, fc_units=(64, 64), gate=F.relu):
        super().__init__()
        input_shape = num_states
        self.gate = gate
        self.fcs = nn.ModuleList()
        for f in fc_units:
            self.fcs.append(nn.Linear(input_shape, f))
            input_shape = f
        self.fc_last = nn.Linear(input_shape, num_actions)

    def forward(self, state):
        x = state
        for fc in self.fcs:
            x = self.gate(fc(x))
        return torch.tanh(self.fc_last(x))


class ValueModel(nn.Module):

    def __init__(self, num_states, fc_units=(64, 64), gate=F.relu):
        super().__init__()
        input_shape = num_states
        self.gate = gate
        self.fcs = nn.ModuleList()
        for f in fc_units:
            self.fcs.append(nn.Linear(input_shape, f))
            input_shape = f
        self.fc_last = nn.Linear(input_shape, 1)

    def forward(self, state):
        x = state
        for fc in self.fcs:
            x = self.gate(fc(x))
        return torch.tanh(self.fc_last(x))


class ActorCriticModel(nn.Module):

    def __init__(self, num_states, num_actions, fc_units=(64, 64), gate=F.relu):
        super().__init__()
        self.actor = PolicyModel(num_states, num_actions, fc_units, gate)
        self.critic = ValueModel(num_states, fc_units, gate)
        self.std = torch.ones(1, num_actions)

    def forward(self, state):
        action = self.actor(state)
        value = self.critic(state)
        mean = torch.tanh(action)
        dist = Normal(mean, self.std)

        return dist, value

    def log_prob(self, dist, action):
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        return log_prob

    def save_model(self, filepath):
        logger.info("saving checkpoint to : %s" % filepath)
        torch.save(self.actor.state_dict(), "%s.actor.pth" % filepath)
        torch.save(self.critic.state_dict(), "%s.critic.pth" % filepath)

    def load_model(self, filepath):
        logger.info("loading checkpoint from : %s" % filepath)
        self.actor.load_state_dict(torch.load("%s.actor.pth" % filepath))
        self.critic.load_state_dict(torch.load("%s.critic.pth" % filepath))
