# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

def fanin_init(tensor):
    '''
    :param tensor: torch.tensor
    used to initialize network parameters
    '''
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)

def init(module, weight_init, bias_init, gain=1):
    '''
      :param tensor: torch.tensor
      used to initialize network parameters
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1)
        if module.bias is not None:
            module.bias.data.fill_(0.001)

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def binary(x, bits):
    mask = 2**torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


class ML3_NNloss(nn.Module):
    def __init__(self, in_dim, hidden_dim, inner_loop):
        super(ML3_NNloss, self).__init__()

        self.fc1 = nn.ModuleList([])
        self.fc2 = nn.ModuleList([])
        self.output = nn.ModuleList([])

        for i in range(inner_loop):
            fc1 = nn.Linear(in_dim * 2, hidden_dim[0])
            self.fc1.append(fc1)
            fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
            self.fc2.append(fc2)
            output = nn.Linear(hidden_dim[1], 1)
            self.output.append(output)

        self.hidden_activation = nn.ELU()
        self.last = nn.Softplus()


    def forward(self, y_in, y_target, inner_loop):
        '''
        :param y_in: shope(batch, 301, 2)
        :param y_target: shope(batch, 301, 2)
        :return: loss
        '''
        y = torch.abs(y_in - y_target)*1e-1
        #y = torch.mean(y, dim=2)
        y = y.reshape(-1, 301*2)

        y = self.fc1[inner_loop](y)
        y = self.hidden_activation(y)
        y = self.fc2[inner_loop](y)
        y = self.hidden_activation(y)
        y = self.output[inner_loop](y)
        return y.mean()

    def parameterised(self, y_in, y_target, weights):

        y = torch.abs(y_in - y_target)
        y = torch.sum(y, dim=2)

        y = F.relu(F.linear(y, weights[0], weights[1]))
        y = F.relu(F.linear(y, weights[2], weights[3]))
        y = F.linear(y, weights[4], weights[5])
        return y.mean()


    def reset(self):
        for m in self.modules():
            weight_init(m)

        nn.init.uniform_(self.output.weight, a=0.0, b=1)

class ML3_StructuredLoss(nn.Module):
    def __init__(self, inner_loops):
        super(ML3_StructuredLoss, self).__init__()
        self.hidden_activation =  F.relu
        self.inner = inner_loops
        self.phi = torch.nn.Parameter(torch.ones(self.inner, 301, 2))
        self.eye = torch.eye(self.inner)

    def forward(self, y_in, y_target, inner_loop):
        '''
        :param y_in: shope(batch, 301, 2)
        :param y_target: shope(batch, 301, 2)
        :return: loss
        '''

        weights = self.phi[inner_loop].reshape(1, 301, 2).repeat(y_in.shape[0], 1, 1)
        y = torch.abs(y_in - y_target) * weights
        y = torch.mean(y, dim=2)
        return y.mean()



class LearnedLossWeightedMse(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(LearnedLossWeightedMse, self).__init__()
        self.hidden_activation = F.relu
        self.fc1 = nn.Linear(301, 20)
        self.fc2 = nn.Linear(20, 20)
        self.output = nn.Linear(20, 301)
        self.out_dim = out_dim

        self.reset()
    def forward(self, y_in, y_target):
        err = torch.abs(y_in - y_target)
        l = torch.zeros_like(err[:,0])

        for i in range(self.out_dim):
            l+= self.phi[i] * err[:,i]
        return l.mean()

    def reset(self):
        self.phi = torch.nn.Parameter(torch.ones(self.out_dim))


    def get_parameters(self):
        return self.phi


class StructuredLossMFRLNN(nn.Module):
    def __init__(self, out_dim, batch_size):
        super(StructuredLossMFRLNN, self).__init__()

        self.out_dim = out_dim
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self.phi = torch.nn.Parameter(torch.ones(self.out_dim))

    def forward(self, state, mean, sig, action):
        dists = torch.distributions.Normal(mean, sig)
        logprob = dists.log_prob(action)
        rew_1 = (torch.sum(state, axis=1))
        rew_2 = (torch.sum(action, axis=1))


        # https://discuss.pytorch.org/t/repeat-a-nn-parameter-for-efficient-computation/25659/2
        # should hold and share gradients
        phi = self.phi.repeat(self.batch_size).view(-1, 1)
        phi2 = self.phi2.repeat(self.batch_size).view(-1, 1)

        l = (phi * rew_1 + phi2 * rew_2)
        selected_logprobs = l * logprob.sum(dim=-1)
        return -selected_logprobs.mean()


class DeepNNsigmalearning(nn.Module):
    def __init__(self, max_std=5, hidden=64):
        super(DeepNNsigmalearning, self).__init__()
        self.ended_bits = 4
        self.features = 3 * 3 * 5
        self.activation = nn.ReLU()

        self.fc1 = nn.Linear(3, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

        self.conv1 = nn.Conv2d(1, 5, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=1)

        self.final_layer = nn.Linear(hidden, 2)
        self.max_std = max_std
        self.activation = nn.ELU()

        self.scale = torch.linspace(0.5, 1, 10)
        self.phi = torch.nn.Parameter(torch.ones(10,2)*0.5)


    def forward(self, state, obs_i):
        t = torch.clone(state[:, 2:]) / 5
        #im = obs_i.view(-1,1,5,5)

        x = state*1e-1

        #im = F.relu(self.conv1(im))
        #im = F.relu(self.conv2(im))
       # y = im.view(-1, self.features)

        #x = torch.cat((y, x), dim=1)

        y = self.fc1(x)
        y = torch.tanh(y)
        y = self.fc2(y)
        y = torch.tanh(y)
        y = self.final_layer(y)
        std = torch.sigmoid(y)*0.5 + 0.0001

        #std = self.phi[t.type(torch.long)]
        return std.view(-1, 2, 1)

class PPORewardsLearning(nn.Module):
    def __init__(self, hidden=200):
        super(PPORewardsLearning, self).__init__()

        self.size = 50
        self.conv1 = nn.Conv2d(2, 10, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1)
        self.ended_bits = 5
        self.features = 5*5*10

        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(self.features + 1, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, 1)

        self.phi = torch.nn.Parameter(torch.ones(10, 5))


    def forward(self, obs, id):
        im = obs.view(-1, 2, 5, 5)
        id = id.view(-1, 1)/10
        #time_bits = binary(id.type(torch.long), self.ended_bits).view(-1, self.ended_bits).view(-1, 5)

        im = F.relu(self.conv1(im))
        im = F.relu(self.conv2(im))

        y = im.view(-1,  self.features)
        y = torch.cat((y, id), dim=1)

        x = self.fc1(y)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        reward = self.output(x) # shape(workers*workerr_steps*5, 1)

        #factor = self.phi[t.type(torch.long), :].view(-1,5)
        #reward = image * factor

        reward = reward.reshape(-1, 1, 5)
        reward = reward.repeat(1, 2, 1)

        return reward.view(-1, 2, 5)


