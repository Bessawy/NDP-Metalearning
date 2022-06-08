import numpy as np
import cv2
import torch.nn as nn
import torch
import torch.nn.functional as F
from dmp.utils.dmp_layer import DMPIntegrator
from dmp.utils.dmp_layer import DMPParameters

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

class Trjectory(torch.nn.Module):
    def __init__(self, n_actions=30, action_space=2, start_values=0.5):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_actions, action_space)* start_values)

    def forward(self, x):
        t = x[:,2:]
        t = t.type(torch.long).flatten()
        mean = self.weights[t,:]
        sigma = torch.tensor([1.0]).to(x.device)
        return mean, sigma

def binary(x, bits):
    mask = 2**torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


class CNNndpPolicySigma(nn.Module):
    def __init__(self, state_space=2, action_space=2,
                 N=3, T=5, l=1, tau=1,
                 state_index=np.arange(2), b_init_value=0.1,
                 rbf='gaussian', az=True,
                 only_g=False):
        '''
        Deep Neural Network for ndp
        :param N: No of basis functions (int)
        :param state_index: index of available states (np.array)
        :param hidden_activation: hidden layer activation function
        :param b_init_value: initial bias value
        '''

        super(CNNndpPolicySigma, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.T, self.N, self.l = T, N, l
        self.hidden = 64
        dt = 1.0 / (T * self.l)
        self.output_size = N * len(state_index) + len(state_index)
        self.ended_bits = 0

        self.state_index = state_index
        self.DMPparam = DMPParameters(N, tau, dt, len(state_index), None, a_z=az)
        self.func = DMPIntegrator(rbf=rbf, only_g=only_g, az=az)
        self.register_buffer('DMPp', self.DMPparam.data_tensor)
        self.register_buffer('param_grad', self.DMPparam.grad_tensor)

        self.fc1 = torch.nn.Linear(3 + self.ended_bits, self.hidden)
        fanin_init(self.fc1.weight)
        self.fc1.bias.data.fill_(b_init_value)

        self.fc2 = torch.nn.Linear(self.hidden, self.hidden)
        fanin_init(self.fc2.weight)
        self.fc2.bias.data.fill_(b_init_value)

        self.fc1_mean = torch.nn.Linear(self.hidden, self.hidden)
        fanin_init(self.fc1_mean.weight)
        self.fc1_mean.bias.data.fill_(b_init_value)

        # actor
        self.fc2_mean = init_(torch.nn.Linear(self.hidden, self.output_size))
        # self.fc2_mean = torch.nn.Linear(self.hidden, self.output_size)
        # fanin_init(self.fc2_mean.weight)
        # self.fc2_mean.bias.data.fill_(b_init_value)
        self.sigma = torch.nn.Linear(self.hidden, action_space)

        # critic
        self.fc1_value = torch.nn.Linear(3 + self.ended_bits, 3 * self.hidden)
        self.fc2_value = torch.nn.Linear(3 * self.hidden, 3 * self.hidden)
        self.fc3_value = torch.nn.Linear(3 * self.hidden, action_space)

        self.scale = torch.linspace(0.5, 1, 10)
        self.phi = torch.nn.Parameter(torch.ones(10, 2) * 0.5)
        self.set = torch.tensor([0, 0, 1, 2, 2, 3, 4, 5, 6, 7])

    def forward(self, state):

        t = torch.clone(state[:, 2:]) / 5
        #subtract = t - self.set[t.type(torch.long)].to(device=state.device)
        #time_bits = binary(t.type(torch.long), self.ended_bits).view(-1, self.ended_bits)
        state = state * 1e-1

        x = torch.clone(state.view(-1, 3))
        x[:, :2] = x[:, :2]
        # x[:, 2:] = (x[ :, 2:]- 2.5)*20.0
        # time = x[:, 2:]

        # = torch.cat((x, image), dim=1)
        # x = torch.cat((x, time_bits), dim=1)

        v = self.fc1_value(x)
        v = torch.tanh(v)

        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)

        y0 = state[:, :2]
        y0 = y0.reshape(-1, 1)[:, 0]
        dy0 = torch.zeros_like(y0) + 0.01

        # critic for T actions
        v = self.fc2_value(v)
        v = torch.tanh(v)
        value = self.fc3_value(v).repeat(1, self.T).view(-1, self.T, 2)
        value = torch.transpose(value, 1, 2)

        act = self.fc1_mean(x)
        actor = torch.tanh(act)
        actor = self.fc2_mean(actor)

        sigma = self.sigma(act)
        scale = self.scale[t.type(torch.long)]  # note change t
        scale = scale.to(state.device)

        # sigma = torch.exp(sigma)
        #sigma = torch.sigmoid(sigma)
        sigma = self.phi[t.type(torch.long)]
        sigma = sigma.view(-1, 2, 1)

        ndp_wg = actor
        # actions
        y, dy, ddy = self.func.forward(ndp_wg, self.DMPp, self.param_grad, None, y0, dy0)  # y = [200,301]
        y = y.view(state.shape[0], len(self.state_index), -1)
        y = y[:, :, ::self.l]
        a = y[:, :, 1:] - y[:, :, :-1]

        return a, sigma, value
