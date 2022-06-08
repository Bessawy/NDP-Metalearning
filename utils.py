import numpy as np
import torch
import matplotlib.pyplot as plt
from visdom import Visdom
from smnistenv2 import Worker
from dmp.utils.smnist_loader import MatLoader, Separate
import cv2

def obs_to_torch(obs: np.ndarray, device):
    return torch.tensor(obs, dtype=torch.float32, device=device)

def pre_process(samples, device):
    samples_flat = {}
    for k, v in samples.items():
        v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
        samples_flat[k] = obs_to_torch(v, device)
    return samples_flat

def to_torch(samples, device):
    samples_torch = {}
    for k, v in samples.items():
        samples_torch[k] = obs_to_torch(v, device)
    return samples_torch


def load_data(desired_idx, data_path='./dmp/data/s-mnist/40x40-smnist.mat'):
    '''
    :param desired_idx: list of desired digits 0-9 (ex: [0, 1, 9])
    :param data_path: path to smnist dataset
    :return: list of desired digits.
    '''
    images, outputs, scale, or_tr = MatLoader.load_data(data_path, load_original_trajectories=True)
    images = np.array([cv2.resize(img, (28, 28)) for img in images]) / 255.0
    data_sep = Separate()
    digit_indices = data_sep.no_separation()

    digits_idx = []
    for k in range(len(desired_idx)):
        d = desired_idx[k]
        digit_idx = digit_indices[d].astype(int)
        digits_idx.append(digit_idx)

    return images, or_tr, digits_idx


def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("device: ", device)
    return device

def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()

def save_Tasklosses_plt(loss, inner_iter, range, name):
    plt.plot(loss)
    plt.legend(['Task losses'])
    plt.xlabel('ml3 updates')
    plt.ylabel('losses')
    plt.ylim(range)
    plt.title("Task losses during ml3 training for inner: " + str(iter))
    plt.savefig(name + '_losses.png')
    plt.clf()

def save_Taskrewards_plt(r1, r2, range, name):
    plt.plot(r1)
    plt.plot(r2)
    plt.legend(['ml3_reward', 'normal_reward'])
    plt.xlabel('iterations')
    plt.ylabel('reward')
    plt.ylim(range)
    plt.title("Task rewards")
    plt.savefig(name + '_rewards.png')
    plt.clf()

def store_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    weights = torch.load(path, map_location=device)
    model.load_state_dict(weights, strict=False)
    return "model_loaded"

def show_trajectory(task_goal, trajecx, trajecy, name):
    length = len(trajecx)

    R = [255, 255, 255, 51]
    G = [204, 128, 0, 0]
    B = [153, 0, 0, 0]

    Red = np.linspace(R[0], R[1], length // 2) / 255.0
    Green = np.linspace(G[0], G[1], length // 2) / 255.0
    Blue = np.linspace(B[0], B[1], length // 2) / 255.0

    Red2 = np.linspace(R[2], R[3], length // 2) / 255.0
    Green2 = np.linspace(G[2], G[3], length // 2) / 255.0
    Blue2 = np.linspace(B[2], B[3], length // 2) / 255.0

    Red = np.concatenate((Red, Red2))
    Green = np.concatenate((Green, Green2))
    Blue = np.concatenate((Blue, Blue2))

    x = np.array(trajecx) * 0.668
    y = np.array(trajecy) * 0.668

    fig, ax = plt.subplots()
    im = ax.imshow(task_goal)

    for i in range(length):
        color = (Red[i], Green[i], Blue[i])
        ax.plot(x[i], y[i], 'x', ls='dotted', linewidth=5, color=color)

    plt.savefig(name + '_digit.png')
    plt.clf()
    plt.close(fig)

class VisdomLinePlotter(object):
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

