import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np
from Ndp import Ndp
from Policies.CNN_ndp import CNNndp
import sys
import higher
import matplotlib.pyplot as plt

class MetaNDP(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, model, exp_cfg):
        """
        :param args:
        """
        super(MetaNDP, self).__init__()

        self.update_lr = exp_cfg['inner_lr']
        self.update_step = exp_cfg['inner_itr']
        self.task_num = exp_cfg['task_num']
        self.net = model
        self.loss = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.update_lr)

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        losses_q = []
        for i in range(self.task_num):
            # 1. run the i-th task and compute loss for k=0
            yp = self.net.forward(x_spt[i], y_spt[i][:, 0, :])
            loss = self.loss(yp, y_spt[i][:])
            grad = torch.autograd.grad(loss, self.net.parameters())
            new_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                yp = self.net.parameterised(x_spt[i], y_spt[i][:, 0, :], new_weights)
                loss = self.loss(yp, y_spt[i][:])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, new_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                new_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, new_weights)))

            yp = self.net.parameterised(x_qry[i], y_qry[i][:, 0, :], new_weights)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            losses_q.append(self.loss(yp, y_qry[i][:]))

        return sum(losses_q)/len(losses_q)

    def higher_forward(self, x_spt, y_spt, x_qry, y_qry):
        losses_q = []
        for i in range(self.task_num):
            with higher.innerloop_ctx(self.net, self.optimizer,
                             copy_initial_weights=False) as (fmodel, diffopt):

                for k in range(self.update_step):
                    yp = fmodel(x_spt[i], y_spt[i][:, 0, :])
                    pred_loss = self.loss(yp, y_spt[i][:])
                    diffopt.step(pred_loss)

                yp = fmodel(x_qry[i], y_qry[i][:, 0, :])
                task_loss = self.loss(yp, y_qry[i][:])
                losses_q.append(task_loss)


        return sum(losses_q)/len(losses_q)


    def regular_train(self, X, Y, exp_cfg, name='maml'):
        loss_trace = []
        eval_loss = 0
        with higher.innerloop_ctx(self.net, self.optimizer,
                                  copy_initial_weights=False) as (fmodel, diffopt):

            for k in range(self.update_step):
                yp = fmodel(X, Y[:, 0, :])
                pred_loss = self.loss(yp, Y[:])
                diffopt.step(pred_loss)

            yp = fmodel(X, Y[:, 0, :])
            eval_loss = self.loss(yp, Y[:]).detach()

            for k in range(5):
                plt.plot(0.667 * yp[k, :, 0].detach().cpu().numpy(), -0.667 * yp[k, :, 1].detach().cpu().numpy(),
                         c='r', linewidth=5)
                plt.axis('off')
                plt.savefig(exp_cfg['model_save_path'] + '/test_img_' + str(k) + "_" + name + '.png')
                plt.clf()

                img = X[k].cpu().numpy() * 255
                img = np.asarray(img * 255, dtype=np.uint8)
                plt.imshow(img, cmap='gray')
                plt.axis('off')
                plt.savefig(exp_cfg['model_save_path'] + '/ground_test_img_' + str(k) + "_" + name + '.png',
                            bbox_inches='tight', pad_inches=0)
                plt.clf()

        return eval_loss.item()
