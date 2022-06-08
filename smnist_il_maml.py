import numpy as np
import torch
import cv2
from datetime import datetime
import os
import sys
from Ndp import Ndp
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from Policies.CNN_ndp import CNNndp
from dmp.utils.smnist_loader import MatLoader, Separate
from MamlNDP import MetaNDP
from utils import load_data
import logging
import higher


def main(exp_cfg):
    np.random.seed(0)
    torch.manual_seed(0)
    device = exp_cfg['device']
    print(device)

    # --------------load-data---------------------------------------------
    digits = exp_cfg['digits']
    task_num = exp_cfg['task_num']
    images, or_tr, digits_inds = load_data(digits)
    data_points = 2000
    inds = np.arange(data_points)
    np.random.shuffle(inds)
    qry_inds = inds[1000:1400]
    spt_inds = inds[:1000]
    test_inds = inds[1400:1500]

    # lists contain data for each task
    X_spt, X_qry, X_test = [], [], []
    Y_spt, Y_qry, Y_test = [], [], []

    for k in range(task_num):
        X = torch.Tensor(images[digits_inds[k]]).float()
        t = torch.Tensor(np.array(or_tr)[:, :, :2]).float()[digits_inds[k]]
        Y = t[:, ::1, :]

        X_spt.append(X[spt_inds].to(device))
        X_qry.append(X[qry_inds].to(device))
        X_test.append(X[test_inds].to(device))
        Y_spt.append(Y[spt_inds].to(device))
        Y_qry.append(Y[qry_inds].to(device))
        Y_test.append(t[test_inds].to(device))

    # -----------------------------------------------------------------------

    # -----------Create storage file-----------------------------------------
    time = str(datetime.now())
    time = time.replace(' ', '_')
    time = time.replace(':', '_')
    time = time.replace('-', '_')
    time = time.replace('.', '_')
    model_save_path = './data/' + '_' + time
    exp_cfg['model_save_path'] = model_save_path
    os.mkdir(model_save_path)
    # ----------------------------------------------------------------------
    MAE = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')
    # -------------------initialize policies and optimizers------------------
    num_epochs = exp_cfg['n_outer_iter']
    batch_size = exp_cfg['batch_size']
    outer_lr = exp_cfg['outer_lr']
    k = 1
    T = 300 / k
    N = 30

    DNN = CNNndp(N=N, state_index=np.arange(2))
    ndpn = Ndp(DNN, T=T, l=1, N=N, state_index=np.arange(2))
    meta_optimizer = torch.optim.Adam(ndpn.parameters(), lr=outer_lr)
    ndpn.to(torch.device("cuda:0"))
    meta = MetaNDP(ndpn, exp_cfg).to(torch.device("cuda:0"))
    av_task_losses = []

    for outer_i in range(num_epochs):
        spt_inds = np.arange(X_spt[0].shape[0])
        qry_inds = np.arange(X_qry[0].shape[0])
        np.random.shuffle(spt_inds)

        for ind in np.split(spt_inds, len(spt_inds) // batch_size):

            np.random.shuffle(qry_inds)
            ind_q = qry_inds[:10]

            X_spt_batch, X_qry_batch = [], []
            Y_spt_batch, Y_qry_batch = [], []

            for k in range(task_num):
                X_spt_batch.append(X_spt[k][ind])
                Y_spt_batch.append(Y_spt[k][ind])
                X_qry_batch.append(X_qry[k][ind_q])
                Y_qry_batch.append(Y_qry[k][ind_q])

            task_loss = meta.higher_forward(X_spt_batch, Y_spt_batch, X_qry_batch, Y_qry_batch)
            meta_optimizer.zero_grad()
            task_loss.backward()
            av_task_losses.append(task_loss.item())
            meta_optimizer.step()

            LOG.info(
                f' [Epoch {outer_i:.2f}] Task loss: {task_loss.item():.2f}]| '
            )
        if outer_i % 5 == 0:
            torch.save(ndpn.state_dict(), 'maml.pth')

            for i in range(task_num):
                loss_ = meta.regular_train(X_test[i][:100], Y_test[i][:100], exp_cfg, "maml" + str(i))
                LOG.info(f' -------[Epoch {outer_i:.2f}] Test loss: {loss_:.2f}]| ')


if __name__ == "__main__":
    global LOG
    logging.basicConfig(level=logging.INFO)
    LOG = logging.getLogger(__name__)

    exp_cfg = {}
    exp_cfg['seed'] = 0
    exp_cfg['num_train_tasks'] = 1
    exp_cfg['num_test_tasks'] = 1
    exp_cfg['n_outer_iter'] = 1000
    exp_cfg['digits'] = [2, 3]
    exp_cfg['task_num'] = len(exp_cfg['digits'])
    exp_cfg['n_gradient_steps_at_test'] = 100
    exp_cfg['batch_size'] = 100
    exp_cfg['inner_lr'] = 1e-3
    exp_cfg['outer_lr'] = 1e-3
    exp_cfg['inner_itr'] = 5
    exp_cfg['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(exp_cfg)