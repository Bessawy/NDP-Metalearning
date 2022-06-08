import numpy as np
import torch
import cv2
from datetime import datetime
import os
from Ndp import Ndp
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from Policies.CNN_ndp import CNNndp
from dmp.utils.smnist_loader import MatLoader, Separate
from Policies.Learned_loss import ML3_NNloss, ML3_StructuredLoss
from MamlNDP import MetaNDP
from utils import load_data
import logging
import higher
import sys

def regular_train(loss_fn, eval_loss_fn, task_model, X, Y, exp_cfg, ml3=False):
    loss_trace = []
    lr = exp_cfg['inner_lr']
    inner_itr = exp_cfg['inner_itr']
    optimizer = torch.optim.SGD(task_model.parameters(), lr=lr)

    with higher.innerloop_ctx(task_model, optimizer,
                              copy_initial_weights=False) as (fmodel, diffopt):
        # update model parameters via meta loss
        for i in range(inner_itr):
            yp = fmodel(X, Y[:, 0, :])
            if ml3:
                pred_loss = loss_fn(yp, Y, i)
            else:
                pred_loss = loss_fn(yp, Y)

            eval_loss = eval_loss_fn(yp, Y)
            loss_trace.append(eval_loss.item())
            diffopt.step(pred_loss)

        torch.save(fmodel.state_dict(), 'regular_train_weights.pth')

    return loss_trace

def eval(exp_cfg, train_loss_fn, eval_loss_fn, x, y, name, ml3=False):
    seed = exp_cfg['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    mse = []
    nmse = []
    loss_trace = []

    k = 1
    T = 300 / k
    N = 30

    for i in range(1):
        DNN_ = CNNndp(N=N, state_index=np.arange(2))
        ndpn_ = Ndp(DNN_, T=T, l=1, N=N, state_index=np.arange(2))
        ndpn_.load_state_dict(torch.load('init_wieghts.pth'))
        ndpn_.to(torch.device("cuda:0"))

        loss = regular_train(loss_fn=train_loss_fn, eval_loss_fn=eval_loss_fn, task_model = ndpn_ ,
                                X = x, Y =y, exp_cfg=exp_cfg, ml3=ml3)

        ndpn_.load_state_dict(torch.load("regular_train_weights.pth"))
        yp = ndpn_.forward(x, y[:, 0, :])
        l = eval_loss_fn(yp, y)
        loss.append(l.item())

        for k in range(5):
            plt.plot(0.667 * yp[k, :, 0].detach().cpu().numpy(), -0.667 * yp[k, :, 1].detach().cpu().numpy(),
                     c='r', linewidth=5)
            plt.axis('off')
            plt.savefig(exp_cfg['model_save_path'] + '/train_img_' + str(k) + "_" + name + '.png')
            plt.clf()

            img = x[k].cpu().numpy() * 255
            img = np.asarray(img * 255, dtype=np.uint8)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.savefig(exp_cfg['model_save_path'] + '/ground_test_img_' + str(k) + "_" + name + '.png',
                        bbox_inches='tight', pad_inches=0)
            plt.clf()

            fig, ax = plt.subplots()
            im = ax.imshow(x[k].cpu().numpy() * 255)
            ax.plot(0.668 * yp[k, :, 0].detach().cpu().numpy(), 0.668 * yp[k, :, 1].detach().cpu().numpy(), 'x', ls='dotted', linewidth=10, color='red')
            plt.savefig(exp_cfg['model_save_path'] + '/results_test_img_' + str(k) + "_" + name + '.png',
                        bbox_inches='tight', pad_inches=0)
            plt.clf()
            plt.close(fig)




        mse.append(l.item())
        nmse.append(l.item()/y.cpu().numpy().var())
        loss_trace.append(loss)

    res = {'nmse': nmse, 'mse': mse, 'loss_trace': loss_trace}
    return res


def main(exp_cfg):
    device = exp_cfg['device']
    seed = exp_cfg['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(device)

    # --------------load-data---------------------------------------------
    digits = exp_cfg['digits']
    task_num = exp_cfg['task_num']
    images, or_tr, digits_inds = load_data(digits)
    data_points = 2000
    inds = np.arange(data_points)
    np.random.shuffle(inds)
    qry_inds = inds[1000:1500]
    spt_inds = inds[:1000]
    test_inds = inds[1500:1600]

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
        Y_test.append(Y[test_inds].to(device))

    # -----------------------------------------------------------------------
    #digits = exp_cfg['digits_test']
    #images, or_tr, digits_inds = load_data(digits)

    #X = torch.Tensor(images[digits_inds[0]]).float()
    #t = torch.Tensor(np.array(or_tr)[:, :, :2]).float()[digits_inds[0]]
    #Y = t[:, ::1, :]

    #X_test.append(X[test_inds].to(device))
    #Y_test.append(Y[test_inds].to(device))

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
    inner_itr = exp_cfg['inner_itr']
    outer_lr = exp_cfg['outer_lr']
    inner_lr = exp_cfg['inner_lr']

    k = 1
    T = 300 / k
    N = 30

    DNN = CNNndp(N=N, state_index=np.arange(2))
    ndpn = Ndp(DNN, T=T, l=1, N=N, state_index=np.arange(2))
    torch.save(ndpn.state_dict(), 'init_wieghts.pth')
    #ndpn.load_state_dict(torch.load('init_wieghts.pth'))
    optimizer = torch.optim.SGD(ndpn.parameters(), lr=inner_lr)
    meta_loss_network = ML3_NNloss(301, [100, 100], inner_itr)
    #meta_loss_network = ML3_StructuredLoss(inner_itr)
    meta_optimizer = torch.optim.Adam(meta_loss_network.parameters(), lr=outer_lr)
    ndpn.to(torch.device(device))
    meta_loss_network.to(torch.device(device))

    all_inner_losses = []
    x_axis = []
    test_losses_ml3 = []
    test_losses_norm = []

    res_test_eval_norm = eval(exp_cfg=exp_cfg,
                              train_loss_fn=MAE, eval_loss_fn=MAE, x=X_test[0], y=Y_test[0], name="norm")

    res_test_eval_ml3 = eval(exp_cfg=exp_cfg,
                             train_loss_fn=meta_loss_network, eval_loss_fn=MAE, x=X_test[0], y=Y_test[0], name="ml3",
                             ml3=True)

    test_loss_ml3 = np.mean(res_test_eval_ml3['mse'])
    test_loss_norm = np.mean(res_test_eval_norm['mse'])
    test_losses_ml3.append(test_loss_ml3)
    test_losses_norm.append(test_loss_norm)

    LOG.info(
        f' -- [Epoch {0:.2f}] Test Loss ML3: {test_loss_ml3:.2f} | TestLoss MAE: {test_loss_norm:.2f} --'
    )

    x_axis.append(0)

    for outer_i in range(num_epochs):
        spt_inds = np.arange(X_spt[0].shape[0])
        qry_inds = np.arange(X_qry[0].shape[0])
        np.random.shuffle(spt_inds)

        for ind in np.split(spt_inds, len(spt_inds) // batch_size):
            np.random.shuffle(qry_inds)
            ind_q = qry_inds[:100]
            ndpn.load_state_dict(torch.load('init_wieghts.pth'))
            meta_optimizer.zero_grad()

            losses_task = [0 for _ in range(inner_itr + 1)]
            task_losses = []
            for k in range(task_num):
                with higher.innerloop_ctx(ndpn, optimizer,
                                          copy_initial_weights=False) as (fmodel, diffopt):
                    # update model parameters via meta loss
                    for i in range(inner_itr):
                        yp = fmodel(X_spt[k][ind], Y_spt[k][ind, 0, :])
                        pred_loss = meta_loss_network.forward(yp, Y_spt[k][ind], i)
                        diffopt.step(pred_loss)

                        # eval i step
                        eval_loss = MAE(yp, Y_spt[k][ind]).detach()
                        losses_task[i] += eval_loss.item()/task_num

                    yp = fmodel(X_spt[k][ind], Y_spt[k][ind, 0, :])
                    eval_loss = MAE(yp, Y_spt[k][ind]).detach()
                    losses_task[i + 1] += eval_loss.item() / task_num

                    yp = fmodel(X_qry[k][ind_q], Y_qry[k][ind_q, 0, :])
                    task_loss = MAE(yp, Y_qry[k][ind_q])
                    task_losses.append(task_loss)


            meta_losses = sum(task_losses)/len(task_losses)
            meta_losses.backward()
            meta_optimizer.step()
            all_inner_losses.append(losses_task)

            LOG.info(
                f' [Epoch {outer_i:.2f}] Task loss: {task_loss.item():.2f} ] Spt loss: {losses_task[-1]:.2f}]|  '
            )

        if outer_i % 5 == 0 and outer_i != 0:

            torch.save(meta_loss_network.state_dict(), model_save_path + '/meta-loss.pth')
            torch.save(ndpn.state_dict(),  model_save_path + '/init_wieghts.pth')
            torch.save(meta_loss_network.state_dict(), 'meta-loss.pth')

            res_test_eval_norm = eval(exp_cfg=exp_cfg,
                                     train_loss_fn=MAE, eval_loss_fn=MAE, x=X_test[0], y=Y_test[0], name="norm")

            res_test_eval_ml3 = eval(exp_cfg=exp_cfg,
                                     train_loss_fn=meta_loss_network, eval_loss_fn=MAE, x=X_test[0], y=Y_test[0], name="ml3",
                                     ml3=True)

            x_axis.append(outer_i)

            test_loss_ml3 = np.mean(res_test_eval_ml3['mse'])
            test_loss_norm = np.mean(res_test_eval_norm['mse'])
            test_loss_trace_ml3 = res_test_eval_ml3['loss_trace']
            test_loss_trace_norm = res_test_eval_norm['loss_trace']

            LOG.info(
                f' -- [Epoch {outer_i:.2f}] Test Loss ML3: {test_loss_ml3:.2f} | TestLoss MAE: {test_loss_norm:.2f} --'
            )

            test_losses_ml3.append(test_loss_ml3)
            test_losses_norm.append(test_loss_norm)

            plt.plot(test_loss_trace_norm[0], label='ndp MAE loss')
            plt.plot(test_loss_trace_ml3[0], label='ndp ml3 loss')
            plt.legend()
            plt.savefig(model_save_path + "/train_result_trace.png".format(inner_itr))
            plt.clf()

            plt.plot(x_axis, test_losses_norm, label='ndp MAE loss')
            plt.plot(x_axis, test_losses_ml3, label='ndp ml3 loss')
            plt.ylim([0, test_losses_norm[-1] + 5])
            plt.legend()
            plt.savefig(model_save_path + "/train_result_final.png".format(inner_itr))
            plt.clf()

            inner_losses_trace = np.stack(all_inner_losses)
            for i in range(inner_itr + 1):
                plt.plot(inner_losses_trace[:, i], label='inner_loop_loss_' + str(i))
                plt.ylim([0, max(inner_losses_trace[:, i])])
                plt.legend()
                plt.savefig(model_save_path + "/inner_loop_" + str(i) +"_losses.png")
                plt.clf()

            file = open(model_save_path + "/inner_losses.txt", "w+")
            content = str(inner_losses_trace)
            file.write(content)
            file.close()


def test(exp_cfg):
    device = exp_cfg['device']
    seed = exp_cfg['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(device)

    # --------------load-data---------------------------------------------
    digits = exp_cfg['digits']
    task_num = exp_cfg['task_num']
    images, or_tr, digits_inds = load_data(digits)
    data_points = 2000
    inds = np.arange(data_points)
    np.random.shuffle(inds)
    qry_inds = inds[1000:1500]
    spt_inds = inds[:1000]
    test_inds = inds[1500:1600]

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
    inner_itr = exp_cfg['inner_itr']
    # -----------Create storage file-----------------------------------------
    time = str(datetime.now())
    time = time.replace(' ', '_')
    time = time.replace(':', '_')
    time = time.replace('-', '_')
    time = time.replace('.', '_')
    model_save_path = './data/' + '_' + time
    exp_cfg['model_save_path'] = model_save_path
    os.mkdir(model_save_path)

    k = 1
    T = 300 / k
    N = 30
    # ----------------------------------------------------------------------
    MAE = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')
    DNN = CNNndp(N=N, state_index=np.arange(2))
    ndpn = Ndp(DNN, T=T, l=1, N=N, state_index=np.arange(2))
    torch.save(ndpn.state_dict(), 'init_wieghts.pth')

    # -------------------initialize policies and optimizers------------------
    meta_loss_network = ML3_StructuredLoss(inner_itr)
    meta_loss_network.load_state_dict(torch.load('meta-loss.pth'))
    meta_loss_network.to(torch.device(device))

    res_test_eval_norm = eval(exp_cfg=exp_cfg,
                              train_loss_fn=MAE, eval_loss_fn=MAE, x=X_test[0], y=Y_test[0], name="norm")

    res_test_eval_ml3 = eval(exp_cfg=exp_cfg,
                             train_loss_fn=meta_loss_network, eval_loss_fn=MAE, x=X_test[0], y=Y_test[0], name="ml3",
                             ml3=True)

    test_loss_ml3 = np.mean(res_test_eval_ml3['mse'])
    test_loss_norm = np.mean(res_test_eval_norm['mse'])

    LOG.info(
        f' -- [Test Loss ML3: {test_loss_ml3:.2f} | TestLoss MAE: {test_loss_norm:.2f} --'
    )


if __name__ == "__main__":
    global LOG
    logging.basicConfig(level=logging.INFO)
    LOG = logging.getLogger(__name__)

    exp_cfg = {}
    exp_cfg['seed'] = 0
    exp_cfg['num_train_tasks'] = 1
    exp_cfg['num_test_tasks'] = 1
    exp_cfg['n_outer_iter'] = 1000
    exp_cfg['digits'] = [2]
    exp_cfg['digits_test'] = [2]
    exp_cfg['task_num'] = len(exp_cfg['digits'])
    exp_cfg['n_gradient_steps_at_test'] = 100
    exp_cfg['batch_size'] = 400
    exp_cfg['inner_lr'] = 1e-3
    exp_cfg['outer_lr'] = 1e-3
    exp_cfg['inner_itr'] = 1
    exp_cfg['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(exp_cfg)