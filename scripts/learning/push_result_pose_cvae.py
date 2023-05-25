import copy
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

from models import PoseCVAE
from helpers import draw_object, draw_base_rect
from constants import *

def process_data():
    # Read data
    data = pd.read_csv("../../dat/push_data/push_data_with_final_object_pose.csv")

    # only want to train on successful pushes
    data = data.drop(data[data.r != 0].index)
    # do not need result column anymore
    data = data.drop('r', axis=1)

    # we will use achieved push columns, not desired
    data = data.drop('m_dir_des', axis=1)
    data = data.drop('m_dist_des', axis=1)

    # center object coordinates at origin
    data.loc[:, 'o_ox'] -= TABLE[0]
    data.loc[:, 'o_oy'] -= TABLE[1]
    data.loc[:, 'o_oz'] -= TABLE[2] + TABLE_SIZE[2]

    # normalise object yaw angle
    sin = np.sin(data.loc[:, 'o_oyaw'])
    cos = np.cos(data.loc[:, 'o_oyaw'])
    data.loc[:, 'o_oyaw'] = np.arctan2(sin, cos)

    # center push start pose coordinates at origin
    data.loc[:, 's_x'] -= TABLE[0]
    data.loc[:, 's_y'] -= TABLE[1]
    data.loc[:, 's_z'] -= TABLE[2] + TABLE_SIZE[2]

    # center push end pose coordinates at origin
    data.loc[:, 'e_x'] -= TABLE[0]
    data.loc[:, 'e_y'] -= TABLE[1]
    data.loc[:, 'e_z'] -= TABLE[2] + TABLE_SIZE[2]

    # # predicting first two columns of rotation matrix, so we drop the third
    # data = data.drop('e_r13', axis=1)
    # data = data.drop('e_r23', axis=1)
    # data = data.drop('e_r33', axis=1)

    # normalise push direction angle
    sin = np.sin(data.loc[:, 'm_dir_ach'])
    cos = np.cos(data.loc[:, 'm_dir_ach'])
    data.loc[:, 'm_dir_ach'] = np.arctan2(sin, cos)

    return data


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_fn(recon_x, x, mu, logvar):
    mae = nn.L1Loss()
    mse = nn.MSELoss()

    recon = mae(recon_x[:, 3:], x[:, 3:]) + mse(recon_x[:, :3], x[:, :3])
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kld

# def eval_fn(model, x_test, c_test, num_test, latent_dim, weights, rscale=1e-2):
#     model.eval()

#     with torch.no_grad():
#         c = Variable(torch.from_numpy(c_test)).cuda()
#         z = Variable(torch.randn(num_test, latent_dim)).cuda()
#         # one sample prediction per validation point
#         y = model.decode(z, c).data.cpu().numpy()

#         weights = np.asarray(weights).astype(np.float32)
#         # out_dim = y.shape[1]
#         # norm = y[:, np.arange(0, out_dim, 2)]**2 + x_test[:, np.arange(0, out_dim, 2)]**2-2*y[:, np.arange(0, out_dim, 2)]*x_test[:, np.arange(0, out_dim, 2)]*np.cos(((y[:, np.arange(1, out_dim, 2)]-x_test[:, np.arange(1, out_dim, 2)]) + np.pi) % (2 * np.pi ) - np.pi)
#         # eval_loss = np.sum(weights * norm, axis=1, keepdims=True)
#         # eval_loss = np.linalg.norm(x_test-y, axis=1, keepdims=True)
#         eval_loss = rscale * np.sum(weights * (y - x_test)**2, axis=1, keepdims=True)
#         total_eval_loss = np.sum(eval_loss)
#         mean_eval_loss = np.mean(eval_loss)
#         std_eval_loss = np.std(eval_loss)

#     return total_eval_loss, mean_eval_loss, std_eval_loss

# Helper function to train one model
def model_train(model, C_train, X_train):
    model = model.to(DEVICE)
    # loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 1000   # number of epochs to run
    batch_size = 64  # size of each batch
    batch_start = torch.arange(0, len(C_train), batch_size)

    # # Hold the best model
    # best_acc = -np.inf   # init to negative infinity
    # best_weights = None

    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description("Epoch {}".format(epoch))
            for start in bar:
                # take a batch
                C_batch = C_train[start:start+batch_size]
                X_batch = X_train[start:start+batch_size]

                # forward pass
                recon_batch, mu, logvar = model(X_batch, C_batch)
                loss = loss_fn(recon_batch, X_batch, mu, logvar)

                # backward pass
                optimizer.zero_grad()
                loss.backward()

                # update weights
                optimizer.step()
                # print progress

                # acc = (X_pred.round() == X_batch).float().mean()
                # bar.set_postfix(
                #     loss=float(loss),
                #     acc=float(acc)
                # )

        # # evaluate accuracy at end of each epoch
        # model.eval()
        # X_pred = model(C_val)
        # acc = (X_pred.round() == X_val).float().mean()
        # acc = float(acc)
        # if acc > best_acc:
        #     best_acc = acc
        #     best_weights = copy.deepcopy(model.state_dict())

    # # restore model and return best accuracy
    # model.load_state_dict(best_weights)
    # return best_acc

def get_yaw_from_R(R):
    return np.arctan2(R[1,0], R[0,0])

def get_euler_from_R(R):
    sy = (R[0,0]**2 + R[1,0]**2)**0.5
    singular = sy < 1e-6

    roll = pitch = yaw = None
    if not singular:
        roll = np.arctan2(R[2,1] , R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = np.arctan2(R[1,0], R[0,0])
    else:
        roll = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = 0

    return roll, pitch, yaw

def make_rotation_matrix(rotvec):
    return np.reshape(rotvec, (3, 3)).transpose()

if __name__ == '__main__':
    # get training data
    data = process_data()
    C = data.iloc[:, :-12]
    X = data.iloc[:, -12:]

    # Convert to 2D PyTorch tensors
    C = torch.tensor(C.values, dtype=torch.float32).to(DEVICE)
    X = torch.tensor(X.values, dtype=torch.float32).to(DEVICE)
    # train-test split: Hold out the test set for final model evaluation
    C_train, C_test, X_train, X_test = train_test_split(C, X, train_size=0.95, shuffle=True)
    X_train = X_train[:, :-3] # drop final column of rotation matrix to be predicted

    # network params
    in_dim = X_train.shape[1]
    cond_dim = C_train.shape[1]
    latent_dims = list(range(2, 10))
    # layers = 4
    h_sizes = [
                [32, 32, 32],
                [256, 256, 256],
                [32, 256, 32],
                [128, 64, 32],
                [64, 32, 64]
            ]
    activation = 'relu'

    figure = plt.figure(figsize=(10,10))
    for L in range(len(latent_dims)):
        for H in range(len(h_sizes)):
            print('Train model: [{},{},{}]-[{}]'.format(h_sizes[H][0], h_sizes[H][1], h_sizes[H][2], latent_dims[L]))

            layers = len(h_sizes[H]) + 1
            model1 = PoseCVAE()
            model1.initialise(in_dim, cond_dim, latent_dims[L], activation=activation, layers=layers, h_sizes=h_sizes[H])
            # print(model1)
            model_train(model1, C_train, X_train)

            model1.eval()

            test_plots = 3
            pred_samples = 10
            with torch.no_grad():
                plot_count = 1
                axes = []
                for t in range(test_plots * test_plots):
                    pidx = np.random.randint(X_test.shape[0])
                    ctest = C_test[pidx, :].cpu().numpy()[None, :]
                    ctest = np.repeat(ctest, pred_samples, axis=0)
                    ctest_torch = torch.tensor(ctest).to(DEVICE)
                    ztest_torch = torch.randn(pred_samples, latent_dims[L]).to(DEVICE)

                    xtest = X_test[pidx, :].cpu().numpy()
                    xpred = model1.decode(ztest_torch, ctest_torch)
                    xpred_rot = model1.get_rotation_matrix(xpred[:, -6:])
                    xpred = xpred[:, :3].cpu().numpy()
                    xpred_rot = xpred_rot.cpu().numpy()

                    ax = plt.subplot(test_plots + 1, test_plots, plot_count)
                    axes.append(ax)
                    plot_count += 1

                    draw_base_rect(ax)
                    draw_object(ax, ctest, alpha=1, zorder=2)

                    push_start = ctest[0, 10:12]
                    push_params = ctest[0, -2:]
                    push_end = push_start + np.array([np.cos(push_params[0]), np.sin(push_params[0])]) * push_params[1]
                    ax.plot([push_start[0], push_end[0]], [push_start[1], push_end[1]], c='k', lw=2)
                    ax.scatter(push_start[0], push_start[1], s=50, c='g', marker='*', zorder=2)
                    ax.scatter(push_end[0], push_end[1], s=50, c='r', marker='*', zorder=2)

                    onew = copy.deepcopy(ctest)
                    onew[0, 0] = xtest[0]
                    onew[0, 1] = xtest[1]
                    onew[0, 2] = xtest[2]
                    R = make_rotation_matrix(xtest[3:])
                    oyaw = get_yaw_from_R(R)
                    onew[0, 3] = oyaw
                    draw_object(ax, onew, alpha=1, zorder=2, colour='cyan')

                    for i in range(xpred.shape[0]):
                        obj = copy.deepcopy(ctest)
                        obj[0, 0] = xpred[i, 0]
                        obj[0, 1] = xpred[i, 1]
                        obj[0, 2] = xpred[i, 2]
                        oyaw = get_yaw_from_R(xpred_rot[i])
                        obj[0, 3] = oyaw
                        draw_object(ax, obj, alpha=0.5, zorder=3)

                    ax.set_xlim(-TABLE_SIZE[0] - 0.01, TABLE_SIZE[0] + 0.01)
                    ax.set_ylim(-TABLE_SIZE[1] - 0.01, TABLE_SIZE[1] + 0.01)
                    ax.set_xticks(())
                    ax.set_yticks(())
                    ax.set_aspect('equal')

                # ax.axis('equal')
                # plt.gca().set_aspect('equal')
                # plt.colorbar(cb)
                plt.tight_layout()
                # plt.show()
                plt.savefig('[{},{},{}]-[{}].png'.format(h_sizes[H][0], h_sizes[H][1], h_sizes[H][2], latent_dims[L]), bbox_inches='tight')
                [ax.cla() for ax in axes]
