import copy
import sys
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

from models import BCNet
from helpers import draw_object, draw_base_rect
from constants import *

def process_data():
    # Read data
    data = pd.read_csv("../../dat/push_data/push_data_for_ik_success.csv")

    # center object coordinates at origin
    data.loc[:, 'o_ox'] -= TABLE[0]
    data.loc[:, 'o_oy'] -= TABLE[1]
    data.loc[:, 'o_oz'] -= TABLE[2] + TABLE_SIZE[2]

    # normalise object yaw angle
    sin = np.sin(data.loc[:, 'o_oyaw'])
    cos = np.cos(data.loc[:, 'o_oyaw'])
    data.loc[:, 'o_oyaw'] = np.arctan2(sin, cos)

    # normalise push direction angle
    sin = np.sin(data.loc[:, 'm_dir_des'])
    cos = np.cos(data.loc[:, 'm_dir_des'])
    data.loc[:, 'm_dir_des'] = np.arctan2(sin, cos)

    # make binary labels
    data.loc[data.r > 0, 'r'] = 2 # push IK failed => temporarily class 2
    data.loc[data.r <= 0, 'r'] = 1 # push IK succeeded => class 1
    data.loc[data.r == 2, 'r'] = 0 # push IK failed => class 0

    return data

# Helper function to train one model
def model_train(model, X_train, y_train, X_val, y_val):
    model = model.to(DEVICE)
    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 100   # number of epochs to run
    batch_size = 128  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_acc = -np.inf   # init to negative infinity
    best_weights = None

    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description("Epoch {}".format(epoch))
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]

                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)

                # backward pass
                optimizer.zero_grad()
                loss.backward()

                # update weights
                optimizer.step()
                # print progress

                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val)
        acc = (y_pred.round() == y_val).float().mean()
        acc = float(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())

    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_acc

if __name__ == '__main__':
    # get training data
    data = process_data()
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    # Convert to 2D PyTorch tensors
    X = torch.tensor(X.values, dtype=torch.float32).to(DEVICE)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(DEVICE)
    # train-test split: Hold out the test set for final model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, shuffle=True)

    # network params
    in_dim = X.shape[1]
    out_dim = 1
    layers = 4
    h_sizes = [32] * (layers-1)
    activation = 'relu'

    model1 = BCNet()
    model1.initialise(in_dim, out_dim, activation=activation, layers=layers, h_sizes=h_sizes)
    acc = model_train(model1, X_train, y_train, X_test, y_test)
    # print(model1)
    print("Final model1 accuracy (4x32 deep, relu): {:.2f}%", acc*100)

    # network params
    layers = 2
    h_sizes = [256] * (layers-1)

    model2 = BCNet()
    model2.initialise(in_dim, out_dim, activation=activation, layers=layers, h_sizes=h_sizes)
    acc = model_train(model2, X_train, y_train, X_test, y_test)
    # print(model2)
    print("Final model2 accuracy (2x256 wide, relu): {:.2f}%", acc*100)

    # network params
    layers = 4
    h_sizes = [256] * (layers-1)

    model3 = BCNet()
    model3.initialise(in_dim, out_dim, activation=activation, layers=layers, h_sizes=h_sizes)
    acc = model_train(model3, X_train, y_train, X_test, y_test)
    # print(model3)
    print("Final model3 accuracy (4x256 deep+wide, relu): {:.2f}%", acc*100)

    model1.eval()
    model2.eval()
    model3.eval()

    test_plots = 5
    with torch.no_grad():
        # # Test out inference with 5 samples
        # for i in range(5):
        #     y_pred = model(X_test[i:i+1])
        #     print("{} -> {} (expected {})".format(X_test[i].cpu().numpy(), y_pred[0].cpu().numpy(), y_test[i].cpu().numpy()))

        # # Plot the ROC curve
        # y_pred = model(X_test)
        # fpr, tpr, thresholds = roc_curve(y_test.cpu(), y_pred.cpu())
        # plt.plot(fpr, tpr) # ROC curve = TPR vs FPR
        # plt.title("Receiver Operating Characteristics")
        # plt.xlabel("False Positive Rate")
        # plt.ylabel("True Positive Rate")
        # plt.show()

        h = 0.01
        xx, yy = np.meshgrid(np.arange(-TABLE_SIZE[0], TABLE_SIZE[0], h),
                             np.arange(-TABLE_SIZE[1], TABLE_SIZE[1], h))
        push_to_xy = np.c_[xx.ravel(), yy.ravel()]

        figure = plt.figure(figsize=(15,15))
        plot_count = 1
        for t in range(test_plots):
            pidx = np.random.randint(X_test.shape[0])
            ptest = X_test[pidx, :-2].cpu().numpy()[None, :]
            in_pts = np.repeat(ptest, push_to_xy.shape[0], axis=0)

            dirs = np.arctan2(push_to_xy[:, 1] - in_pts[:, 1], push_to_xy[:, 0] - in_pts[:, 0])
            dists = np.linalg.norm(in_pts[:, :2] - push_to_xy, axis=1)

            in_pts = np.hstack([in_pts, dirs[:, None], dists[:, None]])
            in_pts = torch.tensor(in_pts, dtype=torch.float32).to(DEVICE)

            preds1 = model1(in_pts)
            preds2 = model2(in_pts)
            preds3 = model3(in_pts)
            preds1 = preds1.cpu().numpy()
            preds1 = preds1.reshape(xx.shape)
            preds2 = preds2.cpu().numpy()
            preds2 = preds2.reshape(xx.shape)
            preds3 = preds3.cpu().numpy()
            preds3 = preds3.reshape(xx.shape)

            ax1 = plt.subplot(test_plots, 3, plot_count)
            ax2 = plt.subplot(test_plots, 3, plot_count+1)
            ax3 = plt.subplot(test_plots, 3, plot_count+2)
            plot_count += 3

            draw_base_rect(ax1)
            draw_base_rect(ax2)
            draw_base_rect(ax3)

            draw_object(ax1, ptest)
            draw_object(ax2, ptest)
            draw_object(ax3, ptest)

            cb1 = ax1.contourf(xx, yy, preds1, cmap=plt.cm.Greens, alpha=.8)
            cb2 = ax2.contourf(xx, yy, preds2, cmap=plt.cm.Greens, alpha=.8)
            cb3 = ax3.contourf(xx, yy, preds3, cmap=plt.cm.Greens, alpha=.8)
            ax1.set_xticks(())
            ax1.set_yticks(())
            ax1.set_aspect('equal')
            ax2.set_xticks(())
            ax2.set_yticks(())
            ax2.set_aspect('equal')
            ax3.set_xticks(())
            ax3.set_yticks(())
            ax3.set_aspect('equal')
        # ax.axis('equal')
        # plt.gca().set_aspect('equal')
        # plt.colorbar(cb)
        plt.tight_layout()
        plt.show()
