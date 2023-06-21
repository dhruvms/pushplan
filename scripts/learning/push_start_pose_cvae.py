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

	# do not care about final object pose to predict ee start pose
	data = data.drop('e_x', axis=1)
	data = data.drop('e_y', axis=1)
	data = data.drop('e_z', axis=1)
	data = data.drop('e_r11', axis=1)
	data = data.drop('e_r21', axis=1)
	data = data.drop('e_r31', axis=1)
	data = data.drop('e_r12', axis=1)
	data = data.drop('e_r22', axis=1)
	data = data.drop('e_r32', axis=1)
	data = data.drop('e_r13', axis=1)
	data = data.drop('e_r23', axis=1)
	data = data.drop('e_r33', axis=1)

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

	# # predicting first two columns of rotation matrix, so we drop the third
	# data = data.drop('s_r13', axis=1)
	# data = data.drop('s_r23', axis=1)
	# data = data.drop('s_r33', axis=1)

	# normalise push direction angle
	sin = np.sin(data.loc[:, 'm_dir_ach'])
	cos = np.cos(data.loc[:, 'm_dir_ach'])
	data.loc[:, 'm_dir_ach'] = np.arctan2(sin, cos)

	# reorder columns for easy train/test split
	cols = data.columns.tolist()
	cols = cols[:10] + cols[-2:] + cols[10:-2]
	data = data[cols]

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

def eval_fn(model, latent_dim, C_eval, X_eval, num_eval=50):
	with torch.no_grad():
		C_eval = torch.repeat_interleave(C_eval, num_eval, dim=0)
		Z_eval = torch.randn(C_eval.shape[0], latent_dim).to(DEVICE)

		# num_eval sample prediction per validation point
		X_eval_pred = model.decode(Z_eval, C_eval).cpu().numpy()
		# average every num_eval rows
		X_eval_pred = X_eval_pred.transpose().reshape(-1, num_eval).mean(axis=1).reshape(X_eval.shape[1] - 3, -1).transpose()

		mae = np.sum(np.abs(X_eval_pred[:, 3:] - X_eval[:, 3:-3].cpu().numpy()), axis=1, keepdims=True)
		mse = np.sum((X_eval_pred[:, :3] - X_eval[:, :3].cpu().numpy())**2, axis=1, keepdims=True)
		recon = np.mean(mae + mse)

	return recon

# Helper function to train one model
def model_train(model, C_train, X_train, latent_dim, C_eval, X_eval):
	model = model.to(DEVICE)
	# loss function and optimizer
	optimizer = optim.Adam(model.parameters(), lr=0.0001)

	N = X_train.shape[0]
	n_epochs = 250   # number of epochs to run
	batch_size = 1<<(int(np.sqrt(N))-1).bit_length()  # size of each batch
	batch_start = torch.arange(0, len(X_train), batch_size)

	# Hold the best model
	best_recon = np.inf   # init to negative infinity
	best_weights = None
	best_recon_update_epoch = -1

	for epoch in range(n_epochs):
		idxs = torch.tensor(np.random.permutation(np.arange(0, N)))
		model.train()
		with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=epoch % 50 != 0) as bar:
			bar.set_description("Epoch {}".format(epoch))
			for start in bar:
				batch_idxs = idxs[start:start+batch_size]
				# take a batch
				C_batch = C_train[batch_idxs]
				X_batch = X_train[batch_idxs]

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

		# evaluate model at end of each epoch
		recon = eval_fn(model, latent_dim, C_eval, X_eval)
		if recon < best_recon:
			best_recon = recon
			best_weights = copy.deepcopy(model.state_dict())
			best_recon_update_epoch = epoch
		elif ((best_recon_update_epoch >= 0) and epoch > best_recon_update_epoch + n_epochs//10):
			print('Training stopped at epoch {}. Last acc update was at epoch {}.'.format(epoch, best_recon_update_epoch))
			break

	# restore model and return best accuracy
	model.load_state_dict(best_weights)
	return best_recon

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
	# Input columns - ['o_ox', 'o_oy', 'o_oz', 'o_oyaw', 'o_shape', 'o_xs', 'o_ys', 'o_zs', 'o_mass', 'o_mu', 'm_dir_ach', 'm_dist_ach']
	# Predicted columns - ['s_x', 's_y', 's_z', 's_r11', 's_r21', 's_r31', 's_r12', 's_r22', 's_r32', {'s_r13', 's_r23', 's_r33'}]

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
			model_train(model1, C_train, X_train, latent_dims[L], C_test, X_test)

			model1.eval()

			test_plots = 3
			pred_samples = 100
			with torch.no_grad():
				plot_count = 1
				axes = []
				for t in range(test_plots * test_plots):
					pidx = np.random.randint(X_test.shape[0])
					ctest = C_test[pidx, :].cpu().numpy()[None, :]
					ctest = np.repeat(ctest, pred_samples, axis=0)
					ctest_torch = torch.tensor(ctest).to(DEVICE)
					ztest_torch = torch.randn(pred_samples, latent_dims[L]).to(DEVICE)

					xpred = model1.decode(ztest_torch, ctest_torch)
					xpred_rot = model1.get_rotation_matrix(xpred[:, -6:])
					xpred = xpred[:, :3].cpu().numpy()
					xpred_rot = xpred_rot.cpu().numpy()

					ax = plt.subplot(test_plots + 1, test_plots, plot_count)
					axes.append(ax)
					plot_count += 1

					draw_base_rect(ax)
					draw_object(ax, ctest, alpha=1, zorder=2)
					push_to = ctest[0, :2] + (ctest[0, -1] * np.array([np.cos(ctest[0, -2]), np.sin(ctest[0, -2])]))
					ax.plot([ctest[0, 0], push_to[0]], [ctest[0, 1], push_to[1]], c='k', lw=2)
					ax.scatter(ctest[0, 0], ctest[0, 1], s=50, c='g', marker='*', zorder=2)
					ax.scatter(push_to[0], push_to[1], s=50, c='r', marker='*', zorder=2)

					asize = 0.08
					for i in range(xpred.shape[0]):
						ayaw = get_yaw_from_R(xpred_rot[i])
						ax.arrow(xpred[i, 0], xpred[i, 1], asize * np.cos(ayaw), asize * np.sin(ayaw),
									length_includes_head=True, head_width=0.02, head_length=0.02,
									ec='gold', fc='gold', alpha=0.8)

					xtest = X_test[pidx, :].cpu().numpy()
					R = make_rotation_matrix(xtest[3:])
					true_yaw = get_yaw_from_R(R)
					ax.arrow(xtest[0], xtest[1], asize * np.cos(true_yaw), asize * np.sin(true_yaw),
								length_includes_head=True, head_width=0.02, head_length=0.02,
								ec='magenta', fc='magenta')

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
				plt.savefig('[{}]'.format(','.join(str(x) for x in h_sizes[H])) + '-[{}]-startpose.png'.format(latent_dims[L]), bbox_inches='tight')
				[ax.cla() for ax in axes]
