import copy
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture

from models import BCNet, CVAE, PoseCVAE
from helpers import draw_object, draw_base_rect, normalize_angle, shortest_angle_dist, shortest_angle_diff
from constants import *

def process_data():
	# Read data
	data = pd.read_csv("../../dat/push_data/push_data_with_final_object_pose.csv")

	# center object coordinates at origin
	data.loc[:, 'o_ox'] -= TABLE[0]
	data.loc[:, 'o_oy'] -= TABLE[1]
	data.loc[:, 'o_oz'] -= TABLE[2] + TABLE_SIZE[2]

	# center push start pose coordinates at origin
	data.loc[:, 's_x'] -= TABLE[0]
	data.loc[:, 's_y'] -= TABLE[1]
	data.loc[:, 's_z'] -= TABLE[2] + TABLE_SIZE[2]

	# center object end pose coordinates at origin
	data.loc[:, 'e_x'] -= TABLE[0]
	data.loc[:, 'e_y'] -= TABLE[1]
	data.loc[:, 'e_z'] -= TABLE[2] + TABLE_SIZE[2]

	# normalise object yaw angle
	sin = np.sin(data.loc[:, 'o_oyaw'])
	cos = np.cos(data.loc[:, 'o_oyaw'])
	data.loc[:, 'o_oyaw'] = np.arctan2(sin, cos)

	# normalise desired push direction angle
	sin = np.sin(data.loc[:, 'm_dir_des'])
	cos = np.cos(data.loc[:, 'm_dir_des'])
	data.loc[:, 'm_dir_des'] = np.arctan2(sin, cos)

	# normalise achieved push direction angle
	sin = np.sin(data.loc[:, 'm_dir_ach'])
	cos = np.cos(data.loc[:, 'm_dir_ach'])
	data.loc[:, 'm_dir_ach'] = np.arctan2(sin, cos)

	# reorder columns
	# 0 : [o_ox, o_oy, o_oz, o_oyaw, o_shape, o_xs, o_ys, o_zs, o_mass, o_mu] : 9
	# 10 : [s_x, s_y, s_z, s_r11, s_r21, s_r31, s_r12, s_r22, s_r32, s_r13, s_r23, s_r33] : 21
	# 22 : [m_dir_des, m_dist_des] : 23
	# 24 : [m_dir_ach, m_dist_ach] : 25
	# 26 : [e_x, e_y, e_z, e_r11, e_r21, e_r31, e_r12, e_r22, e_r32, e_r13, e_r23, e_r33] : 37
	# 38 : [r]
	cols = data.columns.tolist()
	cols = cols[:10] +\
			cols[12:24] +\
			cols[10:12] +\
			cols[24:26] +\
			cols[26:-1] +\
			cols[-1:]
	data = data[cols]

	return data

# Helper function to train one model
def model_train_ik(model, X_train, y_train, X_val, y_val):
	model = model.to(DEVICE)
	# loss function and optimizer
	loss_fn = nn.BCELoss()  # binary cross entropy
	optimizer = optim.Adam(model.parameters(), lr=0.0001)

	N = X_train.shape[0]
	n_epochs = 250   # number of epochs to run
	batch_size = 1<<(int(np.sqrt(N))-1).bit_length()  # size of each batch
	batch_start = torch.arange(0, len(X_train), batch_size)

	# Hold the best model
	best_acc = -np.inf   # init to negative infinity
	best_weights = None
	best_acc_update_epoch = -1

	for epoch in range(n_epochs):
		idxs = torch.tensor(np.random.permutation(np.arange(0, N)))
		model.train()
		with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=epoch % 50 != 0) as bar:
			bar.set_description("Epoch {}".format(epoch))
			for start in bar:
				batch_idxs = idxs[start:start+batch_size]
				# take a batch
				X_batch = X_train[batch_idxs]
				y_batch = y_train[batch_idxs]

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
			# print('\tAccuracy improved at epoch {}'.format(epoch))
			best_acc = acc
			best_weights = copy.deepcopy(model.state_dict())
			best_acc_update_epoch = epoch
		elif ((best_acc_update_epoch >= 0) and epoch > best_acc_update_epoch + n_epochs//10):
			print('Training stopped at epoch {}. Last acc update was at epoch {}.'.format(epoch, best_acc_update_epoch))
			break

	# restore model and return best accuracy
	model.load_state_dict(best_weights)
	return best_acc

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_fn_final_cvae(recon_x, x, mu, logvar):
	mse = nn.MSELoss()
	recon = mse(recon_x[:, :2], x[:, :2])

	# torch version of shortest_angle_dist
	ang_diff = recon_x[:, 2] - x[:, 2]
	test = torch.abs(ang_diff) > 2*np.pi
	if (torch.any(test)):
		ang_diff[test] = torch.fmod(ang_diff[test], 2*np.pi)
	while (torch.any(ang_diff < -np.pi)):
		test = ang_diff < -np.pi
		ang_diff[test] += 2*np.pi
	while (torch.any(ang_diff > np.pi)):
		test = ang_diff > np.pi
		ang_diff[test] -= 2*np.pi
	recon += torch.mean(torch.abs(ang_diff))

	# see Appendix B from VAE paper:
	# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
	# https://arxiv.org/abs/1312.6114
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	return recon + kld

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_fn_pose_cvae(recon_x, x, mu, logvar):
	mae = nn.L1Loss()
	mse = nn.MSELoss()

	recon = mae(recon_x[:, 3:], x[:, 3:]) + mse(recon_x[:, :3], x[:, :3])
	# see Appendix B from VAE paper:
	# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
	# https://arxiv.org/abs/1312.6114
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	return recon + kld

def eval_fn_final_cvae(model, latent_dim, C_eval, X_eval, num_eval=5):
	with torch.no_grad():
		C_eval = torch.repeat_interleave(C_eval, num_eval, dim=0)
		Z_eval = torch.randn(C_eval.shape[0], latent_dim).to(DEVICE)

		# num_eval sample prediction per validation point
		X_eval_pred = model.decode(Z_eval, C_eval).data.cpu().numpy()
		# average every num_eval rows
		X_eval_pred = X_eval_pred.transpose().reshape(-1, num_eval).mean(axis=1).reshape(X_eval.shape[1], -1).transpose()

		ang_diff = shortest_angle_dist(X_eval_pred[:, 2], X_eval[:, 2].cpu().numpy())
		ang_loss = np.fabs(ang_diff)[:, None]

		mse = np.sum((X_eval_pred[:, :2] - X_eval[:, :2].cpu().numpy())**2, axis=1, keepdims=True)
		recon = np.mean(ang_loss + mse)

	return recon

def eval_fn_pose_cvae(model, latent_dim, C_eval, X_eval, num_eval=50):
	with torch.no_grad():
		C_eval = torch.repeat_interleave(C_eval, num_eval, dim=0)
		Z_eval = torch.randn(C_eval.shape[0], latent_dim).to(DEVICE)

		# num_eval sample prediction per validation point
		X_eval_pred = model.decode(Z_eval, C_eval).data.cpu().numpy()
		# average every num_eval rows
		X_eval_pred = X_eval_pred.transpose().reshape(-1, num_eval).mean(axis=1).reshape(X_eval.shape[1] - 3, -1).transpose()

		mae = np.sum(np.abs(X_eval_pred[:, 3:] - X_eval[:, 3:-3].cpu().numpy()), axis=1, keepdims=True)
		mse = np.sum((X_eval_pred[:, :3] - X_eval[:, :3].cpu().numpy())**2, axis=1, keepdims=True)
		recon = np.mean(mae + mse)

	return recon

# Helper function to train one model
def model_train_cvae(model, C_train, X_train, latent_dim, C_eval, X_eval, loss_fn, eval_fn):
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

def get_euler_from_R(R, deg=False):
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

	if deg:
		return np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)
	else:
		return roll, pitch, yaw

def get_euler_from_R_tensor(R, deg=False):
	if (len(R.shape) < 3):
		R = R[None, :, :]

	sy = (R[:,0,0]**2 + R[:,1,0]**2)**0.5
	singular = sy < 1e-6
	singular = singular.astype(np.int)

	roll = np.ones(singular.shape[0]) * np.nan
	pitch = np.ones(singular.shape[0]) * np.nan
	yaw = np.ones(singular.shape[0]) * np.nan

	roll[singular == 0] = np.arctan2(R[singular == 0, 2, 1] , R[singular == 0, 2, 2])
	pitch[singular == 0] = np.arctan2(-R[singular == 0, 2, 0], sy[singular == 0])
	yaw[singular == 0] = np.arctan2(R[singular == 0, 1, 0], R[singular == 0, 0, 0])

	roll[singular == 1] = np.arctan2(-R[singular == 1, 1, 2], R[singular == 1, 1, 1])
	pitch[singular == 1] = np.arctan2(-R[singular == 1, 2, 0], sy[singular == 1])
	yaw[singular == 1] = 0

	if deg:
		return np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)
	else:
		return roll, pitch, yaw

def make_rotation_matrix(rotvec):
	return np.reshape(rotvec, (3, 3)).transpose()

def vis_gmm(axis, gmm, obj, to_score):
	h = 0.01
	xx, yy = np.meshgrid(np.arange(-TABLE_SIZE[0], TABLE_SIZE[0], h),
						 np.arange(-TABLE_SIZE[1], TABLE_SIZE[1], h))
	push_to_xy = np.c_[xx.ravel(), yy.ravel()]

	# oyaw = obj[3]
	# R = np.array([	[np.cos(oyaw), -np.sin(oyaw), 0],
	# 				[np.sin(oyaw), np.cos(oyaw), 0],
	# 				[0, 0, 1]])
	# Rvec = R.transpose().reshape(1, -1)[0, :6][None, :]
	# Rvec = np.repeat(Rvec, push_to_xy.shape[0], axis=0)
	# xyz = np.hstack([push_to_xy, np.repeat(np.array([obj[2]])[None, :], push_to_xy.shape[0], axis=0)])
	# test_poses = np.hstack([xyz, Rvec])

	# scores = gmm.score_samples(test_poses)
	scores = gmm.score_samples(push_to_xy)
	scores = scores.reshape(xx.shape)
	axis.contourf(xx, yy, scores, cmap=plt.cm.magma, alpha=.8, zorder=2)

	# test_pt = test_poses[0, :].reshape(1, -1)
	test_pt = push_to_xy[0, :].reshape(1, -1)
	test_pt[0, 0] = to_score[0]
	test_pt[0, 1] = to_score[1]
	return gmm.score(test_pt)

if __name__ == '__main__':
	all_data = process_data()

	# get training data - IK
	X_ik = copy.deepcopy(all_data.iloc[:, :24])
	X_ik = X_ik.drop('o_mass', axis=1)
	X_ik = X_ik.drop('o_mu', axis=1)
	y_ik = copy.deepcopy(all_data.iloc[:, -1])
	# make binary labels
	y_ik.loc[y_ik > 0] = 2 # push IK failed => temporarily class 2
	y_ik.loc[y_ik <= 0] = 1 # push IK succeeded => class 1
	y_ik.loc[y_ik == 2] = 0 # push IK failed => class 0
	# Convert to 2D PyTorch tensors
	X_ik = torch.tensor(X_ik.values, dtype=torch.float32).to(DEVICE)
	y_ik = torch.tensor(y_ik, dtype=torch.float32).reshape(-1, 1).to(DEVICE)
	# train-test split: Hold out the test set for final model evaluation
	X_ik_train, X_ik_test, y_ik_train, y_ik_test = train_test_split(X_ik, y_ik, train_size=0.85, shuffle=True)

	# get training data - final pose
	C_final = copy.deepcopy(all_data.loc[all_data.r == 0, [all_data.columns.tolist()[i] for i in list(range(0, 22)) + list(range(24, 26))]])
	X_final = copy.deepcopy(all_data.loc[all_data.r == 0, [all_data.columns.tolist()[i] for i in list(range(-13, -1))]])
	X_final_rot = X_final.iloc[:, 3:].values.reshape(X_final.shape[0], 3, 3).transpose(0, 2, 1)
	_, _, yaws = get_euler_from_R_tensor(X_final_rot)
	X_final = X_final.drop(['e_z', 'e_r11', 'e_r21', 'e_r31', 'e_r12', 'e_r22', 'e_r32', 'e_r13', 'e_r23', 'e_r33'], axis=1)
	X_final['e_yaw'] = yaws
	# Convert to 2D PyTorch tensors
	C_final = torch.tensor(C_final.values, dtype=torch.float32).to(DEVICE)
	X_final = torch.tensor(X_final.values, dtype=torch.float32).to(DEVICE)
	# train-test split: Hold out the test set for final model evaluation
	C_final_train, C_final_test, X_final_train, X_final_test = train_test_split(C_final, X_final, train_size=0.85, shuffle=True)

	# get training data - start pose
	C_start = copy.deepcopy(all_data.loc[all_data.r == 0, [all_data.columns.tolist()[i] for i in list(range(0, 10)) + list(range(24, 26))]])
	X_start = copy.deepcopy(all_data.loc[all_data.r == 0, [all_data.columns.tolist()[i] for i in list(range(10, 22))]])
	# Convert to 2D PyTorch tensors
	C_start = torch.tensor(C_start.values, dtype=torch.float32).to(DEVICE)
	X_start = torch.tensor(X_start.values, dtype=torch.float32).to(DEVICE)
	# train-test split: Hold out the test set for start model evaluation
	C_start_train, C_start_test, X_start_train, X_start_test = train_test_split(C_start, X_start, train_size=0.85, shuffle=True)
	X_start_train = X_start_train[:, :-3] # drop final column of rotation matrix to be predicted

	# create networks
	layers = 4
	h_sizes = [256] * (layers-1)
	ik_net = BCNet()
	ik_net.initialise(X_ik.shape[1], 1, activation='relu', layers=layers, h_sizes=h_sizes)
	latent_dim = 6

	final_pose_net = CVAE()
	final_pose_net.initialise(X_final_train.shape[1], C_final_train.shape[1], latent_dim, activation='relu', layers=layers, h_sizes=h_sizes)

	start_pose_net = PoseCVAE()
	start_pose_net.initialise(X_start_train.shape[1], C_start_train.shape[1], latent_dim, activation='relu', layers=layers, h_sizes=h_sizes)

	# train networks
	ik_acc = model_train_ik(ik_net, X_ik_train, y_ik_train, X_ik_test, y_ik_test)
	model_train_cvae(final_pose_net, C_final_train, X_final_train, latent_dim, C_final_test, X_final_test, loss_fn_final_cvae, eval_fn_final_cvae)
	model_train_cvae(start_pose_net, C_start_train, X_start_train, latent_dim, C_start_test, X_start_test, loss_fn_pose_cvae, eval_fn_pose_cvae)

	# eval networks
	ik_net.eval()
	final_pose_net.eval()
	start_pose_net.eval()

	test_obj_grid_size = 5
	test_objs = test_obj_grid_size**2
	start_pose_samples = 3
	final_pose_samples = 10
	with torch.no_grad():
		figure = plt.figure(figsize=(test_objs,test_objs))
		axes = []

		h = 0.01
		xx, yy = np.meshgrid(np.arange(-TABLE_SIZE[0], TABLE_SIZE[0], h),
							 np.arange(-TABLE_SIZE[1], TABLE_SIZE[1], h))
		push_to_xy = np.c_[xx.ravel(), yy.ravel()]
		num_test_pts = push_to_xy.shape[0]
		for o in range(test_objs):
			pidx = np.random.randint(all_data.shape[0])
			obj_props = all_data.iloc[pidx, :10].values
			start_pose = all_data.iloc[pidx, 10:22].values

			desired_dirs = np.arctan2(push_to_xy[:, 1] - obj_props[1], push_to_xy[:, 0] - obj_props[0])[:, None]
			desired_dists = np.linalg.norm(obj_props[:2] - push_to_xy, axis=1)[:, None]
			# desired_push_stack = np.hstack([desired_dirs, desired_dists])

			ax = plt.subplot(test_obj_grid_size, test_obj_grid_size, o+1)
			axes.append(ax)
			draw_base_rect(ax)
			draw_object(ax, obj_props[None, :], zorder=3)

			start_pose_c = np.hstack([np.repeat(obj_props[None, :], num_test_pts, axis=0), desired_dirs, desired_dists])
			# start_pose_c = np.repeat(start_pose_c, start_pose_samples, axis=0)
			start_pose_c_torch = torch.tensor(start_pose_c, dtype=torch.float32).to(DEVICE)
			start_pose_z_torch = torch.randn(num_test_pts, latent_dim).to(DEVICE)

			start_pose_pred = start_pose_net.decode(start_pose_z_torch, start_pose_c_torch)
			start_pose_pred_rot = start_pose_net.get_rotation_matrix(start_pose_pred[:, -6:])
			start_pose_pred = start_pose_pred[:, :3].cpu().numpy()
			start_pose_pred_rot = start_pose_pred_rot.cpu().numpy()

			start_pose_pred_rot = np.transpose(start_pose_pred_rot, axes=[0, 2, 1]).reshape(start_pose_pred_rot.shape[0], -1)
			start_pose_pred = np.hstack([start_pose_pred, start_pose_pred_rot])

			obj_props_stack_ik = np.repeat(obj_props[None, :8], num_test_pts, axis=0)
			ik_score_input = np.hstack([obj_props_stack_ik, start_pose_pred, desired_dirs, desired_dists])
			ik_score_input_torch = torch.tensor(ik_score_input, dtype=torch.float32).to(DEVICE)
			ik_scores = ik_net(ik_score_input_torch).cpu().numpy() # num_test_pts x 1

			# obj_props_stack = np.repeat(obj_props[None, :], num_test_pts, axis=0)
			# final_pose_c = np.hstack([obj_props_stack, ik_score_input[:, 8:]])
			# final_pose_c = np.repeat(final_pose_c, repeats=final_pose_samples, axis=0)
			# final_pose_c_torch = torch.tensor(final_pose_c, dtype=torch.float32).to(DEVICE)
			# final_pose_z_torch = torch.randn(num_test_pts * final_pose_samples, latent_dim).to(DEVICE)

			# final_pose_mu, final_pose_logvar = final_pose_net.output_dist(final_pose_z_torch, final_pose_c_torch)
			# final_pose_mu = final_pose_mu.cpu().numpy()
			# final_pose_logvar = final_pose_logvar.cpu().numpy()
			# final_pose_half_logdet = -0.5 * final_pose_logvar.sum(axis=1, keepdims=True)
			# final_pose_var_inv = 1.0/np.exp(final_pose_logvar)
			# logpi = -0.5 * final_pose_mu.shape[1] * np.log(2 * np.pi)

			# obj_yaw_stack = np.repeat(normalize_angle(obj_props[3]), num_test_pts * final_pose_samples)[:, None]
			# final_pose_des = np.hstack([np.repeat(push_to_xy, final_pose_samples, axis=0), obj_yaw_stack])

			# residuals = (final_pose_des - final_pose_mu)
			# residuals[:, 2] = shortest_angle_diff(final_pose_des[:, 2], final_pose_mu[:, 2])
			# loglikelihood = -0.5 * residuals * final_pose_var_inv * residuals
			# loglikelihood = loglikelihood.sum(axis=1)[:, None] + final_pose_half_logdet + logpi
			# loglikelihood = loglikelihood.reshape(-1, final_pose_samples).sum(axis=1)[:, None]
			# # action_scores = np.exp(loglikelihood) * ik_scores
			action_scores = ik_scores
			action_scores = action_scores.reshape(xx.shape)
			cb = ax.contourf(xx, yy, action_scores, cmap=plt.cm.Greens, alpha=.8)

			ax.set_xlim(-TABLE_SIZE[0] - 0.01, TABLE_SIZE[0] + 0.01)
			ax.set_ylim(-TABLE_SIZE[1] - 0.01, TABLE_SIZE[1] + 0.01)
			ax.set_xticks(())
			ax.set_yticks(())
			ax.set_aspect('equal')
			divider = make_axes_locatable(ax)
			cax = divider.append_axes('right', size='5%', pad=0.05)
			figure.colorbar(cb, cax=cax, orientation='vertical')

		plt.tight_layout()
		plt.show()
		# plt.savefig('posterior-[{}].png'.format(o+1), bbox_inches='tight')
		[ax.cla() for ax in axes]
