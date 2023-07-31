import copy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import pandas as pd

import torch
from sklearn.model_selection import train_test_split

from helpers import *
from constants import *
from models import *

if __name__ == '__main__':
	all_data = process_data()

	# get training data - IK
	# ipdb> p X_ik.columns.tolist()
	# ['o_ox', 'o_oy', 'o_oz', 'o_oyaw', 'o_shape', 'o_xs', 'o_ys', 'o_zs', 's_x', 's_y', 's_z', 's_yaw', 'm_dir_des', 'm_dist_des']

	X_ik = copy.deepcopy(all_data.iloc[:, :24])
	# extract only yaw angle from rotation matrices of poses
	# X_ik_rot = X_ik.iloc[:, 13:22].values.reshape(X_ik.shape[0], 3, 3).transpose(0, 2, 1)
	# _, _, yaws = get_euler_from_R_tensor(X_ik_rot)
	X_ik = X_ik.drop(['o_mass', 'o_mu', 's_x', 's_y', 's_z', 's_r11', 's_r21', 's_r31', 's_r12', 's_r22', 's_r32', 's_r13', 's_r23', 's_r33'], axis=1)
	# X_ik.insert(loc=11, column='s_yaw', value= yaws)
	# make binary labels
	y_ik = copy.deepcopy(all_data.iloc[:, -1])
	y_ik.loc[y_ik > 0] = 2 # push IK failed => temporarily class 2
	y_ik.loc[y_ik <= 0] = 1 # push IK succeeded => class 1
	y_ik.loc[y_ik == 2] = 0 # push IK failed => class 0
	# Convert to 2D PyTorch tensors
	X_ik = torch.tensor(X_ik.values, dtype=torch.float32).to(DEVICE)
	y_ik = torch.tensor(y_ik, dtype=torch.float32).reshape(-1, 1).to(DEVICE)
	# train-test split: Hold out the test set for final model evaluation
	X_ik_train, X_ik_test, y_ik_train, y_ik_test = train_test_split(X_ik, y_ik, train_size=0.85, shuffle=True)

	# get training data - final pose
	# ipdb> p C_final.columns.tolist()
	# ['o_ox', 'o_oy', 'o_oz', 'o_oyaw', 'o_shape', 'o_xs', 'o_ys', 'o_zs', 'o_mass', 'o_mu', 's_x', 's_y', 's_z', 's_yaw', 'm_dir_ach', 'm_dist_ach']
	# ipdb> p X_final.columns.tolist()
	# ['e_x', 'e_y', 'e_yaw']

	C_final = copy.deepcopy(all_data.loc[all_data.r == 0, [all_data.columns.tolist()[i] for i in list(range(0, 22)) + list(range(24, 26))]])
	X_final = copy.deepcopy(all_data.loc[all_data.r == 0, [all_data.columns.tolist()[i] for i in list(range(-13, -1))]])
	# extract only yaw angle from rotation matrices of poses
	# C_final_rot = C_final.iloc[:, 13:22].values.reshape(C_final.shape[0], 3, 3).transpose(0, 2, 1)
	# _, _, yaws = get_euler_from_R_tensor(C_final_rot)
	C_final = C_final.drop(['s_x', 's_y', 's_z', 's_r11', 's_r21', 's_r31', 's_r12', 's_r22', 's_r32', 's_r13', 's_r23', 's_r33'], axis=1)
	# C_final.insert(loc=13, column='s_yaw', value= yaws)
	X_final_rot = X_final.iloc[:, 3:].values.reshape(X_final.shape[0], 3, 3).transpose(0, 2, 1)
	_, _, yaws = get_euler_from_R_tensor(X_final_rot)
	X_final = X_final.drop(['e_z', 'e_r11', 'e_r21', 'e_r31', 'e_r12', 'e_r22', 'e_r32', 'e_r13', 'e_r23', 'e_r33'], axis=1)
	X_final.insert(loc=2, column='e_yaw', value= yaws)
	# Convert to 2D PyTorch tensors
	C_final = torch.tensor(C_final.values, dtype=torch.float32).to(DEVICE)
	X_final = torch.tensor(X_final.values, dtype=torch.float32).to(DEVICE)
	# train-test split: Hold out the test set for final model evaluation
	C_final_train, C_final_test, X_final_train, X_final_test = train_test_split(C_final, X_final, train_size=0.85, shuffle=True)

	# # get training data - start pose
	# # ipdb> p C_start.columns.tolist()
	# # ['o_ox', 'o_oy', 'o_oz', 'o_oyaw', 'o_shape', 'o_xs', 'o_ys', 'o_zs', 'o_mass', 'o_mu', 'm_dir_ach', 'm_dist_ach']
	# # ipdb> p X_start.columns.tolist()
	# # ['s_x', 's_y', 's_z', 's_yaw']

	# C_start = copy.deepcopy(all_data.loc[all_data.r == 0, [all_data.columns.tolist()[i] for i in list(range(0, 10)) + list(range(24, 26))]])
	# X_start = copy.deepcopy(all_data.loc[all_data.r == 0, [all_data.columns.tolist()[i] for i in list(range(10, 22))]])
	# # extract only yaw angle from rotation matrices of poses
	# X_start_rot = X_start.iloc[:, 3:].values.reshape(X_start.shape[0], 3, 3).transpose(0, 2, 1)
	# _, _, yaws = get_euler_from_R_tensor(X_start_rot)
	# X_start = X_start.drop(['s_r11', 's_r21', 's_r31', 's_r12', 's_r22', 's_r32', 's_r13', 's_r23', 's_r33'], axis=1)
	# X_start.insert(loc=3, column='s_yaw', value= yaws)
	# # Convert to 2D PyTorch tensors
	# C_start = torch.tensor(C_start.values, dtype=torch.float32).to(DEVICE)
	# X_start = torch.tensor(X_start.values, dtype=torch.float32).to(DEVICE)
	# # train-test split: Hold out the test set for start model evaluation
	# C_start_train, C_start_test, X_start_train, X_start_test = train_test_split(C_start, X_start, train_size=0.85, shuffle=True)

	# create networks
	layers = 4
	h_sizes = [256] * (layers-1)
	ik_net = BCNet()
	ik_net.initialise(X_ik.shape[1], 1, activation='relu', layers=layers, h_sizes=h_sizes)
	latent_dim = 3

	final_pose_net = CVAE()
	final_pose_net.initialise(X_final_train.shape[1], C_final_train.shape[1], latent_dim, activation='relu', layers=layers, h_sizes=h_sizes)

	# start_pose_net = CVAE()
	# start_pose_net.initialise(X_start_train.shape[1], C_start_train.shape[1], latent_dim, activation='relu', layers=layers, h_sizes=h_sizes)

	# train networks
	ik_acc = model_train_ik(ik_net, X_ik_train, y_ik_train, X_ik_test, y_ik_test, epochs=1000)
	model_train_cvae(final_pose_net, C_final_train, X_final_train, latent_dim, C_final_test, X_final_test, loss_fn_cvae, eval_fn_cvae, epochs=1000)
	# model_train_cvae(start_pose_net, C_start_train, X_start_train, latent_dim, C_start_test, X_start_test, loss_fn_cvae, eval_fn_cvae)

	# eval networks
	ik_net.eval()
	final_pose_net.eval()
	# start_pose_net.eval()

	test_obj_grid_num = 5
	test_objs = test_obj_grid_num**2
	start_pose_samples = 1
	final_pose_samples = 10
	with torch.no_grad():
		figure_ik, ax_ik = plt.subplots(test_obj_grid_num, test_obj_grid_num, figsize=(15,15))
		figure_dists, ax_dists = plt.subplots(test_obj_grid_num, test_obj_grid_num, figsize=(15,15))

		h = 0.01
		xx, yy = np.meshgrid(np.arange(-TABLE_SIZE[0], TABLE_SIZE[0], h),
							 np.arange(-TABLE_SIZE[1], TABLE_SIZE[1], h))
		push_to_xy = np.c_[xx.ravel(), yy.ravel()]
		num_test_pts = push_to_xy.shape[0]

		for o in range(test_objs):
			ax_row = o//test_obj_grid_num
			ax_col = o % test_obj_grid_num

			pidx = np.random.randint(all_data.shape[0])
			obj_props = all_data.iloc[pidx, :10].values

			desired_dirs = np.arctan2(push_to_xy[:, 1] - obj_props[1], push_to_xy[:, 0] - obj_props[0])[:, None]
			desired_dists = np.linalg.norm(obj_props[:2] - push_to_xy, axis=1)[:, None]

			# ax = plt.subplot(test_obj_grid_num, test_obj_grid_num, o+1)
			# axes.append(ax)
			draw_base_rect(ax_ik[ax_row, ax_col])
			draw_base_rect(ax_dists[ax_row, ax_col])
			draw_object(ax_ik[ax_row, ax_col], obj_props[None, :], zorder=3)
			draw_object(ax_dists[ax_row, ax_col], obj_props[None, :], zorder=3)

			# start_pose_c = np.hstack([np.repeat(obj_props[None, :], num_test_pts, axis=0), desired_dirs, desired_dists])
			# # start_pose_c = np.repeat(start_pose_c, start_pose_samples, axis=0)
			# start_pose_c_torch = torch.tensor(start_pose_c, dtype=torch.float32).to(DEVICE)
			# start_pose_z_torch = torch.randn(num_test_pts, latent_dim).to(DEVICE)
			# start_pose_pred = start_pose_net.decode(start_pose_z_torch, start_pose_c_torch).cpu().numpy()

			obj_props_stack_ik = np.repeat(obj_props[None, :8], num_test_pts, axis=0)
			ik_score_input = np.hstack([obj_props_stack_ik, desired_dirs, desired_dists])
			ik_score_input_torch = torch.tensor(ik_score_input, dtype=torch.float32).to(DEVICE)
			ik_scores = ik_net(ik_score_input_torch).cpu().numpy() # num_test_pts x 1
			ik_scores = ik_scores.reshape(xx.shape)
			cb_ik = ax_ik[ax_row, ax_col].contourf(xx, yy, ik_scores, cmap=plt.cm.Greens, alpha=.8)

			ax_ik[ax_row, ax_col].set_xlim(-TABLE_SIZE[0] - 0.01, TABLE_SIZE[0] + 0.01)
			ax_ik[ax_row, ax_col].set_ylim(-TABLE_SIZE[1] - 0.01, TABLE_SIZE[1] + 0.01)
			ax_ik[ax_row, ax_col].set_xticks(())
			ax_ik[ax_row, ax_col].set_yticks(())
			ax_ik[ax_row, ax_col].set_aspect('equal')
			divider_ik = make_axes_locatable(ax_ik[ax_row, ax_col])
			cax_ik = divider_ik.append_axes('right', size='5%', pad=0.05)
			figure_ik.colorbar(cb_ik, cax=cax_ik, orientation='vertical')

			obj_props_stack = np.repeat(obj_props[None, :], num_test_pts, axis=0)
			final_pose_c = np.hstack([obj_props_stack, ik_score_input[:, 8:]])
			final_pose_c = np.repeat(final_pose_c, repeats=final_pose_samples, axis=0)
			final_pose_c_torch = torch.tensor(final_pose_c, dtype=torch.float32).to(DEVICE)
			final_pose_z_torch = torch.randn(num_test_pts * final_pose_samples, latent_dim).to(DEVICE)
			final_pose_pred = final_pose_net.decode(final_pose_z_torch, final_pose_c_torch).cpu().numpy()

			obj_yaw_stack = np.repeat(normalize_angle(obj_props[3]), num_test_pts * final_pose_samples)[:, None]
			final_pose_des = np.hstack([np.repeat(push_to_xy, repeats=final_pose_samples, axis=0), obj_yaw_stack])
			final_pose_dist = (np.linalg.norm(final_pose_des[:, :2] - final_pose_pred[:, :2], axis=1, keepdims=True)/RES_XY)
			if (obj_props[4] == 0):
				if (np.abs(obj_props[5] - obj_props[6]) >= RES_XY): # asymmetric cuboid
					d1 = shortest_angle_dist(final_pose_pred[:, 2], final_pose_des[:, 2])[:, None]/RES_YAW
					d2 = shortest_angle_dist(final_pose_pred[:, 2] + np.pi, final_pose_des[:, 2])[:, None]/RES_YAW
					final_pose_dist += np.minimum(d1, d2)
			final_pose_dist = final_pose_dist.transpose().reshape(-1, final_pose_samples).mean(axis=1, keepdims=True)
			# final_pose_dist /= ik_scores
			final_pose_dist = final_pose_dist.reshape(xx.shape)
			cb_dists = ax_dists[ax_row, ax_col].contourf(xx, yy, final_pose_dist, cmap=plt.cm.YlOrRd, alpha=.8)

			ax_dists[ax_row, ax_col].set_xlim(-TABLE_SIZE[0] - 0.01, TABLE_SIZE[0] + 0.01)
			ax_dists[ax_row, ax_col].set_ylim(-TABLE_SIZE[1] - 0.01, TABLE_SIZE[1] + 0.01)
			ax_dists[ax_row, ax_col].set_xticks(())
			ax_dists[ax_row, ax_col].set_yticks(())
			ax_dists[ax_row, ax_col].set_aspect('equal')
			divider_dists = make_axes_locatable(ax_dists[ax_row, ax_col])
			cax_dists = divider_dists.append_axes('right', size='5%', pad=0.05)
			figure_dists.colorbar(cb_dists, cax=cax_dists, orientation='vertical')

		plt.tight_layout()
		# plt.show()
		figure_ik.savefig('posterior-nostart-ik-[{}].png'.format(o+1), bbox_inches='tight')
		figure_dists.savefig('posterior-nostart-dists-[{}].png'.format(o+1), bbox_inches='tight')
		# [ax.cla() for ax in axes]
