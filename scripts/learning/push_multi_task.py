import copy
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from helpers import *
from constants import *
from models import PushSuccessNet

TEST_OBJ_GRID_NUM = 5
FIGURE_IK, AX_IK = plt.subplots(TEST_OBJ_GRID_NUM, TEST_OBJ_GRID_NUM, figsize=(15,15))
FIGURE_DISTS, AX_DISTS = plt.subplots(TEST_OBJ_GRID_NUM, TEST_OBJ_GRID_NUM, figsize=(15,15))
FIGURE_COMBINED, AX_COMBINED = plt.subplots(TEST_OBJ_GRID_NUM, TEST_OBJ_GRID_NUM, figsize=(15,15))
CB_IK = []
CB_DISTS = []
CB_COMBINED = []

def draw_checkpoint(model, model_name, epoch=None, suffix=None):
	device = DEVICE if suffix is None else torch.device("cpu")

	test_objs = TEST_OBJ_GRID_NUM**2

	h = 0.01
	xx, yy = np.meshgrid(np.arange(-TABLE_SIZE[0], TABLE_SIZE[0], h),
						 np.arange(-TABLE_SIZE[1], TABLE_SIZE[1], h))
	push_to_xy = np.c_[xx.ravel(), yy.ravel()]
	num_test_pts = push_to_xy.shape[0]

	model.eval()
	with torch.no_grad():
		for o in range(test_objs):
			ax_row = o//TEST_OBJ_GRID_NUM
			ax_col = o % TEST_OBJ_GRID_NUM

			pidx = np.random.randint(X_test.shape[0])
			obj_props = X_test[pidx, :-2].cpu().numpy()[None, :]

			draw_base_rect(AX_IK[ax_row, ax_col])
			draw_base_rect(AX_DISTS[ax_row, ax_col])
			draw_base_rect(AX_COMBINED[ax_row, ax_col])
			draw_object(AX_IK[ax_row, ax_col], obj_props, zorder=3)
			draw_object(AX_DISTS[ax_row, ax_col], obj_props, zorder=3)
			draw_object(AX_COMBINED[ax_row, ax_col], obj_props, zorder=3)

			# desired_dirs = np.arctan2(push_to_xy[:, 1] - obj_props[0, 1], push_to_xy[:, 0] - obj_props[0, 0])[:, None]
			# desired_dists = np.linalg.norm(obj_props[0, :2] - push_to_xy, axis=1)[:, None]
			delta_x_test = push_to_xy[:, 0]
			delta_y_test = push_to_xy[:, 1]

			obj_props_stack = np.repeat(obj_props, num_test_pts, axis=0)
			# test_input = np.hstack([obj_props_stack, desired_dirs, desired_dists])
			test_input = np.hstack([obj_props_stack, delta_x_test[:, None], delta_y_test[:, None]])
			test_input_torch = torch.tensor(test_input, dtype=torch.float32).to(device)

			preds = model(test_input_torch)
			preds = preds.cpu().numpy()
			ik_preds = preds[:, 0].reshape(xx.shape)
			dist_preds = preds[:, 1].reshape(xx.shape)
			combined_preds = np.maximum(dist_preds, ik_preds) * ik_preds * 100

			contour_ik = AX_IK[ax_row, ax_col].contourf(xx, yy, ik_preds, cmap=plt.cm.Greens, alpha=.8)
			AX_IK[ax_row, ax_col].set_xlim(-TABLE_SIZE[0] - 0.01, TABLE_SIZE[0] + 0.01)
			AX_IK[ax_row, ax_col].set_ylim(-TABLE_SIZE[1] - 0.01, TABLE_SIZE[1] + 0.01)
			AX_IK[ax_row, ax_col].set_xticks(())
			AX_IK[ax_row, ax_col].set_yticks(())
			AX_IK[ax_row, ax_col].set_aspect('equal')
			divider_ik = make_axes_locatable(AX_IK[ax_row, ax_col])
			cax_ik = divider_ik.append_axes('right', size='5%', pad=0.05)
			CB_IK.append(FIGURE_IK.colorbar(contour_ik, cax=cax_ik, orientation='vertical'))

			contour_dists = AX_DISTS[ax_row, ax_col].contourf(xx, yy, dist_preds, cmap=plt.cm.YlOrRd, alpha=.8)
			AX_DISTS[ax_row, ax_col].set_xlim(-TABLE_SIZE[0] - 0.01, TABLE_SIZE[0] + 0.01)
			AX_DISTS[ax_row, ax_col].set_ylim(-TABLE_SIZE[1] - 0.01, TABLE_SIZE[1] + 0.01)
			AX_DISTS[ax_row, ax_col].set_xticks(())
			AX_DISTS[ax_row, ax_col].set_yticks(())
			AX_DISTS[ax_row, ax_col].set_aspect('equal')
			divider_dists = make_axes_locatable(AX_DISTS[ax_row, ax_col])
			cax_dists = divider_dists.append_axes('right', size='5%', pad=0.05)
			CB_DISTS.append(FIGURE_DISTS.colorbar(contour_dists, cax=cax_dists, orientation='vertical'))

			contour_combined = AX_COMBINED[ax_row, ax_col].contourf(xx, yy, combined_preds, cmap=plt.cm.plasma, alpha=.8)
			AX_COMBINED[ax_row, ax_col].set_xlim(-TABLE_SIZE[0] - 0.01, TABLE_SIZE[0] + 0.01)
			AX_COMBINED[ax_row, ax_col].set_ylim(-TABLE_SIZE[1] - 0.01, TABLE_SIZE[1] + 0.01)
			AX_COMBINED[ax_row, ax_col].set_xticks(())
			AX_COMBINED[ax_row, ax_col].set_yticks(())
			AX_COMBINED[ax_row, ax_col].set_aspect('equal')
			divider_combined = make_axes_locatable(AX_COMBINED[ax_row, ax_col])
			cax_combined = divider_combined.append_axes('right', size='5%', pad=0.05)
			CB_COMBINED.append(FIGURE_COMBINED.colorbar(contour_combined, cax=cax_combined, orientation='vertical'))

		plt.tight_layout()
		# plt.show()
		checkpoint_filename = model_name
		if suffix is None:
			checkpoint_filename += '-epoch_{}'.format(epoch)
		else:
			checkpoint_filename += suffix

		FIGURE_IK.savefig('posterior-nostart-maxfactor-ik-' + checkpoint_filename + '.png', bbox_inches='tight')
		FIGURE_DISTS.savefig('posterior-nostart-maxfactor-dists-' + checkpoint_filename + '.png', bbox_inches='tight')
		FIGURE_COMBINED.savefig('posterior-nostart-maxfactor-combined-' + checkpoint_filename + '.png', bbox_inches='tight')
		for r in range(TEST_OBJ_GRID_NUM):
			for c in range(TEST_OBJ_GRID_NUM):
				AX_IK[r, c].cla()
				AX_DISTS[r, c].cla()
				AX_COMBINED[r, c].cla()
		[cb.remove() for cb in CB_IK]
		[cb.remove() for cb in CB_DISTS]
		[cb.remove() for cb in CB_COMBINED]
		del CB_IK[:]
		del CB_DISTS[:]
		del CB_COMBINED[:]

def model_train_multi_head(model, model_name, X_train, y_train, loss_mask_train, X_val, y_val, loss_mask_val, epochs=1000):
	model = model.to(DEVICE)
	# loss function and optimizer
	bce = nn.BCELoss(reduction='none')  # binary cross entropy
	mse = nn.MSELoss(reduction='none')
	optimizer = optim.Adam(model.parameters(), lr=0.0001)

	N = X_train.shape[0]
	n_epochs = epochs   # number of epochs to run
	batch_size = 512  # size of each batch
	batch_start = torch.arange(0, len(X_train), batch_size)

	# Hold the best model
	best_eval_loss = np.inf   # init to negative infinity
	best_weights = None
	last_update_epoch = -1

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
				loss_mask_batch = loss_mask_train[batch_idxs]

				# forward pass
				combined_pred = model(X_batch)
				bce_loss = bce(combined_pred[:, 0], y_batch[:, 0])
				bce_loss = bce_loss * loss_mask_batch[:, 0]
				mse_loss = mse(combined_pred[:, 1], y_batch[:, 1])
				mse_loss = mse_loss * loss_mask_batch[:, 1]
				loss = bce_loss + mse_loss
				loss = loss.mean()

				# backward pass
				optimizer.zero_grad()
				loss.backward()

				# update weights
				optimizer.step()
		# evaluate accuracy at end of each epoch
		model.eval()
		with torch.no_grad():
			combined_pred_eval = model(X_val)
			eval_ik_loss = F.binary_cross_entropy(combined_pred_eval[:, 0], y_val[:, 0], reduction='none') * loss_mask_val[:, 0]
			eval_dist_loss = F.mse_loss(combined_pred_eval[:, 1], y_val[:, 1], reduction='none') * loss_mask_val[:, 1]
			eval_loss = eval_ik_loss + eval_dist_loss
			eval_loss = eval_loss.mean().cpu().numpy()
			if eval_loss < best_eval_loss:
				# print('\tAccuracy improved at epoch {}'.format(epoch))
				best_eval_loss = eval_loss
				best_weights = copy.deepcopy(model.state_dict())
				last_update_epoch = epoch
			elif ((last_update_epoch >= 0) and epoch > last_update_epoch + n_epochs//10):
				print('Training stopped at epoch {}. Last update was at epoch {}.'.format(epoch, last_update_epoch))
				break

		# if (epoch % 50 == 0):
		# 	draw_checkpoint(model, model_name, epoch=epoch)

	# restore model and return best accuracy
	model.eval()
	model.to(torch.device("cpu"))
	with torch.no_grad():
		draw_checkpoint(model1, model_name, epoch=-1, suffix='-inputpushto-end-test')
		torch.save(model.state_dict(), 'models/' + model_name + '_end.pth')

		model.load_state_dict(best_weights)
		torch.save(model.state_dict(), 'models/' + model_name + '_best.pth')
		draw_checkpoint(model1, model_name, epoch=-1, suffix='-inputpushto-best-test')

		x_sample = X_val[0].unsqueeze(0).to(torch.device("cpu"))
		print()
		print('x_sample: ', x_sample)
		print('x_sample prediction: ', model(x_sample))
		print('ground truth prediction: ', y_val[0].unsqueeze(0))
		print()

		# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
		traced_model = torch.jit.trace(model, (x_sample))
		torch.jit.save(traced_model, 'models/' + model_name + '_best_traced.pth')

		model.load_state_dict(torch.load('models/' + model_name + '_end.pth'))
		traced_model = torch.jit.trace(model, (x_sample))
		torch.jit.save(traced_model, 'models/' + model_name + '_end_traced.pth')

	return best_eval_loss

if __name__ == '__main__':
	all_data = process_data()

	# desired_x = all_data.loc[:, 'o_ox'] + (np.cos(all_data.loc[:, 'm_dir_des']) * all_data.loc[:, 'm_dist_des'])
	# desired_y = all_data.loc[:, 'o_oy'] + (np.sin(all_data.loc[:, 'm_dir_des']) * all_data.loc[:, 'm_dist_des'])
	# achieved_x = all_data.loc[:, 'o_ox'] + (np.cos(all_data.loc[:, 'm_dir_ach']) * all_data.loc[:, 'm_dist_ach'])
	# achieved_y = all_data.loc[:, 'o_oy'] + (np.sin(all_data.loc[:, 'm_dir_ach']) * all_data.loc[:, 'm_dist_ach'])
	# xy_discrepancy = np.sqrt((desired_x - achieved_x)**2 + (desired_y - achieved_y)**2)

	X = copy.deepcopy(all_data.iloc[:, [i for i in list(range(0, 10)) + list(range(22, 24))]])
	delta_x = X.loc[:, 'o_ox'] + np.cos(X.loc[:, 'm_dir_des']) * X.loc[:, 'm_dist_des']
	delta_y = X.loc[:, 'o_oy'] + np.sin(X.loc[:, 'm_dir_des']) * X.loc[:, 'm_dist_des']
	X.insert(loc=10, column='delta_x', value=delta_x)
	X.insert(loc=11, column='delta_y', value=delta_y)
	# make binary labels
	y = copy.deepcopy(all_data.iloc[:, [i for i in list(range(24, 26)) + [-1]]])
	# binarise ik success labels
	y.loc[y['r'] > 0, 'r'] = 2 # push IK failed => temporarily class 2
	y.loc[y['r'] <= 0, 'r'] = 0 # push IK succeeded => class 0
	y.loc[y['r'] == 2, 'r'] = 1 # push IK failed => class 1

	# compute distance between desired and achieved poses in polar coordinates
	final_pose_rot = all_data.iloc[:, 29:38].values.reshape(all_data.shape[0], 3, 3).transpose(0, 2, 1)
	_, _, yaws = get_euler_from_R_tensor(final_pose_rot)
	yaw_discrepancy = shortest_angle_dist(yaws, all_data.loc[:, 'o_oyaw'].values)
	yaw_discrepancy[all_data['r'] != 0] = 0.0

	ang_dist = shortest_angle_dist(y.loc[:, 'm_dir_ach'], X.loc[:, 'm_dir_des'])
	polar_dist = y.loc[:, 'm_dist_ach']**2 + X.loc[:, 'm_dist_des']**2 - 2*y.loc[:, 'm_dist_ach']*X.loc[:, 'm_dist_des']*np.cos(ang_dist)
	polar_dist = np.sqrt(polar_dist)
	pose_dist = polar_dist + yaw_discrepancy
	y.insert(loc=3, column='pose_dist', value=pose_dist)

	X = X.drop(['m_dir_des', 'm_dist_des'], axis=1)
	y = y.drop(['m_dir_ach', 'm_dist_ach'], axis=1)

	# compute loss mask - task 2 (distance/discrepancy prediction) is only if we sucessfully simulated
	task1_loss = np.ones(all_data.shape[0])
	task2_loss = all_data['r'] == 0
	task2_loss = task2_loss.values.astype(np.int)
	loss_mask = np.vstack([task1_loss, task2_loss]).transpose()
	y = np.hstack([y.values, loss_mask])

	# Convert to 2D PyTorch tensors
	X = torch.tensor(X.values, dtype=torch.float32).to(DEVICE)
	y = torch.tensor(y, dtype=torch.float32).to(DEVICE)
	# train-test split: Hold out the test set for final model evaluation
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.95, shuffle=True)
	loss_mask_train = y_train[:, -2:]
	y_train = y_train[:, :-2]
	loss_mask_test = y_test[:, -2:]
	y_test = y_test[:, :-2]

	# network params
	in_dim = X.shape[1]
	h_sizes = [
				# [32, 32, 32],
				[256, 256, 256],
				# [32, 256, 32],
				# [128, 64, 32],
				# [64, 32, 64]
			]
	activation = 'relu'

	for H in range(len(h_sizes)):
		model_name = '[{}]'.format(','.join(str(x) for x in h_sizes[H]))
		print("Train model: " + model_name)

		layers = len(h_sizes[H]) + 1
		model1 = PushSuccessNet()
		model1.initialise(in_dim, activation=activation, layers=layers, h_sizes=h_sizes[H])
		loss = model_train_multi_head(model1, model_name, X_train, y_train, loss_mask_train, X_test, y_test, loss_mask_test)
		# print(model1)
		print("Final model1 loss: {:.2f}".format(loss))
