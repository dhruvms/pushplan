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

def model_train_multi_head(model, X_train, y_train, loss_mask_train, X_val, y_val, loss_mask_val, epochs=1000):
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
			eval_dist_loss = F.mse_loss(combined_pred_eval[:, 1], y_val[:, 1], reduction='none')
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

	# restore model and return best accuracy
	model.load_state_dict(best_weights)
	return best_eval_loss

if __name__ == '__main__':
	all_data = process_data()

	X = copy.deepcopy(all_data.iloc[:, [i for i in list(range(0, 10)) + list(range(22, 24))]])
	delta_x = np.cos(X.loc[:, 'm_dir_des']) * X.loc[:, 'm_dist_des']
	delta_y = np.sin(X.loc[:, 'm_dir_des']) * X.loc[:, 'm_dist_des']
	X.insert(loc=10, column='delta_x', value=delta_x)
	X.insert(loc=11, column='delta_y', value=delta_y)
	# make binary labels
	y = copy.deepcopy(all_data.iloc[:, [i for i in list(range(24, 26)) + [-1]]])
	# binarise ik success labels
	y.loc[y['r'] > 0, 'r'] = 2 # push IK failed => temporarily class 2
	y.loc[y['r'] <= 0, 'r'] = 1 # push IK succeeded => class 1
	y.loc[y['r'] == 2, 'r'] = 0 # push IK failed => class 0

	# compute distance between desired and achieved poses in polar coordinates
	ang_dist = shortest_angle_dist(y.loc[:, 'm_dir_ach'], X.loc[:, 'm_dir_des'])
	polar_dist = y.loc[:, 'm_dist_ach']**2 + X.loc[:, 'm_dist_des']**2 - 2*y.loc[:, 'm_dist_ach']*X.loc[:, 'm_dist_des']*np.cos(ang_dist)
	polar_dist = np.sqrt(polar_dist)
	y.insert(loc=3, column='polar_dist', value=polar_dist)

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
				[32, 32, 32],
				[256, 256, 256],
				[32, 256, 32],
				[128, 64, 32],
				[64, 32, 64]
			]
	activation = 'relu'

	test_obj_grid_num = 5
	test_objs = test_obj_grid_num**2
	figure_ik, ax_ik = plt.subplots(test_obj_grid_num, test_obj_grid_num, figsize=(15,15))
	figure_dists, ax_dists = plt.subplots(test_obj_grid_num, test_obj_grid_num, figsize=(15,15))
	cb_ik = []
	cb_dists = []

	h = 0.01
	xx, yy = np.meshgrid(np.arange(-TABLE_SIZE[0], TABLE_SIZE[0], h),
						 np.arange(-TABLE_SIZE[1], TABLE_SIZE[1], h))
	push_to_xy = np.c_[xx.ravel(), yy.ravel()]
	num_test_pts = push_to_xy.shape[0]

	for H in range(len(h_sizes)):
		model_name = '[{}]'.format(','.join(str(x) for x in h_sizes[H]))
		print("Train model: " + model_name)

		layers = len(h_sizes[H]) + 1
		model1 = PushSuccessNet()
		model1.initialise(in_dim, activation=activation, layers=layers, h_sizes=h_sizes[H])
		loss = model_train_multi_head(model1, X_train, y_train, loss_mask_train, X_test, y_test, loss_mask_test, epochs=100)
		# print(model1)
		print("Final model1 loss: {:.2f}".format(loss))

		model1.eval()
		with torch.no_grad():
			for o in range(test_objs):
				ax_row = o//test_obj_grid_num
				ax_col = o % test_obj_grid_num

				pidx = np.random.randint(X_test.shape[0])
				obj_props = X_test[pidx, :-2].cpu().numpy()[None, :]

				draw_base_rect(ax_ik[ax_row, ax_col])
				draw_base_rect(ax_dists[ax_row, ax_col])
				draw_object(ax_ik[ax_row, ax_col], obj_props, zorder=3)
				draw_object(ax_dists[ax_row, ax_col], obj_props, zorder=3)

				# desired_dirs = np.arctan2(push_to_xy[:, 1] - obj_props[0, 1], push_to_xy[:, 0] - obj_props[0, 0])[:, None]
				# desired_dists = np.linalg.norm(obj_props[0, :2] - push_to_xy, axis=1)[:, None]
				delta_x_test = push_to_xy[:, 0] - obj_props[0, 0]
				delta_y_test = push_to_xy[:, 1] - obj_props[0, 1]

				obj_props_stack = np.repeat(obj_props, num_test_pts, axis=0)
				# test_input = np.hstack([obj_props_stack, desired_dirs, desired_dists])
				test_input = np.hstack([obj_props_stack, delta_x_test[:, None], delta_y_test[:, None]])
				test_input_torch = torch.tensor(test_input, dtype=torch.float32).to(DEVICE)

				preds = model1(test_input_torch)
				preds = preds.cpu().numpy()
				ik_preds = preds[:, 0].reshape(xx.shape)
				dist_preds = preds[:, 1].reshape(xx.shape)

				contour_ik = ax_ik[ax_row, ax_col].contourf(xx, yy, ik_preds, cmap=plt.cm.Greens, alpha=.8)
				ax_ik[ax_row, ax_col].set_xlim(-TABLE_SIZE[0] - 0.01, TABLE_SIZE[0] + 0.01)
				ax_ik[ax_row, ax_col].set_ylim(-TABLE_SIZE[1] - 0.01, TABLE_SIZE[1] + 0.01)
				ax_ik[ax_row, ax_col].set_xticks(())
				ax_ik[ax_row, ax_col].set_yticks(())
				ax_ik[ax_row, ax_col].set_aspect('equal')
				divider_ik = make_axes_locatable(ax_ik[ax_row, ax_col])
				cax_ik = divider_ik.append_axes('right', size='5%', pad=0.05)
				cb_ik.append(figure_ik.colorbar(contour_ik, cax=cax_ik, orientation='vertical'))

				contour_dists = ax_dists[ax_row, ax_col].contourf(xx, yy, dist_preds, cmap=plt.cm.YlOrRd, alpha=.8)
				ax_dists[ax_row, ax_col].set_xlim(-TABLE_SIZE[0] - 0.01, TABLE_SIZE[0] + 0.01)
				ax_dists[ax_row, ax_col].set_ylim(-TABLE_SIZE[1] - 0.01, TABLE_SIZE[1] + 0.01)
				ax_dists[ax_row, ax_col].set_xticks(())
				ax_dists[ax_row, ax_col].set_yticks(())
				ax_dists[ax_row, ax_col].set_aspect('equal')
				divider_dists = make_axes_locatable(ax_dists[ax_row, ax_col])
				cax_dists = divider_dists.append_axes('right', size='5%', pad=0.05)
				cb_dists.append(figure_dists.colorbar(contour_dists, cax=cax_dists, orientation='vertical'))

			plt.tight_layout()
			# plt.show()
			figure_ik.savefig('posterior-nostart-ik-' + model_name + '.png', bbox_inches='tight')
			figure_dists.savefig('posterior-nostart-dists-' + model_name + '.png', bbox_inches='tight')
			for r in range(test_obj_grid_num):
				for c in range(test_obj_grid_num):
					ax_ik[r, c].cla()
					ax_dists[r, c].cla()
			[cb.remove() for cb in cb_ik]
			[cb.remove() for cb in cb_dists]
			del cb_ik[:]
			del cb_dists[:]
