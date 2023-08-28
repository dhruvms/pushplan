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

def draw_and_run_model(
	model, obj_props,
	figure_ik, figure_dists, figure_combined,
	ax_ik, ax_dists, ax_combined,
	cb_ik, cb_dists, cb_combined,
	torch_device, threshold=0.05):

	h = 0.01
	xx, yy = np.meshgrid(np.arange(-TABLE_SIZE[0], TABLE_SIZE[0], h),
						 np.arange(-TABLE_SIZE[1], TABLE_SIZE[1], h))
	push_to_xy = np.c_[xx.ravel(), yy.ravel()]
	num_test_pts = push_to_xy.shape[0]

	draw_base_rect(ax_ik)
	draw_base_rect(ax_dists)
	draw_base_rect(ax_combined)
	draw_object(ax_ik, obj_props, zorder=3)
	draw_object(ax_dists, obj_props, zorder=3)
	draw_object(ax_combined, obj_props, zorder=3)

	delta_x_test = push_to_xy[:, 0]
	delta_y_test = push_to_xy[:, 1]

	obj_props_stack = np.repeat(obj_props, num_test_pts, axis=0)
	# test_input = np.hstack([obj_props_stack, desired_dirs, desired_dists])
	test_input = np.hstack([obj_props_stack, delta_x_test[:, None], delta_y_test[:, None], threshold * np.ones(obj_props_stack.shape[0])[:, None]])
	test_input_torch = torch.tensor(test_input, dtype=torch.float32).to(torch_device)

	preds = model(test_input_torch)
	preds = preds.cpu().numpy()
	ik_preds = preds[:, 0].reshape(xx.shape)
	dist_preds = preds[:, 1].reshape(xx.shape)
	combined_preds = 1 - (dist_preds * ik_preds)

	contour_ik = ax_ik.contourf(xx, yy, ik_preds, cmap=plt.cm.Greens, alpha=.8)
	ax_ik.set_xlim(-TABLE_SIZE[0] - 0.01, TABLE_SIZE[0] + 0.01)
	ax_ik.set_ylim(-TABLE_SIZE[1] - 0.01, TABLE_SIZE[1] + 0.01)
	ax_ik.set_xticks(())
	ax_ik.set_yticks(())
	ax_ik.set_aspect('equal')
	divider_ik = make_axes_locatable(ax_ik)
	cax_ik = divider_ik.append_axes('right', size='5%', pad=0.05)
	cb_ik.append(figure_ik.colorbar(contour_ik, cax=cax_ik, orientation='vertical'))

	contour_dists = ax_dists.contourf(xx, yy, dist_preds, cmap=plt.cm.YlOrRd, alpha=.8)
	ax_dists.set_xlim(-TABLE_SIZE[0] - 0.01, TABLE_SIZE[0] + 0.01)
	ax_dists.set_ylim(-TABLE_SIZE[1] - 0.01, TABLE_SIZE[1] + 0.01)
	ax_dists.set_xticks(())
	ax_dists.set_yticks(())
	ax_dists.set_aspect('equal')
	divider_dists = make_axes_locatable(ax_dists)
	cax_dists = divider_dists.append_axes('right', size='5%', pad=0.05)
	cb_dists.append(figure_dists.colorbar(contour_dists, cax=cax_dists, orientation='vertical'))

	contour_combined = ax_combined.contourf(xx, yy, combined_preds, cmap=plt.cm.plasma, alpha=.8)
	ax_combined.set_xlim(-TABLE_SIZE[0] - 0.01, TABLE_SIZE[0] + 0.01)
	ax_combined.set_ylim(-TABLE_SIZE[1] - 0.01, TABLE_SIZE[1] + 0.01)
	ax_combined.set_xticks(())
	ax_combined.set_yticks(())
	ax_combined.set_aspect('equal')
	divider_combined = make_axes_locatable(ax_combined)
	cax_combined = divider_combined.append_axes('right', size='5%', pad=0.05)
	cb_combined.append(figure_combined.colorbar(contour_combined, cax=cax_combined, orientation='vertical'))

def draw_checkpoint(model, model_name, epoch=None, suffix=None, threshold=0.05):
	test_obj_grid_num = 5
	figure_ik, ax_ik = plt.subplots(test_obj_grid_num, test_obj_grid_num, figsize=(15,15))
	figure_dists, ax_dists = plt.subplots(test_obj_grid_num, test_obj_grid_num, figsize=(15,15))
	figure_combined, ax_combined = plt.subplots(test_obj_grid_num, test_obj_grid_num, figsize=(15,15))
	cb_ik = []
	cb_dists = []
	cb_combined = []

	device = DEVICE if suffix is None else torch.device("cpu")
	test_objs = test_obj_grid_num**2

	h = 0.01
	xx, yy = np.meshgrid(np.arange(-TABLE_SIZE[0], TABLE_SIZE[0], h),
						 np.arange(-TABLE_SIZE[1], TABLE_SIZE[1], h))
	push_to_xy = np.c_[xx.ravel(), yy.ravel()]
	num_test_pts = push_to_xy.shape[0]

	model.eval()
	with torch.no_grad():
		for o in range(test_objs):
			ax_row = o//test_obj_grid_num
			ax_col = o % test_obj_grid_num

			pidx = np.random.randint(X_test.shape[0])
			obj_props = X_test[pidx, :-3].cpu().numpy()[None, :]

			draw_and_run_model(
				model, obj_props,
				figure_ik, figure_dists, figure_combined,
				ax_ik[ax_row, ax_col], ax_dists[ax_row, ax_col], ax_combined[ax_row, ax_col],
				cb_ik, cb_dists, cb_combined,
				device, threshold=threshold
			)

		plt.tight_layout()
		# plt.show()
		checkpoint_filename = model_name
		if suffix is None:
			checkpoint_filename += '-epoch_{}'.format(epoch)
		else:
			checkpoint_filename += suffix

		figure_ik.savefig('posterior-nostart-maxfactor-ik-' + checkpoint_filename + '.png', bbox_inches='tight')
		figure_dists.savefig('posterior-nostart-maxfactor-dists-' + checkpoint_filename + '.png', bbox_inches='tight')
		figure_combined.savefig('posterior-nostart-maxfactor-combined-' + checkpoint_filename + '.png', bbox_inches='tight')
		for r in range(test_obj_grid_num):
			for c in range(test_obj_grid_num):
				ax_ik[r, c].cla()
				ax_dists[r, c].cla()
				ax_combined[r, c].cla()
		[cb.remove() for cb in cb_ik]
		[cb.remove() for cb in cb_dists]
		[cb.remove() for cb in cb_combined]
		del cb_ik[:]
		del cb_dists[:]
		del cb_combined[:]

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
				bce_loss_ik = bce(combined_pred[:, 0], y_batch[:, 0])
				bce_loss_ik = bce_loss_ik * loss_mask_batch[:, 0]
				# mse_loss = mse(combined_pred[:, 1], y_batch[:, 1])
				# mse_loss = mse_loss * loss_mask_batch[:, 1]
				bce_loss_disc = bce(combined_pred[:, 1], y_batch[:, 1])
				bce_loss_disc = bce_loss_disc * loss_mask_batch[:, 1]
				loss = bce_loss_ik + bce_loss_disc
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
			eval_disc_loss = F.binary_cross_entropy(combined_pred_eval[:, 1], y_val[:, 1], reduction='none') * loss_mask_val[:, 1]
			eval_loss = eval_ik_loss + eval_disc_loss
			eval_loss = eval_loss.mean().cpu().numpy()
			if eval_loss < best_eval_loss:
				# print('\tAccuracy improved at epoch {}'.format(epoch))
				best_eval_loss = eval_loss
				best_weights = copy.deepcopy(model.state_dict())
				last_update_epoch = epoch
			elif ((last_update_epoch >= 0) and epoch > last_update_epoch + n_epochs//10):
				print('Training stopped at epoch {}. Last update was at epoch {}.'.format(epoch, last_update_epoch))
				break

		if (epoch % 50 == 0):
			draw_checkpoint(model, model_name, epoch=epoch)

	# restore model and return best accuracy
	model.eval()
	model.to(torch.device("cpu"))
	with torch.no_grad():
		draw_checkpoint(model1, model_name, epoch=-1, suffix='-double_prob-end-test')
		torch.save(model.state_dict(), 'models/' + model_name + '_end-double_prob.pth')

		model.load_state_dict(best_weights)
		torch.save(model.state_dict(), 'models/' + model_name + '_best-double_prob.pth')
		draw_checkpoint(model1, model_name, epoch=-1, suffix='-double_prob-best-test')

		x_sample = X_val[0].unsqueeze(0).to(torch.device("cpu"))
		print()
		print('x_sample: ', x_sample)
		print('x_sample prediction: ', model(x_sample))
		print('ground truth prediction: ', y_val[0].unsqueeze(0))
		print()

		# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
		traced_model = torch.jit.trace(model, (x_sample))
		torch.jit.save(traced_model, 'models/' + model_name + '_best_traced-double_prob.pth')

		model.load_state_dict(torch.load('models/' + model_name + '_end-double_prob.pth'))
		traced_model = torch.jit.trace(model, (x_sample))
		torch.jit.save(traced_model, 'models/' + model_name + '_end_traced-double_prob.pth')

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
	y.loc[y['r'] <= 0, 'r'] = 1 # push IK succeeded => class 1
	y.loc[y['r'] == 2, 'r'] = 0 # push IK failed => class 0

	# compute distance between desired and achieved poses in polar coordinates
	final_pose_rot = all_data.iloc[:, 29:38].values.reshape(all_data.shape[0], 3, 3).transpose(0, 2, 1)
	_, _, yaws = get_euler_from_R_tensor(final_pose_rot)
	yaw_discrepancy = shortest_angle_dist(yaws, all_data.loc[:, 'o_oyaw'].values)
	yaw_discrepancy[all_data['r'] != 0] = 0.0

	ang_dist = shortest_angle_dist(y.loc[:, 'm_dir_ach'], X.loc[:, 'm_dir_des'])
	polar_dist = y.loc[:, 'm_dist_ach']**2 + X.loc[:, 'm_dist_des']**2 - 2*y.loc[:, 'm_dist_ach']*X.loc[:, 'm_dist_des']*np.cos(ang_dist)
	polar_dist = np.sqrt(polar_dist)
	pose_dist = polar_dist
	y.insert(loc=3, column='pose_dist', value=pose_dist)

	X = X.drop(['m_dir_des', 'm_dist_des'], axis=1)
	y = y.drop(['m_dir_ach', 'm_dist_ach'], axis=1)

	# augment data
	no_disc_rows = all_data['r'] != 0
	X_no_disc = copy.deepcopy(X[no_disc_rows])
	y_no_disc = copy.deepcopy(y[no_disc_rows])
	y_no_disc = y_no_disc.astype(int)

	disc_rows = all_data['r'] == 0
	X_disc = copy.deepcopy(X[disc_rows])
	y_disc = copy.deepcopy(y[disc_rows])

	h, disc_levels = np.histogram(y_disc.iloc[:,1], bins=20)
	X_disc_aug = None
	y_disc_aug = None
	for l in disc_levels:
		y_new = (y_disc.iloc[:, 1] < l).astype(int)
		y_new = np.hstack([y_disc.iloc[:, 0].values[:, None], y_new.values[:, None]])
		X_new = np.hstack([X_disc, l * np.ones(X_disc.shape[0])[:, None]])

		if X_disc_aug is None:
			X_disc_aug = X_new
			y_disc_aug = y_new
		else:
			X_disc_aug = np.vstack([X_disc_aug, X_new])
			y_disc_aug = np.vstack([y_disc_aug, y_new])

	no_disc_new_col = np.random.choice(disc_levels, size=X_no_disc.shape[0]) * np.ones(X_no_disc.shape[0])
	X_no_disc_new_col = np.hstack([X_no_disc, no_disc_new_col[:, None]])

	X_all = np.vstack([X_disc_aug, X_no_disc_new_col])
	y_all = np.vstack([y_disc_aug, y_no_disc])

	# compute loss mask - task 2 (distance/discrepancy prediction) is only if we sucessfully simulated
	loss_mask = np.ones(y_all.shape)
	loss_mask[X_disc.shape[0]:X_disc_aug.shape[0], 0] = 0.0
	loss_mask[X_disc_aug.shape[0]:, 1] = 0.0
	y_all = np.hstack([y_all, loss_mask])

	# Convert to 2D PyTorch tensors
	X_torch = torch.tensor(X_all, dtype=torch.float32).to(DEVICE)
	y_torch = torch.tensor(y_all, dtype=torch.float32).to(DEVICE)
	# train-test split: Hold out the test set for final model evaluation
	X_train, X_test, y_train, y_test = train_test_split(X_torch, y_torch, train_size=0.75, shuffle=True)
	loss_mask_train = y_train[:, -2:]
	y_train = y_train[:, :-2]
	loss_mask_test = y_test[:, -2:]
	y_test = y_test[:, :-2]

	# network params
	in_dim = X.shape[1]-1
	h_sizes = [
				[32, 32, 32],
				# [256, 256, 256],
				# [32, 256, 32],
				# [128, 64, 32],
				# [64, 32, 64]
				# [128, 256, 128],
			]
	activation = 'relu'

	for H in range(len(h_sizes)):
		model_name = '[{}]'.format(','.join(str(x) for x in h_sizes[H]))
		print("Train model: " + model_name)

		layers = len(h_sizes[H]) + 1
		model1 = PushSuccessNet(threshold=True)
		model1.initialise(in_dim, activation=activation, layers=layers, h_sizes=h_sizes[H])
		loss = model_train_multi_head(model1, model_name, X_train, y_train, loss_mask_train, X_test, y_test, loss_mask_test)
		# print(model1)
		print("Final model1 loss: {:.2f}".format(loss))

		# model1.load_state_dict(torch.load('models/' + model_name + '_best-double_prob.pth'))
		# test_model(model1)
