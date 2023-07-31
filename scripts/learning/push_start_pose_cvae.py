import copy
import matplotlib.pyplot as plt
import numpy as np

import torch
from sklearn.model_selection import train_test_split

from helpers import *
from constants import *
from models import *

if __name__ == '__main__':
	all_data = process_data()

	# get training data - start pose
	# ipdb> p C.columns.tolist()
	# ['o_ox', 'o_oy', 'o_oz', 'o_oyaw', 'o_shape', 'o_xs', 'o_ys', 'o_zs', 'o_mass', 'o_mu', 'm_dir_ach', 'm_dist_ach']
	# ipdb> p X.columns.tolist()
	# ['s_x', 's_y', 's_z', 's_yaw']

	C = copy.deepcopy(all_data.loc[all_data.r == 0, [all_data.columns.tolist()[i] for i in list(range(0, 10)) + list(range(24, 26))]])
	X = copy.deepcopy(all_data.loc[all_data.r == 0, [all_data.columns.tolist()[i] for i in list(range(10, 22))]])
	# extract only yaw angle from rotation matrices of poses
	X_rot = X.iloc[:, 3:].values.reshape(X.shape[0], 3, 3).transpose(0, 2, 1)
	_, _, yaws = get_euler_from_R_tensor(X_rot)
	X = X.drop(['s_r11', 's_r21', 's_r31', 's_r12', 's_r22', 's_r32', 's_r13', 's_r23', 's_r33'], axis=1)
	X.insert(loc=3, column='s_yaw', value= yaws)
	# Convert to 2D PyTorch tensors
	C = torch.tensor(C.values, dtype=torch.float32).to(DEVICE)
	X = torch.tensor(X.values, dtype=torch.float32).to(DEVICE)
	# train-test split: Hold out the test set for start model evaluation
	C_train, C_test, X_train, X_test = train_test_split(C, X, train_size=0.95, shuffle=True)
	# X_train = X_train[:, :-3] # drop final column of rotation matrix to be predicted

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
			model1 = CVAE()
			# model1 = PoseCVAE()
			model1.initialise(in_dim, cond_dim, latent_dims[L], activation=activation, layers=layers, h_sizes=h_sizes[H])
			# print(model1)
			model_train_cvae(model1, C_train, X_train, latent_dims[L], C_test, X_test, loss_fn_cvae, eval_fn_cvae, epochs=100)
			# model_train_cvae(model1, C_train, X_train, latent_dims[L], C_test, X_test, loss_fn_pose_cvae, eval_fn_pose_cvae, epochs=100)

			model1.eval()

			test_plots = 5
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

					xpred = model1.decode(ztest_torch, ctest_torch).cpu().numpy()
					# xpred = model1.decode(ztest_torch, ctest_torch)
					# xpred_rot = model1.get_rotation_matrix(xpred[:, -6:])
					# xpred = xpred[:, :3].cpu().numpy()
					# xpred_rot = xpred_rot.cpu().numpy()

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
						ayaw = xpred[i, 3]
						# ayaw = get_yaw_from_R(xpred_rot[i])
						ax.arrow(xpred[i, 0], xpred[i, 1], asize * np.cos(ayaw), asize * np.sin(ayaw),
									length_includes_head=True, head_width=0.02, head_length=0.02,
									ec='gold', fc='gold', alpha=0.8)

					xtest = X_test[pidx, :].cpu().numpy()
					true_yaw = xtest[3]
					# R = make_rotation_matrix(xtest[3:])
					# true_yaw = get_yaw_from_R(R)
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
				plt.savefig('startpose_4dstart-[{}]'.format(','.join(str(x) for x in h_sizes[H])) + '-[{}].png'.format(latent_dims[L]), bbox_inches='tight')
				[ax.cla() for ax in axes]
