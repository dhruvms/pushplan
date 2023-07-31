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

	# get training data - final pose
	# ipdb> p C.columns.tolist()
	# ['o_ox', 'o_oy', 'o_oz', 'o_oyaw', 'o_shape', 'o_xs', 'o_ys', 'o_zs', 'o_mass', 'o_mu', 's_x', 's_y', 's_z', 's_yaw', 'm_dir_ach', 'm_dist_ach']
	# ipdb> p X.columns.tolist()
	# ['e_x', 'e_y', 'e_yaw']

	C = copy.deepcopy(all_data.loc[all_data.r == 0, [all_data.columns.tolist()[i] for i in list(range(0, 22)) + list(range(24, 26))]])
	X = copy.deepcopy(all_data.loc[all_data.r == 0, [all_data.columns.tolist()[i] for i in list(range(-13, -1))]])
	# extract only yaw angle from rotation matrices of poses
	C_rot = C.iloc[:, 13:22].values.reshape(C.shape[0], 3, 3).transpose(0, 2, 1)
	_, _, yaws = get_euler_from_R_tensor(C_rot)
	C = C.drop(['s_r11', 's_r21', 's_r31', 's_r12', 's_r22', 's_r32', 's_r13', 's_r23', 's_r33'], axis=1)
	C.insert(loc=13, column='s_yaw', value= yaws)
	X_rot = X.iloc[:, 3:].values.reshape(X.shape[0], 3, 3).transpose(0, 2, 1)
	_, _, yaws = get_euler_from_R_tensor(X_rot)
	X = X.drop(['e_z', 'e_r11', 'e_r21', 'e_r31', 'e_r12', 'e_r22', 'e_r32', 'e_r13', 'e_r23', 'e_r33'], axis=1)
	X.insert(loc=2, column='e_yaw', value= yaws)
	# Convert to 2D PyTorch tensors
	C = torch.tensor(C.values, dtype=torch.float32).to(DEVICE)
	X = torch.tensor(X.values, dtype=torch.float32).to(DEVICE)
	# train-test split: Hold out the test set for final model evaluation
	C_train, C_test, X_train, X_test = train_test_split(C, X, train_size=0.85, shuffle=True)

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
			model1.initialise(in_dim, cond_dim, latent_dims[L], activation=activation, layers=layers, h_sizes=h_sizes[H])
			# print(model1)
			model_train_cvae(model1, C_train, X_train, latent_dims[L], C_test, X_test, loss_fn_cvae, eval_fn_cvae, epochs=100)

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

					xtest = X_test[pidx, :].cpu().numpy()
					xpred = model1.decode(ztest_torch, ctest_torch).cpu().numpy()

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
					onew[0, 3] = xtest[2]
					draw_object(ax, onew, alpha=1, zorder=2, colour='cyan')

					for i in range(xpred.shape[0]):
						obj = copy.deepcopy(ctest)
						obj[0, 0] = xpred[i, 0]
						obj[0, 1] = xpred[i, 1]
						obj[0, 3] = xpred[i, 2]
						draw_object(ax, obj, alpha=0.5, zorder=3, colour='magenta')

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
				plt.savefig('[{}]'.format(','.join(str(x) for x in h_sizes[H])) + '-[{}]-finalpose_4dstart.png'.format(latent_dims[L]), bbox_inches='tight')
				[ax.cla() for ax in axes]
