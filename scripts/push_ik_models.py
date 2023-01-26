# usual imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# scikit imports
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

import matplotlib.cm as cm
from matplotlib.path import Path
import matplotlib.patches as patches

TABLE = np.array([0.78, -0.5])

def robot_ik_model():
	P = pd.read_csv("../dat/PUSHES_IK_2s.csv")
	P = P[P.x1 != -1]

	# we do not want to use these yaw columns for training
	P = P.drop('yaw0', axis=1)
	P = P.drop('yaw1', axis=1)

	# scale data to be centered around (0, 0) being the shelf center
	P.loc[:, 'x0'] -= TABLE[0]
	P.loc[:, 'x1'] -= TABLE[0]
	P.loc[:, 'y0'] -= TABLE[1]
	P.loc[:, 'y1'] -= TABLE[1]

	# group all ik failures under one label
	P.loc[P.result > 0, 'result'] = 1

	# get train/test data
	X_train, X_test, y_train, y_test = train_test_split(P.loc[:, ['x0', 'y0', 'x1', 'y1']], P.result, random_state=0, test_size=0.000001)

	# # setup training pipeline
	# rbf = RBFSampler()
	# clf = SGDClassifier()
	# pipe = Pipeline(steps=[('rbf', rbf), ('clf', clf)])

	# # grid search
	# grid = {
	# 	# 'rbf__gamma' : list(np.logspace(-4, 4, 25)) + ['scale'],
	#     'clf__loss': ['log', 'hinge', 'squared_hinge', 'modified_huber'],
	#     'clf__penalty': ['l1', 'l2'],
	#     'clf__alpha': np.logspace(-4, 4, 10),
	# }

	# gs = GridSearchCV(pipe, grid, n_jobs=-1, verbose=1)
	# gs.fit(X_train, y_train)
	# print("Best parameter (CV score=%0.3f):" % gs.best_score_)
	# print(gs.best_params_)
	# best = gs.best_estimator_
	# print(best.score(X_test, y_test))

	# # without rbf sampler
	# # best params found earlier: {'loss': 'squared_hinge', 'penalty': 'l2', 'alpha': 0.000774263682681127}
	# best = SGDClassifier(loss='squared_hinge', penalty='l2', alpha=0.000774263682681127)
	# best.fit(X_train, y_train)
	# print(best.score(X_test, y_test))

	# with rbf sampler
	# best params found earlier: {'clf__alpha': 0.0001, 'clf__penalty': 'l1', 'clf__loss': 'modified_huber'}
	rbf = RBFSampler(gamma=10.0)
	clf = SGDClassifier(alpha=0.0001, penalty='l1', loss='modified_huber')
	best = Pipeline(steps=[('rbf', rbf), ('clf', clf)])
	best.fit(X_train, y_train)
	print(best.score(X_test, y_test))

	# from joblib import dump
	# dump(best, '../dat/push_ik.joblib')

	FIG, AX = plt.subplots(figsize=(13, 13))
	cmap = cm.get_cmap('Dark2')

	# setup legend
	push_legend = []
	failure_modes = {
		1: 'success',
		2: 'fail'
	}
	for i in failure_modes:
		push_legend.append(patches.Patch(color=cmap(i), label=failure_modes[i]))

	# draw shelf base rectangle
	codes = [
			Path.MOVETO,
			Path.LINETO,
			Path.LINETO,
			Path.LINETO,
			Path.CLOSEPOLY,
		]
	base_extents = np.array([0.3, 0.4])
	base_bl = TABLE
	base_bl = base_bl - base_extents
	base_rect = patches.Rectangle(
								(base_bl[0], base_bl[1]),
								2 * base_extents[0], 2 * base_extents[1],
								linewidth=2, edgecolor='k', facecolor='none',
								alpha=1, zorder=2)
	AX.add_artist(base_rect)
	AX.legend(handles=push_legend)
	AX.axis('equal')

	# x0 = 0.2
	# y0 = 0.2
	# for dx in range(-1, 2):
	# 	for dy in range(-1, 2):
	# 		AX.scatter(TABLE[0] + dx*x0, TABLE[1] + dy*y0, s=100, c='g', zorder=3, marker='*')
	# 		for x1 in np.arange(-base_extents[0], base_extents[0], step=0.01):
	# 			for y1 in np.arange(-base_extents[1], base_extents[1], step=0.01):
	# 				test_pt = np.array([dx*x0, dy*y0, x1, y1])
	# 				C = best.predict(test_pt[None, :])[0]
	# 				AX.scatter(x1 + TABLE[0], y1 + TABLE[1], c=[cmap(C+1)], zorder=1)

	# 		plt.savefig('({},{}).png'.format(int((TABLE[0] + dx*x0)*100), int((TABLE[1] + dy*y0)*100)), bbox_inches='tight')
	# 		# plt.show()
	# 		plt.cla()
	# 		AX = plt.gca()
	# 		FIG = plt.gcf()
	# 		FIG.set_size_inches(13, 13)

	x0 = 0.53
	y0 = -0.62
	AX.scatter(x0, y0, s=50, c='gold', zorder=3, marker='*')
	# draw predicted points
	for x1 in np.arange(-base_extents[0], base_extents[0], step=0.01):
		for y1 in np.arange(-base_extents[1], base_extents[1], step=0.01):
			test_pt = np.array([x0 - TABLE[0], y0 - TABLE[1], x1, y1])
			C = best.predict(test_pt[None, :])[0]
			AX.scatter(x1 + TABLE[0], y1 + TABLE[1], c=[cmap(C+1)], zorder=1)
	plt.show()

def object_push_model():
	P = pd.read_csv("../dat/push_data/PUSH_DATA_SIM.csv")

	# we do not want to use these yaw columns for training
	P = P.drop('p_x', axis=1)
	P = P.drop('p_y', axis=1)
	P = P.drop('p_z', axis=1)
	P = P.drop('p_r11', axis=1)
	P = P.drop('p_r21', axis=1)
	P = P.drop('p_r31', axis=1)
	P = P.drop('p_r12', axis=1)
	P = P.drop('p_r22', axis=1)
	P = P.drop('p_r32', axis=1)
	P = P.drop('p_r13', axis=1)
	P = P.drop('p_r23', axis=1)
	P = P.drop('p_r33', axis=1)

	# scale data to be centered around (0, 0) being the shelf center
	P.loc[:, 'o_ox'] -= TABLE[0]
	P.loc[:, 'o_oy'] -= TABLE[1]

	# reassign values
	P.loc[P.r == 0, 'r'] = 1
	P.loc[P.r < 0, 'r'] = 0
	P.loc[P.o_shape == 2, 'o_shape'] = 1

	# get train/test data
	X_train, X_test, y_train, y_test = train_test_split(P.loc[:, ['o_ox','o_oy','o_oz','o_oyaw','o_shape','o_xs','o_ys','o_zs','o_mass','o_mu','m_dir','m_dist']], P.r, random_state=0, test_size=0.000001)

	# # setup training pipeline
	# rbf = RBFSampler()
	# clf = SGDClassifier()
	# pipe = Pipeline(steps=[('rbf', rbf), ('clf', clf)])

	# # grid search
	# grid = {
	# 	# 'rbf__gamma' : list(np.logspace(-4, 4, 25)) + ['scale'], # {'rbf__gamma': 0.1}
	#     'clf__loss': ['log', 'hinge', 'squared_hinge', 'modified_huber'],
	#     'clf__penalty': ['l1', 'l2'],
	#     'clf__alpha': np.logspace(-4, 4, 10),
	# }
	# # {'clf__alpha': 0.0001, 'clf__loss': 'modified_huber', 'clf__penalty': 'l2'}

	# gs = GridSearchCV(pipe, grid, n_jobs=-1, verbose=1)
	# gs.fit(X_train, y_train)
	# print("Best parameter (CV score=%0.3f):" % gs.best_score_)
	# print(gs.best_params_)
	# best = gs.best_estimator_
	# print(best.score(X_test, y_test))

	rbf = RBFSampler(gamma=0.1)
	clf = SGDClassifier(alpha=0.0001, penalty='l2', loss='modified_huber')
	best = Pipeline(steps=[('rbf', rbf), ('clf', clf)])
	best.fit(X_train, y_train)
	print(best.score(X_test, y_test))

	FIG, AX = plt.subplots(figsize=(13, 13))
	cmap = cm.get_cmap('Dark2')

	# setup legend
	push_legend = []
	push_result = {
		2: 'success',
		1: 'fail'
	}
	for i in push_result:
		push_legend.append(patches.Patch(color=cmap(i), label=push_result[i]))

	# draw shelf base rectangle
	codes = [
			Path.MOVETO,
			Path.LINETO,
			Path.LINETO,
			Path.LINETO,
			Path.CLOSEPOLY,
		]
	base_extents = np.array([0.3, 0.4])
	base_bl = TABLE
	base_bl = base_bl - base_extents
	base_rect = patches.Rectangle(
								(base_bl[0], base_bl[1]),
								2 * base_extents[0], 2 * base_extents[1],
								linewidth=2, edgecolor='k', facecolor='none',
								alpha=1, zorder=2)
	AX.add_artist(base_rect)
	AX.legend(handles=push_legend)
	AX.axis('equal')

# objects from 100011
# 12,0,1,0.8154373669513246,-0.7878776846316444,0.86,0.0,0.0,4.354556175210488,0.03,0.06002629182662351,0.09,0.34528948312182245,0.9162874932518031,True
# 14,0,1,0.5228407362622941,-0.5994011525625002,0.9060119187406109,0.0,0.0,4.145721344530907,0.03385179269754395,0.03890820111658584,0.1360119187406109,0.2967886596541951,0.9059203586194849,True
# 15,0,1,0.5265759323356647,-0.44525114248804787,0.86,0.0,0.0,3.162050919275471,0.03,0.037425523935246584,0.09,0.0993357633340925,0.9985290606712391,True
# object from 100069
# 13,0,1,0.5396405594291757,-0.6202659961061698,0.86,0.0,0.0,3.968932883363816,0.05894977023814522,0.06382387226194496,0.09,0.4047051216355124,0.8664349515856553,True

	obj = [0.5396405594291757,-0.6202659961061698,0.86,3.968932883363816,0,0.05894977023814522,0.06382387226194496,0.09,0.4047051216355124,0.8664349515856553]
	# obj = [0.8154373669513246,-0.7878776846316444,0.86,4.354556175210488,0,0.03,0.06002629182662351,0.09,0.34528948312182245,0.9162874932518031]
	# obj = [0.5228407362622941,-0.5994011525625002,0.9060119187406109,4.145721344530907,0,0.03385179269754395,0.03890820111658584,0.1360119187406109,0.2967886596541951,0.9059203586194849]
	# obj = [0.5265759323356647,-0.44525114248804787,0.86,3.162050919275471,0,0.03,0.037425523935246584,0.09,0.0993357633340925,0.9985290606712391]
	for x1 in np.arange(-base_extents[0], base_extents[0], step=0.01):
		for y1 in np.arange(-base_extents[1], base_extents[1], step=0.01):
			m_dir = np.arctan2(y1 + TABLE[1] - obj[1], x1 + TABLE[0] - obj[0])
			m_dist = ((x1 + TABLE[0] - obj[0])**2 + (y1 + TABLE[1] - obj[1])**2)**0.5
			test_pt = obj + [m_dir, m_dist]
			test_pt[0] -= TABLE[0]
			test_pt[1] -= TABLE[1]
			test_pt = np.asarray(test_pt)
			C = best.predict(test_pt[None, :])[0]
			AX.scatter(x1 + TABLE[0], y1 + TABLE[1], c=[cmap(C+1)], zorder=1)
	plt.show()

if __name__ == '__main__':
	object_push_model()
