# usual imports
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.path import Path
import matplotlib.patches as patches

# scikit imports
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier, LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
# for classifier comparison
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

TABLE = np.array([0.78, -0.5])
TABLE_SIZE = np.array([0.3, 0.4])

############################
# Robot push success model #
############################
# Predict if robot IK will succeed from point-to-point,
# no obstacles inside shelf
def robot_ik_model():
	P = pd.read_csv("../dat/PUSHES_IK_2s.csv")
	P = P[P.x1 != -1]

	# we do not want to use these yaw columns for training
	P = P.drop(['yaw0', 'yaw1'], axis=1)

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
	base_extents = TABLE_SIZE
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


###################################
# Object-aware push success model #
###################################
# Predict if robot will succeed in pushing object along (direction, distance)
def object_push_model():
	P = pd.read_csv("../dat/push_data/PUSH_DATA_SIM.csv")

	# we do not want to use these yaw columns for training
	P = P.drop(['p_x','p_y','p_z','p_r11','p_r21','p_r31','p_r12','p_r22','p_r32','p_r13','p_r23','p_r33'], axis=1)

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
	base_extents = TABLE_SIZE
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

###################################
# Classifier Comparison - sklearn #
###################################

def read_push_db(filename, robot=False):
	P = pd.read_csv(filename)

	# # we do not want to use these push pose columns for training
	# P = P.drop(['p_x','p_y','p_z','p_r11','p_r21','p_r31','p_r12','p_r22','p_r32','p_r13','p_r23','p_r33'], axis=1)

	# normalise angles
	sin = np.sin(P.loc[:, 'o_oyaw'])
	cos = np.cos(P.loc[:, 'o_oyaw'])
	P.loc[:, 'o_oyaw'] = np.arctan2(sin, cos)
	sin = np.sin(P.loc[:, 'm_dir_des'])
	cos = np.cos(P.loc[:, 'm_dir_des'])
	P.loc[:, 'm_dir_des'] = np.arctan2(sin, cos)

	# reassign values
	if robot:
		P.loc[P.r > 0, 'r'] = 1 # push IK failed => class 1
		P.loc[P.r <= 0, 'r'] = 0 # push IK succeeded => class 0
	else:
		# P.loc[P.r == 0, 'r'] = 0 # push success => class 0
		P.loc[P.r != 0, 'r'] = 1 # push failed (in sim?) => class 0
	P.loc[P.o_shape == 2, 'o_zs'] /= 2
	P.loc[P.o_shape == 2, 'o_shape'] = 1

	# # drop some additional columns
	# P = P.drop(['o_oz', 'o_shape'], axis=1)

	# add some new columns
	P['delta_x'] = P['o_ox'] + (P['m_dist_des'].values * np.cos(P['m_dir_des'].values))
	P['delta_y'] = P['o_oy'] + (P['m_dist_des'].values * np.sin(P['m_dir_des'].values))
	if robot:
		P.loc[:, 'delta_x'] -= TABLE[0]
		P.loc[:, 'delta_y'] -= TABLE[1]
	else:
		P.loc[:, 'delta_x'] -= P.loc[:, 'o_ox'].values
		P.loc[:, 'delta_y'] -= P.loc[:, 'o_oy'].values

	# scale data to be centered around (0, 0) being the shelf center
	P.loc[:, 'o_ox'] -= TABLE[0]
	P.loc[:, 'o_oy'] -= TABLE[1]

	# get train/test data
	X = None
	if robot:
		P.loc[:, 'p_x'] -= TABLE[0]
		P.loc[:, 'p_y'] -= TABLE[1]
		# X = copy.deepcopy(P.loc[:, ['o_ox','o_oy','delta_x','delta_y']])
		X = copy.deepcopy(P.loc[:, ['p_x','p_y','delta_x','delta_y']])
		# X = copy.deepcopy(P.loc[:, ['p_x','p_y','o_ox','o_oy','delta_x','delta_y']])
	else:
		# X = copy.deepcopy(P.loc[:, ['o_ox','o_oy','o_oz','o_oyaw','o_shape','o_xs','o_ys','o_zs','o_mass','o_mu','m_dir_des','m_dist_des']])
		# X = copy.deepcopy(P.loc[:, ['o_ox','o_oy','o_oyaw','o_xs','o_ys','o_zs','o_mass','o_mu','m_dir_des','m_dist_des']])
		X = copy.deepcopy(P.loc[:, ['o_ox','o_oy','o_oyaw','o_xs','o_ys','o_zs','o_mass','o_mu','delta_x','delta_y']])

	y = copy.deepcopy(P.r)

	return (X, y)

def compare_classifiers():
	h = .01  # step size in the mesh

	robot = True
	D1_X, D1_y = read_push_db("../dat/push_data/pushes_fixed_z_all.csv", robot=robot)
	# D2_X, D2_y = read_push_db("../dat/push_data/pushes_fixed_z_simmed.csv")
	# D3_X = pd.concat([D1_X, D2_X])
	# D3_y = pd.concat([D1_y, D2_y])

	datasets = [
		(D1_X, D1_y),
		# (D2_X, D2_y),
		# (D3_X, D3_y),
	]
	weights = np.bincount(datasets[0][1]).sum()/(2 * np.bincount(datasets[0][1]))
	class_weights = { 0: weights[0] * 4, 1: weights[1] }

	names = [
		# "RBF SVM Grid Search",
		# "Nearest Neighbors",
		"Linear SVM",
		# "RBF SVM",
		# "Gaussian Process",
		# "Decision Tree",
		# "Random Forest",
		# "Neural Net",
		# "AdaBoost",
		# "Naive Bayes",
		# "QDA",
		"Linear Regression",
		"Ridge Regression",
		"Best Robot Only",
	]

	# # grid search
	# svc = SVC()
	# grid = {
	# 	# 'rbf__gamma' : list(np.logspace(-4, 4, 25)) + ['scale'],
	#     'C': [0.1, 1, 10, 100, 1000],
	#     'gamma': [100, 10, 5, 2, 1, 0.1, 0.01, 0.001, 0.0001],
	#     'kernel': ['rbf'],
	# }
	# gs = GridSearchCV(svc, grid, n_jobs=-1, verbose=1)

	rbf_r = RBFSampler(gamma=10.0)
	clf_r = SGDClassifier(alpha=0.0001, penalty='l1', loss='modified_huber')
	best_r = Pipeline(steps=[('rbf', rbf_r), ('clf', clf_r)])

	classifiers = [
		# gs,
		# KNeighborsClassifier(3),
		SVC(kernel="linear", C=1),
		# SVC(gamma=2, C=1, class_weight=class_weights),
		# GaussianProcessClassifier(1.0 * RBF(1.0)),
		# DecisionTreeClassifier(max_depth=5),
		# RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
		# MLPClassifier(alpha=1, max_iter=1000),
		# AdaBoostClassifier(),
		# GaussianNB(),
		# QuadraticDiscriminantAnalysis(),
		LinearRegression(fit_intercept=True, n_jobs=-1),
		Ridge(fit_intercept=True),
		best_r,
		]

	objs = [
		np.array([0.5396405594291757,-0.6202659961061698,0.86,3.968932883363816,0,0.05894977023814522,0.06382387226194496,0.09,0.4047051216355124,0.8664349515856553]),
		np.array([0.5265759323356647,-0.44525114248804787,0.86,3.162050919275471,0,0.03,0.037425523935246584,0.09,0.0993357633340925,0.9985290606712391]),
		np.array([0.696405594291757,-0.702659961061698,0.86,3.968932883363816,0,0.05894977023814522,0.16382387226194496,0.09,0.4047051216355124,0.8664349515856553]),
		np.array([0.619540968861,-0.366610269647,0.86,4.80544933564,0,0.03,0.0713504225141,0.09,0.0734181112928,0.845070518711]),
		np.array([0.8154373669513246,-0.7878776846316444,0.86,4.354556175210488,0,0.03,0.06002629182662351,0.09,0.34528948312182245,0.9162874932518031]),
		np.array([0.5228407362622941,-0.5994011525625002,0.9060119187406109,4.145721344530907,0,0.03385179269754395,0.03890820111658584,0.1360119187406109,0.2967886596541951,0.9059203586194849])
	]

	# figure = plt.figure(figsize=(27, 9))
	figure = plt.figure()
	i = 1
	cb = None
	# iterate over datasets
	for ds_cnt, ds in enumerate(datasets):
		# preprocess dataset, split into training and test part
		X, y = ds
		# X = StandardScaler().fit_transform(X)
		X_train, X_test, y_train, y_test = \
			train_test_split(X, y, test_size=.25, random_state=42)

		x_min, x_max = TABLE[0] - TABLE_SIZE[0], TABLE[0] + TABLE_SIZE[0]
		y_min, y_max = TABLE[1] - TABLE_SIZE[1], TABLE[1] + TABLE_SIZE[1]
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
							 np.arange(y_min, y_max, h))
		test_pts_xy = np.c_[xx.ravel(), yy.ravel()]
		# test_pts_xy[:,0] -= TABLE[0]
		# test_pts_xy[:,1] -= TABLE[1]

		# iterate over classifiers
		for name, clf in zip(names, classifiers):
			print("Train {}".format(name))
			# clf = make_pipeline(StandardScaler(), clf)
			clf = Pipeline(steps=[('rbf', rbf_r), ('clf', clf)]) if 'Best' not in name else clf
			clf.fit(X_train, y_train)
			# if 'Grid' in name:
			# 	import ipdb
			# 	ipdb.set_trace()
			# 	clf = clf.best_estimator_
			score = clf.score(X_test, y_test)
			print("{} trained! Score = {}".format(name, score))

			if robot:
				for obj_cnt, obj in enumerate(objs):
					ax = plt.subplot(len(classifiers), len(objs), i)

					test_pts = np.array([obj[0], obj[1]])
					test_pts = np.repeat(test_pts[None, :], test_pts_xy.shape[0], axis=0)
					test_pts = np.hstack([test_pts, test_pts_xy])
					test_pts[:,0] -= TABLE[0]
					test_pts[:,1] -= TABLE[1]
					test_pts[:,2] -= TABLE[0]
					test_pts[:,3] -= TABLE[1]

					if hasattr(clf, "decision_function"):
						Z = clf.decision_function(test_pts)
					elif hasattr(clf, "predict_proba"):
						Z = clf.predict_proba(test_pts)[:, 1]
					else:
						Z = clf.predict(test_pts)

					# Put the result into a color plot
					Z = Z.reshape(xx.shape)
					if cb is None:
						cb = ax.contourf(xx, yy, Z, cmap=plt.cm.RdGy, alpha=.8)
					else:
						ax.contourf(xx, yy, Z, cmap=plt.cm.RdGy, alpha=.8)

					ax.scatter(test_pts[0,0] + TABLE[0], test_pts[0,1] + TABLE[1], s=50, marker='*', c='gold')

					ax.set_xlim(left=xx.min() - 0.05, right=xx.max() + 0.05)
					ax.set_ylim(bottom=yy.min() - 0.05, top=yy.max() + 0.05)
					ax.set_xticks(())
					ax.set_yticks(())
					ax.set_aspect('equal')
					if ds_cnt == 0 and (i % len(objs) == 1):
						ax.set_title(name)
					ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
							size=15, horizontalalignment='right')
					i += 1
			else:
				# Plot the decision boundary. For that, we will assign a color to each
				# point in the mesh [x_min, x_max]x[y_min, y_max].
				for obj_cnt, obj in enumerate(objs):
					ax = plt.subplot(len(classifiers), len(objs), i)

					# test_m_dirs = np.arctan2(test_pts_xy[:,1] - obj[1], test_pts_xy[:,0] - obj[0])
					# test_m_dists = ((test_pts_xy[:,0] - obj[0])**2 + (test_pts_xy[:,1] - obj[1])**2)**0.5
					test_pts_xy[:,0] -= obj[0]
					test_pts_xy[:,1] -= obj[1]
					objs_stack = np.repeat(obj[None, :], test_pts_xy.shape[0], axis=0)
					# test_pts = np.hstack([objs_stack, test_m_dirs[:, None], test_m_dists[:, None]])
					test_pts = np.hstack([objs_stack, test_pts_xy])
					test_pts[:, 0] -= TABLE[0]
					test_pts[:, 1] -= TABLE[1]
					sin = np.sin(test_pts[:, 3])
					cos = np.cos(test_pts[:, 3])
					test_pts[:, 3] = np.arctan2(sin, cos)
					test_pts = np.delete(test_pts, [2, 4], axis=1)

					if hasattr(clf, "decision_function"):
						Z = clf.decision_function(test_pts)
					elif hasattr(clf, "predict_proba"):
						Z = clf.predict_proba(test_pts)[:, 1]
					else:
						Z = clf.predict(test_pts)

					# Put the result into a color plot
					Z = Z.reshape(xx.shape)
					if cb is None:
						cb = ax.contourf(xx, yy, Z, cmap=plt.cm.RdGy, alpha=.8)
					else:
						ax.contourf(xx, yy, Z, cmap=plt.cm.RdGy, alpha=.8)

					# draw shelf base rectangle
					codes = [
							Path.MOVETO,
							Path.LINETO,
							Path.LINETO,
							Path.LINETO,
							Path.CLOSEPOLY,
						]
					base_bl = TABLE - TABLE_SIZE
					base_rect = patches.Rectangle(
												(base_bl[0], base_bl[1]),
												2 * TABLE_SIZE[0], 2 * TABLE_SIZE[1],
												linewidth=2, edgecolor='k', facecolor='none',
												alpha=1, zorder=2)
					ax.add_artist(base_rect)

					# draw object
					obj_shape = obj[4]
					ec = 'b'
					fc = 'b'
					fill = False
					obj_cent = obj[0:2]

					if (obj_shape == 0): # rectangle
						obj_extents = obj[5:7]
						obj_pts = np.vstack([	obj_cent - (obj_extents * [1, 1]),
												obj_cent - (obj_extents * [-1, 1]),
												obj_cent - (obj_extents * [-1, -1]),
												obj_cent - (obj_extents * [1, -1]),
												obj_cent - (obj_extents * [1, 1])]) # axis-aligned

						R = np.array([
								[np.cos(obj[3]), -np.sin(obj[3])],
								[np.sin(obj[3]), np.cos(obj[3])]]) # rotation matrix
						obj_pts = obj_pts - obj_cent # axis-aligned, at origin
						obj_pts = np.dot(obj_pts, R.T) # rotate at origin
						obj_pts = obj_pts + obj_cent # translate back

						path = Path(obj_pts, codes)
						obj_rect = patches.PathPatch(path, ec=ec, fc=fc, lw=2, fill=fill,
											alpha=1, zorder=3)

						ax.add_artist(obj_rect)

					elif (obj_shape == 2): # circle
						obj_rad = obj[5]
						obj_circ = patches.Circle(
									obj_cent, radius=obj_rad,
									ec=ec, fc=fc, lw=2, fill=fill,
									alpha=1, zorder=3)

						ax.add_artist(obj_circ)

					ax.set_xlim(left=xx.min() - 0.05, right=xx.max() + 0.05)
					ax.set_ylim(bottom=yy.min() - 0.05, top=yy.max() + 0.05)
					ax.set_xticks(())
					ax.set_yticks(())
					ax.set_aspect('equal')
					if ds_cnt == 0 and (i % len(objs) == 1):
						ax.set_title(name)
					ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
							size=15, horizontalalignment='right')
					i += 1

	# plt.colorbar(cb)
	plt.tight_layout()
	plt.show()


def vis_push_data():
	codes = [
			Path.MOVETO,
			Path.LINETO,
			Path.LINETO,
			Path.LINETO,
			Path.CLOSEPOLY,
		]
	base_bl = TABLE - TABLE_SIZE
	base_rect = patches.Rectangle(
								(base_bl[0], base_bl[1]),
								2 * TABLE_SIZE[0], 2 * TABLE_SIZE[1],
								linewidth=2, edgecolor='k', facecolor='none',
								alpha=1, zorder=2)

	FIG = plt.figure(figsize=(13, 13))
	AX = plt.subplot(1, 1, 1)
	AX.add_artist(base_rect)

	# data = {
	# 			'm_dir' : [5.97057,-1.13238,-0.499256,3.50443,0.530529,1.87608,1.25147,5.69356,3.91114,0.297185,6.05162,3.03643,3.35799,1.36624,6.05406,5.86822,0.520798,0.635112,0.475216,4.14273,4.29956,6.04036,2.53342,4.97596,0.0705303],
	# 			'm_dist' : [0.239911,0.0639314,0.0813124,0.138217,0.212767,0.184396,0.220826,0.211332,0.232508,0.206531,0.167234,0.153184,0.206785,0.134149,0.143504,0.121415,0.28215,0.270935,0.11742,0.242368,0.141954,0.221639,0.249751,0.128959,0.29456],
	# 			'r' : [3,0,0,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]
	# 		}
	# df = pd.DataFrame(data=data)
	df = pd.read_csv("../dat/push_data/one_obj.csv")

	ec = 'b'
	fc = 'b'
	fill = False

	# obj = np.array([0.71679,-0.398789,0.86,2.00804,0,0.0669627,0.03,0.09,0.210207,0.0902199])
	obj_cent = np.array([df['o_ox'][0], df['o_oy'][0]])

	if (df['o_shape'][0] == 0): # rectangle
		obj_extents = np.array([df['o_xs'][0], df['o_ys'][0]])
		obj_pts = np.vstack([	obj_cent - (obj_extents * [1, 1]),
								obj_cent - (obj_extents * [-1, 1]),
								obj_cent - (obj_extents * [-1, -1]),
								obj_cent - (obj_extents * [1, -1]),
								obj_cent - (obj_extents * [1, 1])]) # axis-aligned
		R = np.array([
				[np.cos(df['o_oyaw'][0]), -np.sin(df['o_oyaw'][0])],
				[np.sin(df['o_oyaw'][0]), np.cos(df['o_oyaw'][0])]]) # rotation matrix
		obj_pts = obj_pts - obj_cent # axis-aligned, at origin
		obj_pts = np.dot(obj_pts, R.T) # rotate at origin
		obj_pts = obj_pts + obj_cent # translate back

		path = Path(obj_pts, codes)
		obj_rect = patches.PathPatch(path, ec=ec, fc=fc, lw=2, fill=fill,
							alpha=1, zorder=3)

		AX.add_artist(obj_rect)

	elif (df['o_shape'][0] == 2): # circle
		obj_rad = df['o_xs'][0]
		obj_circ = patches.Circle(
					obj_cent, radius=obj_rad,
					ec=ec, fc=fc, lw=2, fill=fill,
					alpha=1, zorder=3)

		AX.add_artist(obj_circ)

	push_legend = []
	colors = {
		-2: 'darkgray',
		-1: 'lightgray',
		0: 'g',
		2: 'magenta',
		3: 'cyan',
		5: 'red',
	}
	failure_modes = {
		-2: 'sim fail',
		-1: 'no contact',
		0: 'success',
		2: 'obs coll',
		3: 'joint lim',
		5: 'start not reached',
	}
	for i in failure_modes:
		push_legend.append(patches.Patch(color=colors[i], label=failure_modes[i]))

	AX.legend(handles=push_legend)

	for index, row in df.iterrows():
		push_pt_des = obj_cent + (row['m_dist_des'] * np.array([np.cos(row['m_dir_des']), np.sin(row['m_dir_des'])]))
		push_pt_ach = obj_cent + (row['m_dist_ach'] * np.array([np.cos(row['m_dir_ach']), np.sin(row['m_dir_ach'])]))
		print(push_pt_des, push_pt_ach, row['r'])
		AX.plot([push_pt_des[0], push_pt_ach[0]], [push_pt_des[1], push_pt_ach[1]], c=colors[row['r']])
		AX.scatter(push_pt_des[0], push_pt_des[1], s=50, c=colors[row['r']], marker='D')
		AX.scatter(push_pt_ach[0], push_pt_ach[1], s=50, c=colors[row['r']], marker='*')

	AX.set_xlim(TABLE[0] - TABLE_SIZE[0] - 0.05, TABLE[0] + TABLE_SIZE[0] + 0.05)
	AX.set_ylim(TABLE[1] - TABLE_SIZE[1] - 0.05, TABLE[1] + TABLE_SIZE[1] + 0.05)
	# AX.set_xticks(())
	# AX.set_yticks(())
	AX.axis('equal')

	plt.show()

if __name__ == '__main__':
	vis_push_data()
