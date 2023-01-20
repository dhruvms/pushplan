# usual imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

TABLE = np.array([0.78, -0.5])
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

# scikit imports
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

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

import matplotlib.cm as cm
from matplotlib.path import Path
import matplotlib.patches as patches

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
