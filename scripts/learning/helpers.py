import numpy as np
from matplotlib.path import Path
import matplotlib.patches as patches
import pandas as pd

from constants import *

def draw_object(axis, obj, alpha=1, zorder=3, colour='b'):
	# draw object
	obj_shape = obj[0, 4]
	ec = colour
	fc = colour
	fill = False
	obj_cent = obj[0, 0:2]

	if (obj_shape == 0): # rectangle
		obj_extents = obj[0, 5:7]
		obj_pts = np.vstack([   obj_cent - (obj_extents * [1, 1]),
								obj_cent - (obj_extents * [-1, 1]),
								obj_cent - (obj_extents * [-1, -1]),
								obj_cent - (obj_extents * [1, -1]),
								obj_cent - (obj_extents * [1, 1])]) # axis-aligned

		R = np.array([
				[np.cos(obj[0, 3]), -np.sin(obj[0, 3])],
				[np.sin(obj[0, 3]), np.cos(obj[0, 3])]]) # rotation matrix
		obj_pts = obj_pts - obj_cent # axis-aligned, at origin
		obj_pts = np.dot(obj_pts, R.T) # rotate at origin
		obj_pts = obj_pts + obj_cent # translate back

		path = Path(obj_pts, CODES)
		obj_rect = patches.PathPatch(path, ec=ec, fc=fc, lw=2, fill=fill,
							alpha=alpha, zorder=zorder)

		axis.add_artist(obj_rect)

	elif (obj_shape == 1): # circle
		obj_rad = obj[0, 5]
		obj_circ = patches.Circle(
					obj_cent, radius=obj_rad,
					ec=ec, fc=fc, lw=2, fill=fill,
					alpha=alpha, zorder=zorder)

		axis.add_artist(obj_circ)

	# axis.scatter(obj_cent[0], obj_cent[1], c=colour, marker='*', zorder=3)

def draw_base_rect(axis, alpha=1, zorder=1):
	# draw shelf base rectangle
	base_bl = -TABLE_SIZE
	base_rect = patches.Rectangle(
								(base_bl[0], base_bl[1]),
								2 * TABLE_SIZE[0], 2 * TABLE_SIZE[1],
								linewidth=2, edgecolor='k', facecolor='none',
								alpha=alpha, zorder=zorder)
	axis.add_artist(base_rect)

def normalize_angle(a):
	if (np.any(np.fabs(a) > 2*np.pi)):
		if (type(a) in FLOATS):
			a = np.fmod(a, 2*np.pi)
		else:
			r = np.where(np.fabs(a) > 2*np.pi)
			a[r[0]] = np.fmod(a[r[0]], 2*np.pi)
	while (np.any(a < -np.pi)):
		if (type(a) in FLOATS):
			a += 2*np.pi
		else:
			r = np.where(a < -np.pi)
			a[r[0]] += 2*np.pi
	while (np.any(a > np.pi)):
		if (type(a) in FLOATS):
			a -= 2*np.pi
		else:
			r = np.where(a > np.pi)
			a[r[0]] -= 2*np.pi
	return a

def shortest_angle_diff(af, ai):
	return normalize_angle(af - ai)

def shortest_angle_dist(af, ai):
	return np.fabs(shortest_angle_diff(af, ai))

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

def process_data(filename=None):
	# Read data
	datafile = filename if filename is not None else "../../dat/push_data/push_data_with_final_object_pose.csv"
	data = pd.read_csv(datafile)

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
