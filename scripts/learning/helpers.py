import numpy as np
from matplotlib.path import Path
import matplotlib.patches as patches

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
