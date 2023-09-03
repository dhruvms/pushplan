#!/usr/bin/env python3
import numpy as np
import os, glob, random # for random textures to objects
import time
import math
import collections
import multiprocessing as mp
import copy

import rospy
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client as bc

from comms.srv import AddObject, AddObjectRequest, AddObjectResponse
from comms.srv import AddYCBObject, AddYCBObjectResponse
from comms.srv import ResetSimulation, ResetSimulationRequest, ResetSimulationResponse
from comms.srv import AddRobot, AddRobotResponse
from comms.srv import SetRobotState, SetRobotStateResponse
from comms.srv import ResetArm, ResetArmResponse
from comms.srv import CheckScene, CheckSceneResponse
from comms.srv import ResetScene, ResetSceneResponse, ResetSceneRequest
from comms.srv import SetColours, SetColoursResponse
from comms.srv import ExecTraj, ExecTrajResponse
from comms.srv import SimPushes, SimPushesResponse
from comms.msg import ObjectPose, ObjectsPoses

from utils import *

class BulletSim:
	def __init__(self, gui, shadows=False):
		rospy.init_node('BulletSim')

		self.sims = []
		self.sim_datas = {}
		self.fridge = rospy.get_param("/fridge", False)

		self.table_rgba = list(np.random.rand(3)) + [1]
		self.table_specular = list(np.random.rand(3))

		self.camera_set = False
		self.gui = gui
		self.obj_copies = rospy.get_param("~sim/obj_copies", 1)
		self.robot_collision_group = (2**self.obj_copies) - 1
		self.no_collision_group = 2**self.obj_copies
		self.grasped_mass = 5e-2

		if self.gui:
			self.cpu_count = 1
			succ_frac = rospy.get_param("~sim/succ_frac", 1.0)
			self.succ_needed = max(1, math.floor(succ_frac * self.cpu_count * self.obj_copies))

			sim, info = self.setupSim(shadows)
			self.sims.append(sim)
			self.sim_datas[0] = info
		else:
			cpus = rospy.get_param("~sim/cpus", 0)
			self.cpu_count = cpus if cpus > 0 else mp.cpu_count() - 1
			succ_frac = rospy.get_param("~sim/succ_frac", 1.0)
			self.succ_needed = max(1, math.floor(succ_frac * self.cpu_count * self.obj_copies))

			for i in range(self.cpu_count):
				sim, info = self.setupSim(shadows)
				self.sims.append(sim)
				self.sim_datas[i] = info

		self.add_object = rospy.Service('add_object',
										AddObject, self.AddObject)
		self.add_robot = rospy.Service('add_robot',
										AddRobot, self.AddRobot)
		self.add_ycb_object = rospy.Service('add_ycb_object',
										AddYCBObject, self.AddYCBObject)
		self.set_robot_state = rospy.Service('set_robot_state',
										SetRobotState, self.SetRobotStateAll)
		self.reset_arm = rospy.Service('reset_arm',
										ResetArm, self.ResetArmAll)
		self.check_scene = rospy.Service('check_scene',
										CheckScene, self.CheckScene)
		self.reset_scene = rospy.Service('reset_scene',
										ResetScene, self.ResetScene)
		self.set_colours = rospy.Service('set_colours',
										SetColours, self.SetColours)
		self.reset_simulation = rospy.Service('reset_simulation',
										ResetSimulation, self.ResetSimulation)
		self.exec_traj = rospy.Service('exec_traj',
												ExecTraj, self.ExecTraj)
		self.sim_pushes = rospy.Service('sim_pushes',
												SimPushes, self.SimPushes)
		self.sim_pick_place = rospy.Service('sim_pick_place',
												ExecTraj, self.SimPickPlace)
		self.remove_constraint = rospy.Service('remove_constraint',
										ResetSimulation, self.RemoveConstraint)

		self.ResetSimulation(-1)

	def ResetSimulation(self, req):
		for i, sim in enumerate(self.sims):
			sim_data = self.sim_datas[i]

			sim.resetSimulation()
			sim.setGravity(0,0,-9.81)
			sim_data['ground_plane_id'] = sim.loadURDF("plane.urdf")
			# sim.setRealTimeSimulation(0)
			sim.setTimeStep(1.0/HZ)

			if not self.fridge:
				sim_data['table_id'] = [1]
			else:
				sim_data['table_id'] = [1, 2, 3, 4, 5]
			sim_data['robot_id'] = -1
			sim_data['objs'] = {}
			sim_data['num_objs'] = 0

		return ResetSimulationResponse(True)

	def AddObject(self, req):
		sim_id = req.sim_id
		all_sims = sim_id < 0

		for i, sim in enumerate(self.sims) if all_sims else enumerate(self.sims[sim_id:sim_id + 1]):
			sim_idx = i if all_sims else sim_id
			sim_data = self.sim_datas[sim_idx]

			if (req.shape == 0):
				obj_id = self.addBoxNew(req, sim_idx)
			elif (req.shape == 2):
				obj_id = self.addCylinderNew(req, sim_idx)

			aabb = sim.getAABB(obj_id, -1)
			lower_limits = [x - CLEARANCE for x in aabb[0]]
			upper_limits = [x + CLEARANCE for x in aabb[1]]

			ground_plane_id = sim_data['ground_plane_id']
			table_id = sim_data['table_id']

			overlaps = sim.getOverlappingObjects(lower_limits, upper_limits)
			if (obj_id <= table_id[-1]): # part of table or fridge
				intersects = [oid != ground_plane_id and oid not in table_id and oid != obj_id for (oid, link) in overlaps]
			else:
				intersects = [oid != ground_plane_id and oid != table_id[0] and oid != obj_id for (oid, link) in overlaps]

			remove = False
			for oid, overlap in enumerate(overlaps):
				if not intersects[oid]:
					continue
				sim.performCollisionDetection()
				if sim.getContactPoints(obj_id, overlap[0]):
					remove = True
					break
			if remove: # only necessary overlaps: table and itself
				sim.removeBody(obj_id)
				del sim_data['objs'][obj_id]
				obj_id = -1

			if obj_id != -1:
				friction_min = rospy.get_param("~objects/friction_min", 0.5)
				friction_max = rospy.get_param("~objects/friction_max", 1.2)
				mu = np.random.uniform(friction_min, friction_max) if req.mu < friction_min else req.mu
				if obj_id in table_id:
					mu = friction_min

				sim_data['objs'][obj_id]['mu'] = mu
				sim_data['objs'][obj_id]['movable'] = req.movable
				sim_data['objs'][obj_id]['copies'] = []
				sim_data['num_objs'] += 1

				true_mass = sim_data['objs'][obj_id]['mass']
				true_mu = sim_data['objs'][obj_id]['mu']
				new_mass = np.minimum(2*true_mass, np.maximum(true_mass/2, np.random.normal(true_mass, scale=0.2*true_mass)))
				new_mu = np.minimum(2*true_mu, np.maximum(true_mu/2, np.random.normal(true_mu, scale=0.2*true_mu)))
				sim.changeDynamics(obj_id, -1, mass=new_mass, lateralFriction=new_mu)

				if req.movable:
					sim.setCollisionFilterGroupMask(obj_id, -1, 1, 1)
				else:
					sim.setCollisionFilterGroupMask(obj_id, -1, \
									self.robot_collision_group, self.robot_collision_group)

		return AddObjectResponse(obj_id, [])

	def AddYCBObject(self, req):
		sim_id = req.sim_id
		all_sims = sim_id < 0

		for i, sim in enumerate(self.sims) if all_sims else enumerate(self.sims[sim_id:sim_id + 1]):
			sim_idx = i if all_sims else sim_id
			sim = self.sims[sim_idx]
			sim_data = self.sim_datas[sim_idx]

			obj_id = req.obj_id
			xyz = [req.o_x, req.o_y, req.o_z]
			rpy = [req.o_r, req.o_p, req.o_yaw]

			filename = os.path.dirname(os.path.abspath(__file__)) + '/'
			filename += '../dat/ycb/{object}/urdf/{object}_tsdf.urdf'.format(object=YCB_OBJECTS[obj_id])

			body_id = sim.loadURDF(
							fileName=filename,
							basePosition=xyz,
							baseOrientation=sim.getQuaternionFromEuler(rpy))

			sim_data['objs'][body_id] = {
								'shape': obj_id,
								'xyz': xyz,
								'rpy': rpy,
								'ycb': True,
							}

			aabb = sim.getAABB(body_id, -1)

			lower_limits = [x - CLEARANCE for x in aabb[0]]
			upper_limits = [x + CLEARANCE for x in aabb[1]]

			ground_plane_id = sim_data['ground_plane_id']
			table_id = sim_data['table_id']

			overlaps = sim.getOverlappingObjects(lower_limits, upper_limits)
			if (body_id <= table_id[-1]):
				intersects = [oid != ground_plane_id and oid not in table_id and oid != body_id for (oid, link) in overlaps]
			else:
				intersects = [oid != ground_plane_id and oid != table_id[0] and oid != body_id for (oid, link) in overlaps]
			if any(intersects): # only necessary overlaps: table and itself
				sim.removeBody(body_id)
				del sim_data['objs'][body_id]
				body_id = -1

			if body_id != -1:
				friction_min = rospy.get_param("~objects/friction_min", 0.5)
				friction_max = rospy.get_param("~objects/friction_max", 1.2)
				mu = np.random.uniform(friction_min, friction_max) if req.mu < friction_min else req.mu

				sim.changeDynamics(body_id, -1, lateralFriction=mu)
				sim_data['objs'][body_id]['mu'] = mu
				sim_data['objs'][body_id]['movable'] = req.movable
				sim_data['num_objs'] += 1

		return AddYCBObjectResponse(body_id, [])


	def AddRobot(self, req):
		sim_id = req.sim_id
		all_sims = sim_id < 0

		for i, sim in enumerate(self.sims) if all_sims else enumerate(self.sims[sim_id:sim_id + 1]):
			sim_idx = i if all_sims else sim_id
			sim_data = self.sim_datas[sim_idx]

			xyz = [req.x, req.y, req.z]
			quat = [req.qx, req.qy, req.qz, req.qw]

			urdf_path = os.path.dirname(os.path.abspath(__file__)) + '/' + PR2_URDF
			robot_id = sim.loadURDF(urdf_path,
								basePosition=xyz,
								baseOrientation=quat,
								useFixedBase=1)

			sim_data['robot_id'] = robot_id
			sim_data['robot'] = {
							'xyz': xyz,
							'quat': quat,
						}
			sim_data['joint_idxs'] = joints_from_names(robot_id, PR2_GROUPS['right_arm'], sim=self.sims[sim_idx])

			for joint in get_joints(robot_id, sim=self.sims[sim_idx]):
				sim.setCollisionFilterGroupMask(robot_id, joint, \
								self.robot_collision_group, self.robot_collision_group)

		# print()
		robot_id = self.sim_datas[0]['robot_id']
		self.arm_joints = joints_from_names(robot_id, PR2_GROUPS['right_arm'], sim=sim)
		self.gripper_joints = joints_from_names(robot_id, PR2_GROUPS['right_gripper'], sim=sim)
		self.n_arm_joints = len(self.arm_joints)
		self.n_gripper_joints = len(self.gripper_joints)

		return AddRobotResponse(self.sim_datas[0]['robot_id'])

	def SetRobotStateAll(self, req):
		for i, sim in enumerate(self.sims):
			sim_id = i
			sim = self.sims[sim_id]
			sim_data = self.sim_datas[sim_id]
			robot_id = sim_data['robot_id']

			self.disableCollisionsWithObjects(sim_id)

			sim_data['start_joint_names'] = req.state.name

			joint_idxs = joints_from_names(robot_id, sim_data['start_joint_names'], sim=sim)
			for jidx, jval in zip(joint_idxs, req.state.position):
				sim.resetJointState(robot_id, jidx, jval, targetVelocity=0.0)
			for gjidx in self.gripper_joints:
				sim.resetJointState(robot_id, gjidx, 0.0, targetVelocity=0.0)

			sim.setJointMotorControlArray(
					robot_id, joint_idxs,
					controlMode=sim.VELOCITY_CONTROL,
					targetVelocities=[0.0] * len(joint_idxs))
			sim.setJointMotorControlArray(
				robot_id, self.gripper_joints,
				controlMode=sim.VELOCITY_CONTROL,
				targetVelocities=[0.0] * self.n_gripper_joints)
			sim.stepSimulation()

			self.enableCollisionsWithObjects(sim_id)

		return SetRobotStateResponse(True)

	def ResetArmAll(self, req):
		for i, sim in enumerate(self.sims):
			sim_id = i
			sim = self.sims[sim_id]
			robot_id = self.sim_datas[sim_id]['robot_id']

			self.disableCollisionsWithObjects(sim_id)

			if req.arm == 0:
				joint_idxs = joints_from_names(robot_id, PR2_GROUPS['left_arm'], sim=sim)
				joint_config = REST_LEFT_ARM
			else:
				joint_idxs = joints_from_names(robot_id, PR2_GROUPS['right_arm'], sim=sim)
				joint_config = WIDE_RIGHT_ARM

			for jidx, jval in zip(joint_idxs, joint_config):
				sim.resetJointState(robot_id, jidx, jval, targetVelocity=0.0)
			sim.setJointMotorControlArray(
					robot_id, joint_idxs,
					controlMode=sim.VELOCITY_CONTROL,
					targetVelocities=[0.0] * len(joint_idxs))
			sim.stepSimulation()

			self.enableCollisionsWithObjects(sim_id)

		return ResetArmResponse(True)

	def CheckScene(self, req):
		sim_id = req.sim_id
		all_sims = sim_id < 0
		intersects = []

		for i, sim in enumerate(self.sims) if all_sims else enumerate(self.sims[sim_id:sim_id + 1]):
			intersects = list(set(intersects))

			sim_idx = i if all_sims else sim_id
			sim = self.sims[sim_idx]
			sim_data = self.sim_datas[sim_idx]

			table_id = sim_data['table_id']
			robot_id = sim_data['robot_id']
			ground_plane_id = sim_data['ground_plane_id']

			if req.arm == 0:
				joint_idx = joint_from_name(robot_id, 'l_elbow_flex_joint', sim=sim)
			else:
				joint_idx = joint_from_name(robot_id, 'r_elbow_flex_joint', sim=sim)

			aabb = get_subtree_aabb(robot_id, root_link=joint_idx, sim=sim)
			lower_limits = [x - CLEARANCE for x in aabb[0]]
			upper_limits = [x + CLEARANCE for x in aabb[1]]

			overlaps = sim.getOverlappingObjects(lower_limits, upper_limits)
			for (obj_id, link) in overlaps:
				if (obj_id != ground_plane_id) and (obj_id not in table_id) and (obj_id != robot_id):
					# sim.removeBody(obj_id)
					# del sim_data['objs'][obj_id]
					# sim_data['num_objs'] -= 1
					intersects.append(obj_id)

		intersects = list(set(intersects))
		return CheckSceneResponse(intersects)

	def ResetScene(self, req):
		sim_id = req.sim_id
		all_sims = sim_id < 0
		for sim_id, sim in enumerate(self.sims) if all_sims else enumerate(self.sims[sim_id:sim_id + 1]):
			self.disableCollisionsWithObjects(sim_id, simulator=sim)
			self._reset_scene(sim_id, simulator=sim)
			self.enableCollisionsWithObjects(sim_id, simulator=sim)

		return ResetSceneResponse(True)

	def _reset_scene(self, sim_id, simulator=None):
		sim = self.sims[sim_id] if simulator is None else simulator
		sim_data = self.sim_datas[sim_id]

		if self.gui:
			table_id = sim_data['table_id']
			if (not self.camera_set):
				table_xyz = copy.deepcopy(sim_data['objs'][table_id[0]]['xyz'])
				table_xyz[2] += 0.4
				sim.resetDebugVisualizerCamera(
					cameraDistance=1.0, cameraYaw=-60.0, cameraPitch=-10.0,
					# cameraDistance=0.8, cameraYaw=-120.0, cameraPitch=0.0, # left side camera angle
					cameraTargetPosition=table_xyz)
				self.camera_set = True

		for obj_id in sim_data['objs']:
			xyz = sim_data['objs'][obj_id]['xyz']
			rpy = sim_data['objs'][obj_id]['rpy']
			sim.resetBasePositionAndOrientation(obj_id,
										posObj=xyz,
										ornObj=sim.getQuaternionFromEuler(rpy))

			if (sim_data['objs'][obj_id]['movable']):
				sim.setCollisionFilterGroupMask(obj_id, -1, 1, 1)

				if (not sim_data['objs'][obj_id]['copies'] and self.obj_copies > 1): # have not added copies yet
					if (not sim_data['objs'][obj_id]['ycb']):
						req = AddObjectRequest()

						req.shape = sim_data['objs'][obj_id]['shape']
						req.locked = False
						req.o_x = xyz[0]
						req.o_y = xyz[1]
						req.o_z = xyz[2]
						req.o_r = rpy[0]
						req.o_p = rpy[1]
						req.o_yaw = rpy[2]
						req.x_size = sim_data['objs'][obj_id]['extents'][0]
						req.y_size = sim_data['objs'][obj_id]['extents'][1]
						req.z_size = sim_data['objs'][obj_id]['extents'][2]

						# add object copies
						for copy_num in range(2, self.obj_copies + 1):
							true_mass = sim_data['objs'][obj_id]['mass']
							req.mass = np.minimum(2*true_mass, np.maximum(true_mass/2, np.random.normal(true_mass, scale=0.2*true_mass)))
							if (req.shape == 0):
								obj_copy_id = self.addBoxDuplicate(req, sim_id, obj_id)
							elif (req.shape == 2):
								obj_copy_id = self.addCylinderDuplicate(req, sim_id, obj_id)

							true_mu = sim_data['objs'][obj_id]['mu']
							new_mu = np.minimum(2*true_mu, np.maximum(true_mu/2, np.random.normal(true_mu, scale=0.2*true_mu)))
							sim.changeDynamics(obj_copy_id, -1, lateralFriction=new_mu)
							sim.setCollisionFilterGroupMask(obj_copy_id, -1, \
											2**(copy_num - 1), 2**(copy_num - 1))

				for copy_num, obj_copy_id in enumerate(sim_data['objs'][obj_id]['copies']):
					sim.resetBasePositionAndOrientation(obj_copy_id,
											posObj=xyz,
											ornObj=sim.getQuaternionFromEuler(rpy))
					sim.setCollisionFilterGroupMask(obj_copy_id, -1, 2**((copy_num+2) - 1), 2**((copy_num+2) - 1))

		# all collision groups are valid again
		sim_data['valid_groups'] = list(range(self.obj_copies))

	def SetColours(self, req):
		sim_id = req.sim_id
		all_sims = sim_id < 0

		for i, sim in enumerate(self.sims) if all_sims else enumerate(self.sims[sim_id:sim_id + 1]):
			sim_idx = i if all_sims else sim_id
			sim = self.sims[sim_idx]
			sim_data = self.sim_datas[sim_idx]

			for i in range(120):
				sim.stepSimulation()

			for idx, obj_id in enumerate(req.ids):
				if req.type[idx] == -1: # table
					sim.changeVisualShape(obj_id, -1,
									rgbaColor=TABLE_COLOUR + [1.0],
									specularColor=TABLE_COLOUR)
					sim_data['objs'][obj_id]['type'] = -1

				if req.type[idx] == 0: # static
					sim.changeVisualShape(obj_id, -1,
									rgbaColor=STATIC_COLOUR + [1.0],
									specularColor=STATIC_COLOUR)
					sim_data['objs'][obj_id]['type'] = 0

				if req.type[idx] == 1: # moveable
					sim.changeVisualShape(obj_id, -1,
									rgbaColor=MOVEABLE_COLOUR + [1.0],
									specularColor=MOVEABLE_COLOUR)
					sim_data['objs'][obj_id]['type'] = 1

					if (sim_data['objs'][obj_id]['copies']):
						alpha_factor = 0.5/len(sim_data['objs'][obj_id]['copies'])
						for copy_num, obj_copy_id in enumerate(sim_data['objs'][obj_id]['copies']):
							alpha = 1.0 - (alpha_factor * (copy_num + 1))
							sim.changeVisualShape(obj_copy_id, -1,
											rgbaColor=MOVEABLE_COLOUR + [alpha],
											specularColor=MOVEABLE_COLOUR)

				if req.type[idx] == 999: # ooi
					sim.changeVisualShape(obj_id, -1,
									rgbaColor=OOI_COLOUR + [1.0],
									specularColor=OOI_COLOUR)
					sim_data['objs'][obj_id]['type'] = 999

				obj_xyz, obj_rpy = sim.getBasePositionAndOrientation(obj_id)
				sim_data['objs'][obj_id]['xyz'][2] = obj_xyz[2]

		return SetColoursResponse(True)

	def ExecTraj(self, req):
		params = (req,)
		response = None
		if self.gui:
			retval = dict()
			response = self.exec_traj_job(0, self.sims[0], retval, params)
			if (len(response[0]) < self.succ_needed):
				response[1].violation = True
		else:
			retvals = self.run_jobs_in_threads(self.exec_traj_job, params)

			# count successful simulations
			succ_count = 0
			for sim_id, sim_result in retvals:
				if not sim_result[1].violation:
					succ_count += len(sim_result[0])

			success = succ_count >= self.succ_needed
			# pick one simulation result to return, reset all
			for sim_id, sim_result in retvals:
				if response is None:
					if success and not sim_result[1].violation:
						response = sim_result
						break
					elif not success and sim_result[1].violation:
						response = sim_result
						break

		if (response is None and not success):
			response = retvals[0][1]
			response[1].violation = True
			response[1].interactions = []
			response[1].objects = req.objects

		return response[1]

	def exec_traj_job(self, sim_id, sim, retvals, params):
		sim_data = self.sim_datas[sim_id]
		robot_id = sim_data['robot_id']
		req = params[0]

		curr_timestep = 0
		curr_pose = np.asarray(req.traj.points[0].positions)

		# arm_joints = joints_from_names(robot_id, PR2_GROUPS['right_arm'], sim=sim)
		arm_joints = joints_from_names(robot_id, req.traj.joint_names, sim=sim)
		# ee_link = link_from_name(sim_data['robot_id'], 'r_gripper_finger_dummy_planning_link')

		self.disableCollisionsWithObjects(sim_id, simulator=sim)

		# reset back to true mass and friction
		for obj_id in sim_data['objs']:
			sim.changeDynamics(obj_id, -1, mass=sim_data['objs'][obj_id]['mass'], lateralFriction=sim_data['objs'][obj_id]['mu'])

		if (len(req.objects.poses) != 0):
			self.resetObjects(sim_id, req.objects.poses, simulator=sim)

		for jidx, jval in zip(arm_joints, curr_pose):
			sim.resetJointState(robot_id, jidx, jval, targetVelocity=0.0)
		for gjidx in self.gripper_joints:
			sim.resetJointState(robot_id, gjidx, 0.2, targetVelocity=0.0)
		sim.setJointMotorControlArray(
					robot_id, arm_joints,
					controlMode=sim.VELOCITY_CONTROL,
					targetVelocities=[0.0] * len(arm_joints))
		sim.setJointMotorControlArray(
			robot_id, self.gripper_joints,
			controlMode=sim.VELOCITY_CONTROL,
			targetVelocities=[0.0] * self.n_gripper_joints)
		sim.stepSimulation()
		# arm_vels = get_joint_velocities(robot_id, arm_joints, sim=sim)
		# print("\n\t [ExecTraj] arm_vels = ", arm_vels)

		self.enableCollisionsWithObjects(sim_id, simulator=sim)

		pick_at = req.pick_at
		place_at = req.place_at
		violation_flag = False
		all_interactions = []
		for pidx, point in enumerate(req.traj.points[1:]):
			if (pick_at >= 0):
				# this is either the OOI retrieval trajectory
				# or a pick-place rearrangement trajectory
				if (point == req.traj.points[pick_at]):
					self.grasp(sim_id, True, simulator=sim) # open gripper

				elif (place_at >= 0 and point == req.traj.points[place_at]):
					self.grasp(sim_id, True, simulator=sim) # open gripper
					for constraint in sim_data['grasp_constraint']:
						sim.removeConstraint(constraint[3])
						sim.changeDynamics(constraint[0], -1, mass=constraint[4])
					del sim_data['grasp_constraint'][:]
					# for i in range(2 * int(HZ)):
					# 	sim.stepSimulation()

				elif (point == req.traj.points[pick_at+1]):
					self.grasp(sim_id, False, oid=req.ooi, simulator=sim) # close gripper

					if not sim_data['grasp_constraint']:
						if (0 in sim_data['valid_groups']):
							self.addGraspConstraint(req.ooi, 0, sim_id, simulator=sim)
						for copy_num, obj_copy_id in enumerate(sim_data['objs'][req.ooi]['copies']):
							if (copy_num+1 in sim_data['valid_groups']):
								self.addGraspConstraint(obj_copy_id, copy_num+1, sim_id, simulator=sim)

				# elif (point == req.traj.points[pick_at+2]):
				# 	input("OOI lifted!")
				# 	# For scene 100020
				# 	movable_id = 15
				# 	movable_traj = np.array([[0.619541,-0.36661],[0.62,-0.38],[0.62,-0.39],[0.62,-0.4],[0.63,-0.4],[0.64,-0.4],[0.65,-0.4],[0.66,-0.4],[0.67,-0.4],[0.68,-0.4],[0.69,-0.4],[0.7,-0.4],[0.71,-0.4],[0.72,-0.4],[0.73,-0.4]])

				# 	for joint in get_joints(robot_id, sim=sim):
				# 		sim.changeVisualShape(robot_id, joint, rgbaColor=[0.66, 0.66, 0.66, 0.66])

				# 	input("Start sim-ing MAPF solution?")
				# 	movable_pos = None
				# 	movable_orn = None
				# 	for oid in sim_data['objs']:
				# 		if (oid == movable_id):
				# 			movable_pos, movable_orn = sim.getBasePositionAndOrientation(oid)
				# 			movable_pos = np.asarray(movable_pos)
				# 			movable_orn = np.asarray(movable_orn)
				# 			break
				# 	for pt in movable_traj:
				# 		movable_pos[0] = pt[0]
				# 		movable_pos[1] = pt[1]
				# 		sim.resetBasePositionAndOrientation(movable_id,
				# 						posObj=movable_pos,
				# 						ornObj=movable_orn)
				# 		time.sleep(0.33)
				# 	input("DONE")

				# 	if not self.grasped(req.ooi, sim_id):
				# 		violation_flag = True
				# 		break

				else:
					if not sim_data['grasp_constraint']:
						sim.setJointMotorControlArray(
								robot_id, self.gripper_joints,
								controlMode=sim.POSITION_CONTROL,
								targetPositions=[0.0]*self.n_gripper_joints)
			else:
				# keep gripper slightly open for pushing trajectories
				sim.setJointMotorControlArray(
						robot_id, self.gripper_joints,
						controlMode=sim.POSITION_CONTROL,
						targetPositions=[0.2]*self.n_gripper_joints)

			prev_timestep = curr_timestep
			prev_pose = get_joint_positions(robot_id, arm_joints, sim=sim)
			curr_timestep = point.time_from_start.to_sec()
			curr_pose = np.asarray(point.positions)
			time_diff = (curr_timestep - prev_timestep) * 1
			target_vel = shortest_angle_diff(curr_pose, prev_pose)/time_diff
			duration = math.ceil(time_diff * int(HZ))

			sim.setJointMotorControlArray(
					robot_id, arm_joints,
					controlMode=sim.VELOCITY_CONTROL,
					targetVelocities=target_vel)

			objs_curr = self.getObjects(sim_id, simulator=sim)
			action_interactions = []
			for i in range(int(duration)):
				sim.stepSimulation()

				interactions = self.checkInteractions(sim_id, objs_curr, simulator=sim)
				action_interactions += interactions
				action_interactions = list(np.unique(np.array(action_interactions)))
				action_interactions[:] = [idx for idx in action_interactions if idx != req.ooi]
				violation_flag = any([not sim_data['objs'][x]['movable'] for x in action_interactions])

				if not violation_flag:
					violation_flag = self.checkTableCollision(sim_id, simulator=sim)
				if not violation_flag:
					violation_flag = self.checkPoseConstraints(sim_id, grasp_at=pick_at, ooi=req.ooi, simulator=sim)
				if not violation_flag:
					violation_flag = self.checkVelConstraints(sim_id, grasp_at=pick_at, ooi=req.ooi, simulator=sim)

				if (violation_flag):
					break

			all_interactions += action_interactions
			all_interactions = list(np.unique(np.array(all_interactions)))
			del action_interactions[:]

			if (violation_flag):
				break # trajectory execution failed

		# To simulate the scene after execution of the trajectory
		if (not violation_flag):
			sim.setJointMotorControlArray(
					robot_id, arm_joints,
					controlMode=sim.VELOCITY_CONTROL,
					targetVelocities=len(arm_joints)*[0.0])
			sim.setJointMotorControlArray(
					robot_id, self.gripper_joints,
					controlMode=sim.POSITION_CONTROL,
					targetPositions=[0.2]*self.n_gripper_joints)
			self.holdPosition(sim_id, simulator=sim)

			objs_curr = self.getObjects(sim_id, simulator=sim)
			action_interactions = []
			for i in range(2 * int(HZ)):
				sim.stepSimulation()

				interactions = self.checkInteractions(sim_id, objs_curr, simulator=sim)
				action_interactions += interactions
				action_interactions = list(np.unique(np.array(action_interactions)))
				action_interactions[:] = [idx for idx in action_interactions if idx != req.ooi]
				violation_flag = any([not sim_data['objs'][x]['movable'] for x in action_interactions])

				if not violation_flag:
					violation_flag = self.checkTableCollision(sim_id, simulator=sim)
				if not violation_flag:
					violation_flag = self.checkPoseConstraints(sim_id, grasp_at=pick_at, ooi=req.ooi, simulator=sim)
				if not violation_flag:
					violation_flag = self.checkVelConstraints(sim_id, grasp_at=pick_at, ooi=req.ooi, simulator=sim)

				if (violation_flag):
					break

			all_interactions += action_interactions
			all_interactions = list(np.unique(np.array(all_interactions)))
			del action_interactions[:]

		# ee_dummy_state = get_link_state(sim_data['robot_id'], ee_link).linkWorldPosition
		all_interactions = list(np.unique(np.array(all_interactions).astype(np.int)))

		output = ExecTrajResponse()
		output.violation = violation_flag
		output.interactions = all_interactions
		output.objects = ObjectsPoses()
		output.objects.poses = self.getValidObjects(sim_id, simulator=sim)

		retvals[sim_id] = (sim_data['valid_groups'], output)
		if self.gui:
			return (sim_data['valid_groups'], output)

	def SimPushes(self, req):
		params = (req,)
		response = None
		if self.gui:
			retval = dict()
			response = self.sim_pushes_job(0, self.sims[0], retval, params)
			if (len(response[0]) < self.succ_needed):
				response[1].res = False
				response[1].idx = -1
				response[1].successes = len(response[0])
		else:
			start = time.time()
			retvals = self.run_jobs_in_threads(self.sim_pushes_job, params)

			# count successful simulations
			succ_count = 0
			print()
			for sim_id, sim_result in retvals:
				print(bcolors.BLUE + "Sim ID {} result: successes, push_idx, valid_groups = {}, {}, [{}]".format(sim_id, sim_result[1].successes, sim_result[1].idx, ', '.join(map(str, sim_result[0]))) + bcolors.ENDC)
				if sim_result[1].idx >= 0:
					succ_count += len(sim_result[0])

			success = succ_count >= self.succ_needed
			# pick one simulation result to return, reset all
			for sim_id, sim_result in retvals:
				if response is None:
					if success and sim_result[1].idx >= 0:
						print(bcolors.GREEN + '\tSelect Sim ID {} SUCCESS result!'.format(sim_id) + bcolors.ENDC)
						response = sim_result
						break
					elif not success and sim_result[1].idx < 0:
						print(bcolors.GREEN + '\tSelect Sim ID {} FAIL result!'.format(sim_id) + bcolors.ENDC)
						response = sim_result
						break

			elapsed = time.time() - start
			print('SimPushes took {:.3f} seconds'.format(elapsed))

		if (response is None and not success):
			response = retvals[0][1]
			response[1].res = False
			response[1].idx = -1
			response[1].successes = 0
			response[1].objects = req.objects

		return response[1]

	def sim_pushes_job(self, sim_id, sim, retvals, params):
		print()
		sim_data = self.sim_datas[sim_id]
		robot_id = sim_data['robot_id']
		table_id = sim_data['table_id']
		req = params[0]

		num_pushes = len(req.pushes)
		successes = 0
		best_idx = -1
		best_dist = float('inf')
		best_objs = self.getValidObjects(sim_id, simulator=sim)
		start_objs = None
		goal_pos = None
		if (req.oid != -1):
			goal_pos = np.asarray([req.gx, req.gy])

		for pidx in range(num_pushes):
			push_traj = req.pushes[pidx]

			curr_timestep = push_traj.points[0].time_from_start.to_sec()
			curr_pose = np.asarray(push_traj.points[0].positions)

			self.disableCollisionsWithObjects(sim_id, simulator=sim)

			self._reset_scene(sim_id, simulator=sim)
			if (len(req.objects.poses) != 0):
				self.resetObjects(sim_id, req.objects.poses, simulator=sim)

			for jidx, jval in zip(self.arm_joints, curr_pose):
				sim.resetJointState(robot_id, jidx, jval, targetVelocity=0.0)
			for gjidx in self.gripper_joints:
				sim.resetJointState(robot_id, gjidx, 0.2, targetVelocity=0.0)
			sim.setJointMotorControlArray(
					robot_id, self.arm_joints,
					controlMode=sim.VELOCITY_CONTROL,
					targetVelocities=[0.0] * self.n_arm_joints)
			sim.setJointMotorControlArray(
				robot_id, self.gripper_joints,
				controlMode=sim.VELOCITY_CONTROL,
				targetVelocities=[0.0] * self.n_gripper_joints)
			sim.stepSimulation()
			# arm_vels = get_joint_velocities(robot_id, arm_joints, sim=sim)
			# print("\n\t [SimPushes] arm_vels = ", arm_vels)

			self.enableCollisionsWithObjects(sim_id, simulator=sim)

			start_objs = self.getValidObjects(sim_id, simulator=sim)
			violation_flag = False
			oid_start_xyz = None
			if (req.oid != -1):
				if (0 in sim_data['valid_groups']):
					oid_start_xyz, _ = sim.getBasePositionAndOrientation(req.oid)
				else:
					for copy_num, obj_copy_id in enumerate(sim_data['objs'][req.oid]['copies']):
						if (copy_num+1 in sim_data['valid_groups']):
							oid_start_xyz, _ = sim.getBasePositionAndOrientation(obj_copy_id)
							break

			for point in push_traj.points[1:]:
				sim.setJointMotorControlArray(
						robot_id, self.gripper_joints,
						controlMode=sim.POSITION_CONTROL,
						targetPositions=[0.2]*self.n_gripper_joints)

				prev_timestep = curr_timestep
				prev_pose = get_joint_positions(robot_id, self.arm_joints, sim=sim)
				curr_timestep = point.time_from_start.to_sec()
				curr_pose = np.asarray(point.positions)
				time_diff = (curr_timestep - prev_timestep) * 1
				target_vel = shortest_angle_diff(curr_pose, prev_pose)/time_diff
				duration = math.ceil(time_diff * int(HZ))

				sim.setJointMotorControlArray(
						robot_id, self.arm_joints,
						controlMode=sim.VELOCITY_CONTROL,
						targetVelocities=target_vel)

				action_interactions = []
				objs_curr = self.getObjects(sim_id, simulator=sim)
				for i in range(int(duration)):
					sim.stepSimulation()

					interactions = self.checkInteractions(sim_id, objs_curr, simulator=sim)
					action_interactions += interactions
					action_interactions = list(np.unique(np.array(action_interactions)))
					violation_flag = any([not sim_data['objs'][x]['movable'] for x in action_interactions])

					if not violation_flag:
						violation_flag = self.checkTableCollision(sim_id, simulator=sim)
					else:
						print(bcolors.PINK + "\tPush hit immovable obstacle!" + bcolors.ENDC)
						break

					if not violation_flag:
						violation_flag = self.checkPoseConstraints(sim_id, simulator=sim)
					else:
						print(bcolors.PINK + "\tPush hit table!" + bcolors.ENDC)
						break

					if not violation_flag:
						violation_flag = self.checkVelConstraints(sim_id, simulator=sim)
					else:
						print(bcolors.PINK + "\tPush violated movable object pose constraints!" + bcolors.ENDC)
						break

					if (violation_flag):
						print(bcolors.PINK + "\tPush violated movable object velocity constraints!" + bcolors.ENDC)
						break

				if (violation_flag):
					break # stop simming this push

			if (violation_flag):
				continue # to next push

			# To simulate the scene after execution of the trajectory
			sim.setJointMotorControlArray(
					robot_id, self.arm_joints,
					controlMode=sim.VELOCITY_CONTROL,
					targetVelocities=[0.0]*self.n_arm_joints)
			self.holdPosition(sim_id, simulator=sim)

			objs_curr = self.getObjects(sim_id, simulator=sim)
			for obj_id in sim_data['objs']:
				if obj_id in table_id or not sim_data['objs'][obj_id]['movable']:
					continue

				closest_pts = sim.getClosestPoints(obj_id, robot_id, PUSH_END_THRESH)
				if len(closest_pts) > 0:
					violation_flag = True
					break

			if (violation_flag):
				continue # to next push

			action_interactions = []
			for i in range(2 * int(HZ)):
				sim.stepSimulation()

				interactions = self.checkInteractions(sim_id, objs_curr, simulator=sim)
				action_interactions += interactions
				action_interactions = list(np.unique(np.array(action_interactions)))
				violation_flag = any([not sim_data['objs'][x]['movable'] for x in action_interactions])

				if not violation_flag:
					violation_flag = self.checkTableCollision(sim_id, simulator=sim)
				else:
					print(bcolors.PINK + "\tPush hit immovable obstacle (during hold)???" + bcolors.ENDC)
					break

				if not violation_flag:
					violation_flag = self.checkPoseConstraints(sim_id, simulator=sim)
				else:
					print(bcolors.PINK + "\tPush hit table (during hold)???" + bcolors.ENDC)
					break

				if not violation_flag:
					violation_flag = self.checkVelConstraints(sim_id, simulator=sim)
				else:
					print(bcolors.PINK + "\tPush violated movable object pose constraints (during hold)!" + bcolors.ENDC)
					break

				if (violation_flag):
					print(bcolors.PINK + "\tPush violated movable object velocity constraints (during hold)!" + bcolors.ENDC)
					break

			if (violation_flag):
				continue # to next push
			else:
				oid_xyz = None
				if (req.oid != -1):
					if (0 in sim_data['valid_groups']):
						oid_xyz, _ = sim.getBasePositionAndOrientation(req.oid)
					else:
						for copy_num, obj_copy_id in enumerate(sim_data['objs'][req.oid]['copies']):
							if (copy_num+1 in sim_data['valid_groups']):
								oid_xyz, _ = sim.getBasePositionAndOrientation(obj_copy_id)
								break

					if (np.linalg.norm(np.asarray(oid_start_xyz) - np.asarray(oid_xyz)) <= 0.01):
						continue

				successes += len(sim_data['valid_groups'])
				if (req.oid != -1):
					dist = np.linalg.norm(goal_pos - np.asarray(oid_xyz[:2]))
					if (dist < best_dist):
						best_dist = dist
						best_idx = pidx
						best_objs = self.getValidObjects(sim_id, simulator=sim)
				else:
					best_idx = 0
					best_objs = self.getValidObjects(sim_id, simulator=sim)

		res = best_idx != -1
		return_objs = []
		for object in best_objs:
			if sim_data['objs'][object.id]['movable']:
				return_objs.append(object)

		relevant_ids = []
		if (res):
			for i, object in enumerate(best_objs):
				if sim_data['objs'][object.id]['movable']:
					if (np.linalg.norm(np.asarray(start_objs[i].xyz) - np.asarray(object.xyz)) > 0.01):
						relevant_ids.append(object.id)

		print("\tsim_pushes_job result (sim_id, successes, push_idx, valid_groups): ", sim_id, successes, best_idx, sim_data['valid_groups'])

		output = SimPushesResponse()
		output.res = res
		output.idx = best_idx
		output.successes = successes
		output.objects = ObjectsPoses()
		output.objects.poses = return_objs
		output.relevant_ids = relevant_ids
		retvals[sim_id] = (sim_data['valid_groups'], output)
		if self.gui:
			return (sim_data['valid_groups'], output)

	def SimPickPlace(self, req):
		params = (req,)
		response = None
		if self.gui:
			retval = dict()
			response = self.sim_pick_place_job(0, self.sims[0], retval, params)
			if (len(response[0]) < self.succ_needed):
				response[1].violation = True
		else:
			retvals = self.run_jobs_in_threads(self.sim_pick_place_job, params)

			# count successful simulations
			succ_count = 0
			for sim_id, sim_result in retvals:
				if not sim_result[1].violation:
					succ_count += len(sim_result[0])

			success = succ_count >= self.succ_needed
			# pick one simulation result to return, reset all
			for sim_id, sim_result in retvals:
				if response is None:
					if success and not sim_result[1].violation:
						response = sim_result
					elif not success and sim_result[1].violation:
						response = sim_result

		return response[1]

	def sim_pick_place_job(self, sim_id, sim, retvals, params):
		print()
		sim_data = self.sim_datas[sim_id]
		robot_id = sim_data['robot_id']
		req = params[0]

		# arm_joints = joints_from_names(robot_id, PR2_GROUPS['right_arm'], sim=sim)
		arm_joints = joints_from_names(robot_id, req.traj.joint_names, sim=sim)
		# ee_link = link_from_name(sim_data['robot_id'], 'r_gripper_finger_dummy_planning_link')

		self.disableCollisionsWithObjects(sim_id, simulator=sim)

		self._reset_scene(sim_id, simulator=sim)
		if (len(req.objects.poses) != 0):
			self.resetObjects(sim_id, req.objects.poses, simulator=sim)

		picked_obj = req.ooi
		pick_at = req.pick_at
		place_at = req.place_at
		curr_timestep = req.traj.points[pick_at-1].time_from_start.to_sec()
		curr_pose = np.asarray(req.traj.points[pick_at-1].positions)

		for jidx, jval in zip(arm_joints, curr_pose):
			sim.resetJointState(robot_id, jidx, jval, targetVelocity=0.0)
		for gjidx in self.gripper_joints:
			sim.resetJointState(robot_id, gjidx, 0.0, targetVelocity=0.0)
		sim.setJointMotorControlArray(
				robot_id, arm_joints,
				controlMode=sim.VELOCITY_CONTROL,
				targetVelocities=[0.0] * len(arm_joints))
		sim.setJointMotorControlArray(
				robot_id, self.gripper_joints,
				controlMode=sim.VELOCITY_CONTROL,
				targetVelocities=[0.0] * self.n_gripper_joints)
		sim.stepSimulation()
		# arm_vels = get_joint_velocities(robot_id, arm_joints, sim=sim)
		# print("\n\t [ExecTraj] arm_vels = ", arm_vels)

		self.enableCollisionsWithObjects(sim_id, simulator=sim)

		violation_flag = False
		oid_start_xyz = None
		if (picked_obj != -1):
			if (0 in sim_data['valid_groups']):
				oid_start_xyz, _ = sim.getBasePositionAndOrientation(picked_obj)
			else:
				for copy_num, obj_copy_id in enumerate(sim_data['objs'][picked_obj]['copies']):
					if (copy_num+1 in sim_data['valid_groups']):
						oid_start_xyz, _ = sim.getBasePositionAndOrientation(obj_copy_id)
						break

		# To simulate the pickup action
		for point in req.traj.points[pick_at:pick_at+2]:
			if (point == req.traj.points[pick_at]):
				self.grasp(sim_id, True, simulator=sim) # open gripper

			elif (point == req.traj.points[pick_at+1]):
				self.grasp(sim_id, False, oid=picked_obj, simulator=sim) # close gripper

				if not sim_data['grasp_constraint']:
					if (0 in sim_data['valid_groups']):
						self.addGraspConstraint(picked_obj, 0, sim_id, simulator=sim)
					for copy_num, obj_copy_id in enumerate(sim_data['objs'][req.ooi]['copies']):
						if (copy_num+1 in sim_data['valid_groups']):
							self.addGraspConstraint(obj_copy_id, copy_num+1, sim_id, simulator=sim)

			# else:
			# 	# keep gripper closed while moving to pick pose
			# 	if not sim_data['grasp_constraint']:
			# 		sim.setJointMotorControlArray(
			# 				robot_id, self.gripper_joints,
			# 				controlMode=sim.POSITION_CONTROL,
			# 				targetPositions=[0.0]*self.n_gripper_joints)

			prev_timestep = curr_timestep
			prev_pose = get_joint_positions(robot_id, arm_joints, sim=sim)
			curr_timestep = point.time_from_start.to_sec()
			curr_pose = np.asarray(point.positions)
			time_diff = (curr_timestep - prev_timestep) * 1
			target_vel = shortest_angle_diff(curr_pose, prev_pose)/time_diff
			duration = math.ceil(time_diff * int(HZ))

			sim.setJointMotorControlArray(
					robot_id, arm_joints,
					controlMode=sim.VELOCITY_CONTROL,
					targetVelocities=target_vel)

			action_interactions = []
			objs_curr = self.getObjects(sim_id, simulator=sim)
			for i in range(int(duration)):
				sim.stepSimulation()

				interactions = self.checkInteractions(sim_id, objs_curr, simulator=sim)
				action_interactions += interactions
				action_interactions = list(np.unique(np.array(action_interactions)))
				action_interactions[:] = [idx for idx in action_interactions if idx != picked_obj]
				violation_flag = any([not sim_data['objs'][x]['movable'] for x in action_interactions])

				if not violation_flag:
					violation_flag = self.checkTableCollision(sim_id, simulator=sim)
				else:
					print(bcolors.PINK + "\tPick-Place hit immovable obstacle (during pick)!" + bcolors.ENDC)
					break

				if not violation_flag:
					violation_flag = self.checkPoseConstraints(sim_id, grasp_at=pick_at, ooi=picked_obj, simulator=sim)
				else:
					print(bcolors.PINK + "\tPick-Place hit table (during pick)!" + bcolors.ENDC)
					break

				if not violation_flag:
					violation_flag = self.checkVelConstraints(sim_id, grasp_at=pick_at, ooi=picked_obj, simulator=sim)
				else:
					print(bcolors.PINK + "\tPick-Place violated movable object pose constraints (during pick)!" + bcolors.ENDC)
					break

				if (violation_flag):
					print(bcolors.PINK + "\tPick-Place violated movable object velocity constraints (during pick)!" + bcolors.ENDC)
					break

			del action_interactions[:]
			if (violation_flag):
				break # trajectory execution failed

		# To simulate the placement action
		if (not violation_flag):
			self.disableCollisionsWithObjects(sim_id, ignore=[picked_obj], simulator=sim)
			for obj_id in sim_data['objs']:
				if (obj_id == picked_obj):
					continue
				for constraint in sim_data['grasp_constraint']:
					if (0 in sim_data['valid_groups']):
						sim.setCollisionFilterPair(constraint[0], obj_id, -1, -1,
											enableCollision=0)
					for copy_num, obj_copy_id in enumerate(sim_data['objs'][obj_id]['copies']):
						if (copy_num+1 in sim_data['valid_groups']):
							sim.setCollisionFilterPair(constraint[0], obj_copy_id, -1, -1,
											enableCollision=0)

			prev_timestep = curr_timestep
			prev_pose = get_joint_positions(robot_id, arm_joints, sim=sim)
			curr_timestep = req.traj.points[place_at-2].time_from_start.to_sec()
			curr_pose = np.asarray(req.traj.points[place_at-2].positions)
			time_diff = (curr_timestep - prev_timestep) * 1
			target_vel = shortest_angle_diff(curr_pose, prev_pose)/time_diff
			duration = math.ceil(time_diff * int(HZ))

			sim.setJointMotorControlArray(
					robot_id, arm_joints,
					controlMode=sim.VELOCITY_CONTROL,
					targetVelocities=target_vel)

			for i in range(int(duration)):
					sim.stepSimulation()
			# arm_vels = get_joint_velocities(robot_id, arm_joints, sim=sim)
			# print("\n\t [ExecTraj] arm_vels = ", arm_vels)

			self.enableCollisionsWithObjects(sim_id, ignore=[picked_obj], simulator=sim)
			for obj_id in sim_data['objs']:
				if (obj_id == picked_obj):
					continue
				for constraint in sim_data['grasp_constraint']:
					if (0 in sim_data['valid_groups']):
						sim.setCollisionFilterPair(constraint[0], obj_id, -1, -1,
											enableCollision=1)
					for copy_num, obj_copy_id in enumerate(sim_data['objs'][obj_id]['copies']):
						if (copy_num+1 in sim_data['valid_groups']):
							sim.setCollisionFilterPair(constraint[0], obj_copy_id, -1, -1,
											enableCollision=1)

			for point in req.traj.points[place_at-2:]:
				if (point == req.traj.points[place_at]):
					self.grasp(sim_id, True, simulator=sim) # open gripper
					for constraint in sim_data['grasp_constraint']:
						sim.removeConstraint(constraint[3])
						sim.changeDynamics(constraint[0], -1, mass=constraint[4])
					del sim_data['grasp_constraint'][:]
					# for i in range(2 * int(HZ)):
					# 	sim.stepSimulation()

				prev_timestep = curr_timestep
				prev_pose = get_joint_positions(robot_id, arm_joints, sim=sim)
				curr_timestep = point.time_from_start.to_sec()
				curr_pose = np.asarray(point.positions)
				time_diff = (curr_timestep - prev_timestep) * 1
				target_vel = shortest_angle_diff(curr_pose, prev_pose)/time_diff
				duration = math.ceil(time_diff * int(HZ))

				sim.setJointMotorControlArray(
						robot_id, arm_joints,
						controlMode=sim.VELOCITY_CONTROL,
						targetVelocities=target_vel)

				action_interactions = []
				objs_curr = self.getObjects(sim_id, simulator=sim)
				for i in range(int(duration)):
					sim.stepSimulation()

					interactions = self.checkInteractions(sim_id, objs_curr, simulator=sim)
					action_interactions += interactions
					action_interactions = list(np.unique(np.array(action_interactions)))
					action_interactions[:] = [idx for idx in action_interactions if idx != picked_obj]

					topple = self.checkPoseConstraints(sim_id, grasp_at=pick_at, ooi=picked_obj, simulator=sim)
					immovable = any([not sim_data['objs'][x]['movable'] for x in action_interactions])
					table = self.checkTableCollision(sim_id, simulator=sim)
					velocity = self.checkVelConstraints(sim_id, grasp_at=pick_at, ooi=picked_obj, simulator=sim)

					violation_flag = topple or immovable or table or velocity
					if (violation_flag):
						break

				del action_interactions[:]
				if (violation_flag):
					break # trajectory execution failed

		elif (sim_data['grasp_constraint'] is not None):
			for constraint in sim_data['grasp_constraint']:
				sim.removeConstraint(constraint[3])
				sim.changeDynamics(constraint[0], -1, mass=constraint[4])
			del sim_data['grasp_constraint'][:]

		# To simulate the scene after execution of the trajectory
		if (not violation_flag):
			sim.setJointMotorControlArray(
					robot_id, arm_joints,
					controlMode=sim.VELOCITY_CONTROL,
					targetVelocities=len(arm_joints)*[0.0])
			# open gripper a little for future pushing
			sim.setJointMotorControlArray(
					robot_id, self.gripper_joints,
					controlMode=sim.POSITION_CONTROL,
					targetPositions=[0.2]*self.n_gripper_joints)
			self.holdPosition(sim_id, simulator=sim)

			objs_curr = self.getObjects(sim_id, simulator=sim)
			action_interactions = []
			for i in range(int(HZ)):
				sim.stepSimulation()

				interactions = self.checkInteractions(sim_id, objs_curr, simulator=sim)
				action_interactions += interactions
				action_interactions = list(np.unique(np.array(action_interactions)))
				action_interactions[:] = [idx for idx in action_interactions if idx != picked_obj]
				violation_flag = any([not sim_data['objs'][x]['movable'] for x in action_interactions])

				if not violation_flag:
					violation_flag = self.checkTableCollision(sim_id, simulator=sim)
				if not violation_flag:
					violation_flag = self.checkPoseConstraints(sim_id, simulator=sim)
				if not violation_flag:
					violation_flag = self.checkVelConstraints(sim_id, simulator=sim)

				if (violation_flag):
					break

			del action_interactions[:]

		output = ExecTrajResponse()
		output.violation = violation_flag

		all_objs = self.getValidObjects(sim_id, simulator=sim)
		return_objs = []
		for object in all_objs:
			if sim_data['objs'][object.id]['movable']:
				return_objs.append(object)
		output.objects = ObjectsPoses()
		output.objects.poses = return_objs

		retvals[sim_id] = (sim_data['valid_groups'], output)
		if self.gui:
			return (sim_data['valid_groups'], output)

	def RemoveConstraint(self, req):
		for sim_id, sim in enumerate(self.sims):
			sim_data = self.sim_datas[sim_id]
			if sim_data['grasp_constraint']:
				for constraint in sim_data['grasp_constraint']:
					sim.removeConstraint(constraint[3])
					sim.changeDynamics(constraint[0], -1, mass=constraint[4])
				del sim_data['grasp_constraint'][:]

				self.disableCollisionsWithObjects(sim_id, simulator=sim)
				self._reset_scene(sim_id, simulator=sim)
				self.enableCollisionsWithObjects(sim_id, simulator=sim)

		return ResetSimulationResponse(True)

	def disableCollisionsWithObjects(self, sim_id, ignore=None, simulator=None):
		sim = self.sims[sim_id] if simulator is None else simulator
		sim_data = self.sim_datas[sim_id]

		robot_id = sim_data['robot_id']
		for joint in get_joints(robot_id, sim=sim):
			for obj_id in sim_data['objs']:
				if (not sim_data['objs'][obj_id]['movable']):
					sim.setCollisionFilterPair(robot_id, obj_id, joint, -1,
											enableCollision=0)
				else:
					if (ignore is not None and obj_id in ignore):
						continue

					if (0 in sim_data['valid_groups']):
						sim.setCollisionFilterPair(robot_id, obj_id, joint, -1,
											enableCollision=0)
					for copy_num, obj_copy_id in enumerate(sim_data['objs'][obj_id]['copies']):
						if (copy_num+1 in sim_data['valid_groups']):
							sim.setCollisionFilterPair(robot_id, obj_copy_id, joint, -1,
											enableCollision=0)

	def enableCollisionsWithObjects(self, sim_id, ignore=None, simulator=None):
		sim = self.sims[sim_id] if simulator is None else simulator
		sim_data = self.sim_datas[sim_id]

		robot_id = sim_data['robot_id']
		for joint in get_joints(robot_id, sim=sim):
			for obj_id in sim_data['objs']:
				if (not sim_data['objs'][obj_id]['movable']):
					sim.setCollisionFilterPair(robot_id, obj_id, joint, -1,
											enableCollision=1)
				else:
					if (ignore is not None and obj_id in ignore):
						continue

					if (0 in sim_data['valid_groups']):
						sim.setCollisionFilterPair(robot_id, obj_id, joint, -1,
											enableCollision=1)
					for copy_num, obj_copy_id in enumerate(sim_data['objs'][obj_id]['copies']):
						if (copy_num+1 in sim_data['valid_groups']):
							sim.setCollisionFilterPair(robot_id, obj_copy_id, joint, -1,
											enableCollision=1)

	def addBox(self, req, sim_id, duplicate=False):
		sim = self.sims[sim_id]
		sim_data = self.sim_datas[sim_id]

		xyz = [req.o_x, req.o_y, req.o_z]
		rpy = [req.o_r, req.o_p, req.o_yaw]
		half_extents = [req.x_size, req.y_size, req.z_size]
		mass = 0 if req.locked else 0.1 + np.random.rand()
		mass = mass if req.mass <= 0 else 0.1 + req.mass * 2

		if sim_data['num_objs'] < len(sim_data['table_id']):
			rgba = self.table_rgba
			specular = self.table_specular
		else:
			rgba = list(np.random.rand(3)) + [1]
			specular = list(np.random.rand(3))

		vis_id = sim.createVisualShape(shapeType=sim.GEOM_BOX,
								halfExtents=half_extents,
								rgbaColor=rgba,
								specularColor=specular)
		coll_id = sim.createCollisionShape(shapeType=sim.GEOM_BOX,
								halfExtents=half_extents)
		body_id = sim.createMultiBody(baseMass=mass,
								baseCollisionShapeIndex=coll_id,
								baseVisualShapeIndex=vis_id,
								basePosition=xyz,
								baseOrientation=sim.getQuaternionFromEuler(rpy),
								baseInertialFramePosition=[0, 0, 0],
								baseInertialFrameOrientation=[0, 0, 0, 1])

		if (not duplicate):
			sim_data['objs'][body_id] = {
									'shape': req.shape,
									'vis': vis_id,
									'coll': coll_id,
									'mass': mass,
									'xyz': xyz,
									'rpy': rpy,
									'extents': [req.x_size, req.y_size, req.z_size],
									'ycb': False,
								}

		return body_id

	def addBoxNew(self, req, sim_id):
		return self.addBox(req, sim_id)

	def addBoxDuplicate(self, req, sim_id, orig_id):
		copy_id = self.addBox(req, sim_id, duplicate=True)
		self.sim_datas[sim_id]['objs'][orig_id]['copies'].append(copy_id)

		return copy_id

	def addCylinder(self, req, sim_id, duplicate=False):
		sim = self.sims[sim_id]
		sim_data = self.sim_datas[sim_id]

		xyz = [req.o_x, req.o_y, req.o_z]
		rpy = [req.o_r, req.o_p, req.o_yaw]
		radius = max([req.x_size, req.y_size])
		height = req.z_size
		mass = 0 if req.locked else np.random.rand() * 0.5
		mass = mass if req.mass <= 0 else req.mass

		vis_id = sim.createVisualShape(shapeType=sim.GEOM_CYLINDER,
								radius=radius,
								length=height,
								rgbaColor=list(np.random.rand(3)) + [1],
								specularColor=list(np.random.rand(3)))
		coll_id = sim.createCollisionShape(shapeType=sim.GEOM_CYLINDER,
								radius=radius,
								height=height)
		body_id = sim.createMultiBody(baseMass=mass,
								baseCollisionShapeIndex=coll_id,
								baseVisualShapeIndex=vis_id,
								basePosition=xyz,
								baseOrientation=sim.getQuaternionFromEuler(rpy),
								baseInertialFramePosition=[0, 0, 0],
								baseInertialFrameOrientation=[0, 0, 0, 1])

		if (not duplicate):
			sim_data['objs'][body_id] = {
									'shape': req.shape,
									'vis': vis_id,
									'coll': coll_id,
									'mass': mass,
									'xyz': xyz,
									'rpy': rpy,
									'extents': [req.x_size, req.y_size, req.z_size],
									'ycb': False,
								}

		return body_id

	def addCylinderNew(self, req, sim_id):
		return self.addCylinder(req, sim_id)

	def addCylinderDuplicate(self, req, sim_id, orig_id):
		copy_id = self.addCylinder(req, sim_id, duplicate=True)
		self.sim_datas[sim_id]['objs'][orig_id]['copies'].append(copy_id)

		return copy_id

	def checkInteractions(self, sim_id, objects, simulator=None):
		sim = self.sims[sim_id] if simulator is None else simulator
		sim_data = self.sim_datas[sim_id]

		robot_id = sim_data['robot_id']
		table_id = sim_data['table_id']

		interactions = []
		for obj1 in objects:
			obj1_id = obj1.id
			if (obj1_id in table_id): # do not include interactions with the table
				continue

			if (not sim_data['objs'][obj1_id]['movable']):
				contacts = sim.getContactPoints(obj1_id, robot_id)
				if any(pt[8] < CONTACT_THRESH for pt in contacts):
					# robot interacted with immovable obstacle
					interactions.append(obj1_id)
					# action failed
					# no more checks needed
					break

				contacts = tuple()
				for obj2 in objects:
					obj2_id = obj2.id
					if (obj2_id in table_id or obj2_id == obj1_id or not sim_data['objs'][obj2_id]['movable']):
						continue

					# check if any other movable object interacts with this immovable obstacle
					if (0 in sim_data['valid_groups']):
						contacts += sim.getContactPoints(obj1_id, obj2_id)
					for copy_num, obj_copy_id in enumerate(sim_data['objs'][obj2_id]['copies']):
						if (copy_num+1 in sim_data['valid_groups']):
							contacts += sim.getContactPoints(obj1_id, obj_copy_id)

				if any(pt[8] < CONTACT_THRESH for pt in contacts):
					# some movable object interacted with immovable obstacle
					interactions.append(obj1_id)
					# action failed
					# no more checks needed
					break
			else:
				# (robot, movable object) contacts
				contacts = tuple()
				if (0 in sim_data['valid_groups']):
					contacts += sim.getContactPoints(obj1_id, robot_id)
				for copy_num, obj_copy_id in enumerate(sim_data['objs'][obj1_id]['copies']):
					if (copy_num+1 in sim_data['valid_groups']):
						contacts += sim.getContactPoints(obj_copy_id, robot_id)

				if any(pt[8] < CONTACT_THRESH for pt in contacts):
					interactions.append(obj1_id)

		return interactions

	def checkPoseConstraints(self, sim_id, grasp_at=-1, ooi=-1, simulator=None):
		sim = self.sims[sim_id] if simulator is None else simulator
		sim_data = self.sim_datas[sim_id]

		groups_to_remove = []
		for obj_id in sim_data['objs']:
			if ((grasp_at >= 0 and obj_id == ooi) or not sim_data['objs'][obj_id]['movable']):
				continue

			start_rpy = sim_data['objs'][obj_id]['rpy']
			if (0 in sim_data['valid_groups']):
				curr_xyz, curr_rpy = sim.getBasePositionAndOrientation(obj_id)
				if (shortest_angle_dist(curr_rpy[0], start_rpy[0]) > 0.95 * FALL_POS_THRESH or
					shortest_angle_dist(curr_rpy[1], start_rpy[1]) > 0.95 * FALL_POS_THRESH or
					curr_xyz[2] < 0.5): # off the table/refrigerator
					groups_to_remove.append(0)
			for copy_num, obj_copy_id in enumerate(sim_data['objs'][obj_id]['copies']):
				if (copy_num+1 in sim_data['valid_groups']):
					curr_xyz, curr_rpy = sim.getBasePositionAndOrientation(obj_copy_id)
					if(shortest_angle_dist(curr_rpy[0], start_rpy[0]) > 0.95 * FALL_POS_THRESH or
						shortest_angle_dist(curr_rpy[1], start_rpy[1]) > 0.95 * FALL_POS_THRESH or
						curr_xyz[2] < 0.5): # off the table/refrigerator
						groups_to_remove.append(copy_num+1)

		groups_to_remove = list(np.unique(np.array(groups_to_remove)))
		for obj_id in sim_data['objs']:
			if (not sim_data['objs'][obj_id]['movable']):
				continue

			if (0 in groups_to_remove):
				sim.setCollisionFilterGroupMask(obj_id, -1, self.no_collision_group, self.no_collision_group)
			for copy_num, obj_copy_id in enumerate(sim_data['objs'][obj_id]['copies']):
				if (copy_num+1 in groups_to_remove):
					sim.setCollisionFilterGroupMask(obj_copy_id, -1, self.no_collision_group, self.no_collision_group)

		if (groups_to_remove):
			print(bcolors.YELLOW + "\t\tcheckPoseConstraints valid_groups (before), groups_to_remove: [{}], [{}]".format(', '.join(map(str, sim_data['valid_groups'])), ', '.join(map(str, groups_to_remove))) + bcolors.ENDC)
			[sim_data['valid_groups'].remove(group) for group in groups_to_remove]
			print(bcolors.YELLOW + "\t\tcheckPoseConstraints valid_groups (after): [{}]".format(', '.join(map(str, sim_data['valid_groups']))) + bcolors.ENDC)

		return len(sim_data['valid_groups']) == 0

	def checkTableCollision(self, sim_id, simulator=None):
		sim = self.sims[sim_id] if simulator is None else simulator
		sim_data = self.sim_datas[sim_id]

		robot_id = sim_data['robot_id']
		table_id = sim_data['table_id']

		contact_data = []
		for i in table_id:
			contacts = sim.getContactPoints(i)
			contact_data += [(pt[2], pt[8]) for pt in contacts]

		return (any(x[0] == robot_id and x[1] < CONTACT_THRESH for x in contact_data))

	def checkVelConstraints(self, sim_id, grasp_at=-1, ooi=-1, simulator=None):
		sim = self.sims[sim_id] if simulator is None else simulator
		sim_data = self.sim_datas[sim_id]

		groups_to_remove = []
		for obj_id in sim_data['objs']:
			if ((grasp_at >= 0 and obj_id == ooi) or not sim_data['objs'][obj_id]['movable']):
				continue

			if (0 in sim_data['valid_groups']):
				vel_xyz, vel_rpy = sim.getBaseVelocity(obj_id)
				if (any(np.abs(np.array(vel_xyz)) > 0.95 * FALL_VEL_THRESH)):
					groups_to_remove.append(0)
			for copy_num, obj_copy_id in enumerate(sim_data['objs'][obj_id]['copies']):
				if (copy_num+1 in sim_data['valid_groups']):
					vel_xyz, vel_rpy = sim.getBaseVelocity(obj_copy_id)
					if (any(np.abs(np.array(vel_xyz)) > 0.95 * FALL_VEL_THRESH)):
						groups_to_remove.append(copy_num+1)

		groups_to_remove = list(np.unique(np.array(groups_to_remove)))
		for obj_id in sim_data['objs']:
			if (not sim_data['objs'][obj_id]['movable']):
				continue

			if (0 in groups_to_remove):
				sim.setCollisionFilterGroupMask(obj_id, -1, self.no_collision_group, self.no_collision_group)
			for copy_num, obj_copy_id in enumerate(sim_data['objs'][obj_id]['copies']):
				if (copy_num+1 in groups_to_remove):
					sim.setCollisionFilterGroupMask(obj_copy_id, -1, self.no_collision_group, self.no_collision_group)

		if (groups_to_remove):
			print(bcolors.YELLOW + "\t\tcheckPoseConstraints valid_groups (before), groups_to_remove: [{}], [{}]".format(', '.join(map(str, sim_data['valid_groups'])), ', '.join(map(str, groups_to_remove))) + bcolors.ENDC)
			[sim_data['valid_groups'].remove(group) for group in groups_to_remove]
			print(bcolors.YELLOW + "\t\tcheckPoseConstraints valid_groups (after): [{}]".format(', '.join(map(str, sim_data['valid_groups']))) + bcolors.ENDC)

		return len(sim_data['valid_groups']) == 0

	def holdPosition(self, sim_id, simulator=None):
		sim = self.sims[sim_id] if simulator is None else simulator
		sim_data = self.sim_datas[sim_id]

		robot_id = sim_data['robot_id']
		joint_idxs = sim_data['joint_idxs']
		joints_state = sim.getJointStates(robot_id, joint_idxs)

		joints_pos = []
		for joint_state in joints_state:
			joints_pos.append(joint_state[0])
		sim.setJointMotorControlArray(robot_id, joint_idxs,
								controlMode=sim.POSITION_CONTROL,
								targetPositions=joints_pos)

	def grasp(self, sim_id, open_g, oid=None, simulator=None):
		sim = self.sims[sim_id] if simulator is None else simulator
		sim_data = self.sim_datas[sim_id]

		robot_id = sim_data['robot_id']
		self.holdPosition(sim_id, simulator=sim)

		target_vel = 0.5 * np.ones(self.n_gripper_joints)
		if not open_g:
			target_vel = -1*target_vel

		sim.setJointMotorControlArray(
				robot_id, self.gripper_joints,
				controlMode=sim.VELOCITY_CONTROL,
				targetVelocities=target_vel)
		for i in range(int(HZ)):
			sim.stepSimulation()
			if (not open_g and oid is not None):
				if (self.grasped(oid, sim_id, simulator=sim)):
					break

		if open_g:
			sim.setJointMotorControlArray(
					robot_id, self.gripper_joints,
					controlMode=sim.VELOCITY_CONTROL,
					targetVelocities=target_vel*0)

			# for i in range(int(HZ)):
			# 	sim.stepSimulation()

	def grasped(self, ooi, sim_id, simulator=None):
		sim = self.sims[sim_id] if simulator is None else simulator
		sim_data = self.sim_datas[sim_id]
		robot_id = sim_data['robot_id']

		good_grasp = True
		if (0 in sim_data['valid_groups']):
			contacts = sim.getContactPoints(ooi, robot_id)
			good_grasp = any([c[4] in self.gripper_joints for c in contacts])

		if (good_grasp):
			for copy_num, obj_copy_id in enumerate(sim_data['objs'][ooi]['copies']):
				if (copy_num+1 in sim_data['valid_groups']):
					contacts = sim.getContactPoints(obj_copy_id, robot_id)
					good_grasp = good_grasp and any([c[4] in self.gripper_joints for c in contacts])
					if not good_grasp:
						break

		return good_grasp

	def addGraspConstraint(self, obj_id, group_id, sim_id, simulator=None):
		sim = self.sims[sim_id] if simulator is None else simulator
		sim_data = self.sim_datas[sim_id]

		obj_transform = sim.getBasePositionAndOrientation(obj_id)
		robot_link = link_from_name(sim_data['robot_id'], 'r_gripper_finger_dummy_planning_link', sim=sim)

		ee_state = get_link_state(sim_data['robot_id'], robot_link, sim=sim)
		ee_transform = sim.invertTransform(ee_state.linkWorldPosition, ee_state.linkWorldOrientation)

		grasp_pose = sim.multiplyTransforms(ee_transform[0], ee_transform[1], *obj_transform)
		grasp_point, grasp_quat = grasp_pose

		constraint = sim.createConstraint(
						sim_data['robot_id'], robot_link, obj_id, BASE_LINK,
						sim.JOINT_FIXED, jointAxis=[0., 0., 0.],
						parentFramePosition=grasp_point,
						childFramePosition=[0., 0., 0.],
						parentFrameOrientation=grasp_quat,
						childFrameOrientation=sim.getQuaternionFromEuler([0., 0., 0.]))

		obj_info = sim.getDynamicsInfo(obj_id, -1)
		curr_mass = obj_info[0]
		sim.changeDynamics(obj_id, -1, mass=self.grasped_mass)
		sim_data['grasp_constraint'].append((obj_id, group_id, sim_id, constraint, curr_mass))

	def get_object_pose(self, obj_id, sim_id, simulator=None):
		sim = self.sims[sim_id] if simulator is None else simulator

		xyz, quat = sim.getBasePositionAndOrientation(obj_id)
		rpy = sim.getEulerFromQuaternion(quat)
		obj_pose = ObjectPose()
		obj_pose.id = obj_id
		obj_pose.xyz = xyz
		obj_pose.rpy = rpy
		return obj_pose

	def getObjects(self, sim_id, simulator=None):
		sim = self.sims[sim_id] if simulator is None else simulator
		sim_data = self.sim_datas[sim_id]
		table_id = sim_data['table_id']

		objects = []
		for obj_id in sim_data['objs']:
			if (obj_id in table_id):
				continue

			obj_pose = self.get_object_pose(obj_id, sim_id, simulator=sim)
			objects.append(obj_pose)

		return objects

	def getValidObjects(self, sim_id, simulator=None):
		sim = self.sims[sim_id] if simulator is None else simulator
		sim_data = self.sim_datas[sim_id]
		table_id = sim_data['table_id']

		objects = []
		for obj_id in sim_data['objs']:
			if (obj_id in table_id):
				continue

			if (not sim_data['objs'][obj_id]['movable']):
				obj_pose = self.get_object_pose(obj_id, sim_id, simulator=sim)
				objects.append(obj_pose)
			else:
				if (0 in sim_data['valid_groups']):
					obj_pose = self.get_object_pose(obj_id, sim_id, simulator=sim)
					objects.append(obj_pose)
				else:
					for copy_num, obj_copy_id in enumerate(sim_data['objs'][obj_id]['copies']):
						if (copy_num+1 in sim_data['valid_groups']):
							obj_pose = self.get_object_pose(obj_copy_id, sim_id, simulator=sim)
							obj_pose.id = obj_id
							objects.append(obj_pose)
							break

		return objects

	def resetObjects(self, sim_id, objects, simulator=None):
		sim = self.sims[sim_id] if simulator is None else simulator
		sim_data = self.sim_datas[sim_id]

		for object in objects:
			sim.resetBasePositionAndOrientation(object.id,
										posObj=object.xyz,
										ornObj=sim.getQuaternionFromEuler(object.rpy))

			for obj_copy_id in sim_data['objs'][object.id]['copies']:
				sim.resetBasePositionAndOrientation(obj_copy_id,
									posObj=object.xyz,
									ornObj=sim.getQuaternionFromEuler(object.rpy))

	def setupSim(self, shadows):
		connection = p.GUI if self.gui else p.DIRECT
		sim = bc.BulletClient(connection_mode=connection)
		sim.setAdditionalSearchPath(pybullet_data.getDataPath())
		sim.setGravity(0,0,-9.81)
		ground_plane_id = sim.loadURDF("plane.urdf")
		# sim.setRealTimeSimulation(0)
		sim.setTimeStep(1.0/HZ)
		info = {'ground_plane_id': ground_plane_id}

		sim.configureDebugVisualizer(sim.COV_ENABLE_GUI, False)
		sim.configureDebugVisualizer(sim.COV_ENABLE_TINY_RENDERER, True)
		sim.configureDebugVisualizer(sim.COV_ENABLE_RGB_BUFFER_PREVIEW, False)
		sim.configureDebugVisualizer(sim.COV_ENABLE_DEPTH_BUFFER_PREVIEW, False)
		sim.configureDebugVisualizer(sim.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, False)
		sim.configureDebugVisualizer(sim.COV_ENABLE_SHADOWS, shadows)
		sim.configureDebugVisualizer(sim.COV_ENABLE_WIREFRAME, True)

		if not self.fridge:
			info['table_id'] = [1]
		else:
			info['table_id'] = [1, 2, 3, 4, 5]
		info['robot_id'] = -1
		info['objs'] = {}
		info['num_objs'] = 0
		info['grasp_constraint'] = []
		info['valid_groups'] = list(range(self.obj_copies))

		return sim, info

	def run_jobs_in_threads(self, function, params):
		manager = mp.Manager()
		retvals = manager.dict()

		threads = []
		for sim_id, sim in enumerate(self.sims):
			thread = mp.Process(target=function, args=(sim_id, sim, retvals, params))
			threads.append(thread)
			thread.start()

		for t in threads:
			t.join()

		return retvals.items() # list of tuples (sim_id, output retval)

if __name__ == '__main__':
	sim = BulletSim(GUI)
	rospy.spin()
