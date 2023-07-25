#include <pushplan/agents/robot.hpp>
#include <pushplan/agents/agent.hpp>
#include <pushplan/search/cbs_nodes.hpp>
#include <pushplan/utils/constants.hpp>
#include <pushplan/utils/discretisation.hpp>
#include <pushplan/utils/geometry.hpp>
#include <pushplan/utils/pr2_allowed_collision_pairs.h>

#include <smpl/stl/memory.h>
#include <sbpl_collision_checking/shapes.h>
#include <sbpl_collision_checking/types.h>
#include <sbpl_collision_checking/voxel_operations.h>
#include <sbpl_collision_checking/robot_motion_collision_model.h>
#include <smpl/angles.h>
#include <smpl/debug/marker_utils.h>
#include <smpl/console/console.h>
#include <smpl/ros/factories.h>
#include <smpl_urdf_robot_model/urdf_robot_model.h>
#include <smpl/distance_map/euclid_distance_map.h>
#include <smpl/ros/propagation_distance_field.h>
#include <eigen_conversions/eigen_msg.h>
#include <leatherman/print.h>
#include <moveit_msgs/PlanningScene.h>
#include <moveit_msgs/GetMotionPlan.h>

#include <thread>

namespace clutter
{

size_t HashPush::operator()(
	const std::tuple<ObjectState, Coord, int> &push_info) const
{
	const auto &object_state = std::get<0>(push_info);
	const auto &push_end_coord = std::get<1>(push_info);
	const auto &aidx = std::get<2>(push_info);

	size_t hash_val = 0;

	// hash pushed object initial state
	boost::hash_combine(hash_val, object_state.id());
	const auto &disc_pose = object_state.disc_pose();
	boost::hash_combine(hash_val, disc_pose.x());
	boost::hash_combine(hash_val, disc_pose.y());

	bool p = disc_pose.pitch() != 0, r = disc_pose.roll() != 0;
	if (!object_state.symmetric() || p || r)
	{
		boost::hash_combine(hash_val, disc_pose.yaw());
		if (p) {
			boost::hash_combine(hash_val, disc_pose.pitch());
		}
		if (r) {
			boost::hash_combine(hash_val, disc_pose.roll());
		}
	}

	// hash push end pose
	boost::hash_combine(hash_val, push_end_coord.at(0));
	boost::hash_combine(hash_val, push_end_coord.at(1));

	// hash push action (i.e. push fraction)
	boost::hash_combine(hash_val, aidx);

	return hash_val;
}

bool EqualsPush::operator()(
	const std::tuple<ObjectState, Coord, int> &a,
	const std::tuple<ObjectState, Coord, int> &b) const
{
	return a == b;
}

bool Robot::Setup()
{
	///////////
	// Stats //
	///////////

	m_stats["plan_push_calls"] = 0;
	m_stats["push_start_poses_found"] = 0;
	m_stats["push_actions_found"] = 0;
	m_stats["push_plan_time"] = 0.0;
	m_stats["push_sim_time"] = 0.0;
	m_stats["push_sim_successes"] = 0;
	m_stats["push_db_lookup_time"] = 0.0;
	m_stats["push_db_total_time"] = 0.0;
	m_stats["push_found_in_db"] = 0;
	m_stats["push_db_successes"] = 0;
	m_stats["push_db_failures"] = 0;

	m_debug_push_info["start_inside_obstacle"] = 0;
	m_debug_push_info["start_unreachable"] = 0;
	m_debug_push_info["ik_fail"] = 0;
	m_debug_push_info["ik_joint_limit"] = 0;
	m_debug_push_info["obstacle_collision"] = 0;
	m_debug_push_info["ik_ended_early"] = 0;
	m_debug_push_info["sim_success"] = 0;
	m_debug_push_info["no_collision_ooi"] = 0;
	m_debug_push_info["sim_fail"] = 0;

	/////////////////
	// Robot Model //
	/////////////////

	// Robot description required to initialize collision checker and robot
	// model...
	auto robot_description_key = "robot_description";
	std::string robot_description_param;
	if (!m_nh.searchParam(robot_description_key, robot_description_param))
	{
		ROS_ERROR("Failed to find 'robot_description' key on the param server");
		return false;
	}

	if (!m_nh.getParam(robot_description_param, m_robot_description))
	{
		ROS_ERROR("Failed to retrieve param 'robot_description' from the param server");
		return false;
	}

	if (!readRobotModelConfig(ros::NodeHandle("~robot_model")))
	{
		ROS_ERROR("Failed to read robot model config from param server");
		return false;
	}

	// Everyone needs to know the name of the planning frame for reasons...
	// ...frame_id for the occupancy grid (for visualization)
	// ...frame_id for collision objects (must be the same as the grid, other than that, useless)
	if (!m_ph.getParam("planning_frame", m_planning_frame))
	{
		ROS_ERROR("Failed to retrieve param 'planning_frame' from the param server");
		return false;
	}

	if (!setupRobotModel() || !m_rm)
	{
		ROS_ERROR("Failed to set up Robot Model");
		return false;
	}

	//////////////////////////////////
	// Initialize Collision Checker //
	//////////////////////////////////

	initOccupancyGrids();

	if (!initCollisionChecker())
	{
		ROS_ERROR("Failed to initialise collision checker!");
		return false;
	}
	m_cc_i->setWorldToModelTransform(Eigen::Affine3d::Identity());
	m_cc_m->setWorldToModelTransform(Eigen::Affine3d::Identity());

	// Read in start state from file and update the scene...
	// Start state is also required by the planner...
	m_start_state.joint_state.name.clear();
	m_start_state.joint_state.position.clear();
	if (!readStartState())
	{
		ROS_ERROR("Failed to get initial configuration.");
		return false;
	}

	if(!setReferenceStartState())
	{
		ROS_ERROR("Failed to set start state!");
		return false;
	}

	// from smpl::ManipLattice
	m_min_limits.resize(m_rm->jointVariableCount());
	m_max_limits.resize(m_rm->jointVariableCount());
	m_continuous.resize(m_rm->jointVariableCount());
	m_bounded.resize(m_rm->jointVariableCount());
	for (int jidx = 0; jidx < m_rm->jointVariableCount(); ++jidx)
	{
		m_min_limits[jidx] = m_rm->minPosLimit(jidx);
		m_max_limits[jidx] = m_rm->maxPosLimit(jidx);
		m_continuous[jidx] = m_rm->isContinuous(jidx);
		m_bounded[jidx] = m_rm->hasPosLimit(jidx);
	}

	std::vector<int> discretization(m_rm->jointVariableCount());
	std::vector<double> deltas(m_rm->jointVariableCount());
	std::vector<double> resolutions(m_rm->jointVariableCount());
	readResolutions(resolutions);
	for (size_t vidx = 0; vidx < m_rm->jointVariableCount(); ++vidx)
	{
		if (m_continuous[vidx])
		{
			discretization[vidx] = (int)std::round((2.0 * M_PI) / resolutions[vidx]);
			deltas[vidx] = (2.0 * M_PI) / (double)discretization[vidx];
		}
		else if (m_bounded[vidx])
		{
			auto span = std::fabs(m_max_limits[vidx] - m_min_limits[vidx]);
			discretization[vidx] = std::max(1, (int)std::round(span / resolutions[vidx]));
			deltas[vidx] = span / (double)discretization[vidx];
		}
		else
		{
			discretization[vidx] = std::numeric_limits<int>::max();
			deltas[vidx] = resolutions[vidx];
		}
	}

	m_coord_vals = std::move(discretization);
	m_coord_deltas = std::move(deltas);

	// setup planar approx
	m_grasp_at = -1;
	m_distD = std::uniform_real_distribution<double>(0.0, 1.0);
	m_distG = std::normal_distribution<>(0.0, 0.025);

	smpl::RobotState dummy;
	dummy.insert(dummy.begin(),
		m_start_state.joint_state.position.begin() + 1, m_start_state.joint_state.position.end());
	m_rm->computeFK(dummy); // for reasons unknown
	// if (!reinitStartState())
	// {
	// 	ROS_ERROR("Failed to reinit start state!");
	// 	return false;
	// }
	// ROS_WARN("Start state re-init!");

	m_planner_init = false;
	m_ph.getParam("robot/grasping/tries", m_grasp_tries);
	m_ph.getParam("robot/grasping/lift", m_grasp_lift);

	m_ph.getParam("robot/pushing/plan_time", m_plan_push_time);
	m_ph.param<double>("robot/pushing/control/dt", m_dt, 0.01);
	m_ph.param<double>("robot/pushing/control/T", m_push_T, 3.0);
	m_ph.param<double>("robot/vel_scale", m_vel_scale, 0.05);
	m_ph.param<double>("robot/acc_scale", m_acc_scale, 1.0);

	m_vis_pub = m_nh.advertise<visualization_msgs::Marker>( "/visualization_marker", 10);

	return true;
}

void Robot::AddObstaclesFromSim()
{
	auto immov_objs = m_sim->GetImmovableObjs();
	auto immov_obj_ids = m_sim->GetImmovableObjIDs();
	int num_immov = (int)immov_objs->size();
	int tables = FRIDGE ? 5 : 1;

	std::vector<Object> obstacles;
	for (int i = 0; i < num_immov; ++i)
	{
		Object o;
		o.desc.id = immov_obj_ids->at(i).first;
		o.desc.shape = immov_obj_ids->at(i).second;
		o.desc.type = i < tables ? -1 : 0; // immovable obstacle is either table or not
		o.desc.o_x = immov_objs->at(i).at(0);
		o.desc.o_y = immov_objs->at(i).at(1);
		o.desc.o_z = immov_objs->at(i).at(2);
		o.desc.o_roll = immov_objs->at(i).at(3);
		o.desc.o_pitch = immov_objs->at(i).at(4);
		o.desc.o_yaw = immov_objs->at(i).at(5);
		o.desc.ycb = (bool)immov_objs->at(i).back();
		if (o.desc.ycb)
		{
			auto itr = YCB_OBJECT_DIMS.find(o.desc.shape);
			if (itr != YCB_OBJECT_DIMS.end())
			{
				o.desc.x_size = itr->second.at(0);
				o.desc.y_size = itr->second.at(1);
				o.desc.z_size = itr->second.at(2);
				o.desc.o_yaw += itr->second.at(3);
			}
			o.desc.mass = -1;
			o.desc.mu = immov_objs->at(i).at(6);
		}
		else
		{
			o.desc.x_size = immov_objs->at(i).at(6);
			o.desc.y_size = immov_objs->at(i).at(7);
			o.desc.z_size = immov_objs->at(i).at(8);
			o.desc.mass = immov_objs->at(i).at(9);
			o.desc.mu = immov_objs->at(i).at(10);
		}
		o.desc.movable = false;
		o.desc.locked = i < tables ? true : false;

		o.CreateCollisionObjects();
		o.CreateSMPLCollisionObject();
		o.GenerateCollisionModels();
		o.InitProperties();
		obstacles.push_back(std::move(o));
	}

	auto obs = m_cc->GetObstacles();
	Object virtual_table = obstacles.front();
	virtual_table.desc.o_x -= virtual_table.desc.x_size + 0.12;
	virtual_table.desc.x_size = 0.1;
	virtual_table.desc.id = 999;

	virtual_table.CreateCollisionObjects();
	virtual_table.CreateSMPLCollisionObject();
	virtual_table.GenerateCollisionModels();
	virtual_table.InitProperties();
	obstacles.push_back(std::move(virtual_table));

	ProcessObstacles(obstacles);
	m_table_z = obstacles.front().desc.o_z + obstacles.front().desc.z_size;
}

void Robot::CreateVirtualTable()
{
	m_table_z = m_cc->GetTableHeight();
	auto obs = m_cc->GetObstacles();
	Object virtual_table = obs->front();
	virtual_table.desc.o_x -= virtual_table.desc.x_size + 0.12;
	virtual_table.desc.x_size = 0.1;
	virtual_table.desc.id = 999;

	virtual_table.CreateCollisionObjects();
	virtual_table.CreateSMPLCollisionObject();
	virtual_table.GenerateCollisionModels();
	virtual_table.InitProperties();
	ProcessObstacles({ &virtual_table });
	ProcessFCLObstacles({ &virtual_table });
}

void Robot::SetMovables(const std::vector<std::shared_ptr<Agent> >& agents)
{
	m_movables = agents;
	moveit_msgs::CollisionObject mov_obj;
	for (size_t i = 0; i < m_movables.size(); ++i)
	{
		m_movable_map[m_movables.at(i)->GetID()] = i;

		m_movables.at(i)->GetMoveitObj(mov_obj);
		m_movable_moveit_objs.push_back(mov_obj);
	}
}

void Robot::AddMovablesToCC()
{
	std::vector<Object*> movable_obstacles;
	for (const auto& a: m_movables) {
		movable_obstacles.push_back(a->GetObject());
	}
	ProcessObstacles(movable_obstacles, false, true);
}

bool Robot::SetScene(
	const comms::ObjectsPoses& objects, const smpl::RobotState& state, bool vis)
{
	for (const auto &object : objects.poses)
	{
		// find the collision object with this name
		auto* cc_object = findCollisionObject(std::to_string(object.id), true);
		if (!cc_object) {
			continue;
		}

		Eigen::Affine3d pose = Eigen::Translation3d(object.xyz[0], object.xyz[1], object.xyz[2]) *
			Eigen::AngleAxisd(object.rpy[2], Eigen::Vector3d::UnitZ()) *
			Eigen::AngleAxisd(object.rpy[1], Eigen::Vector3d::UnitY()) *
			Eigen::AngleAxisd(object.rpy[0], Eigen::Vector3d::UnitX());

		if (cc_object->shape_poses[0].translation() == pose.translation()
			&& cc_object->shape_poses[0].rotation() == pose.rotation()) {
			continue;
		}
		else
		{
			cc_object->shape_poses[0] = pose;
			if (!m_cc_m->moveShapes(cc_object)) {
				return false;
			}
		}
	}

	if (vis)
	{
		auto markers = m_cc_i->getCollisionRobotVisualization(state);
		for (auto& m : markers.markers) {
			m.ns = "rrt_goal_state";
		}
		SV_SHOW_INFO(markers);
		ROS_WARN("Visualization of goal state achieved by OMPL!");
		SV_SHOW_INFO(m_cc_m->getCollisionWorldVisualization());
		SV_SHOW_INFO(m_cc_i->getCollisionWorldVisualization());
	}
	return true;
}

bool Robot::ProcessObstacles(const std::vector<Object>& obstacles,
	bool remove, bool movable)
{
	for (const auto &obs: obstacles)
	{
		if (!obs.desc.ycb)
		{
			moveit_msgs::CollisionObject obj_msg;
			if (!getCollisionObjectMsg(obs, obj_msg, remove)) {
				return false;
			}
			if (!processCollisionObjectMsg(obj_msg, movable)) {
				return false;
			}
		}
		else
		{
			if (!processSTLMesh(obs, remove, movable)) {
				return false;
			}
		}
	}

	auto cc = movable ? m_cc_m.get() : m_cc_i.get();
	SV_SHOW_INFO(cc->getCollisionWorldVisualization());
	return true;
}

bool Robot::ProcessObstacles(const std::vector<Object*>& obstacles,
	bool remove, bool movable)
{
	for (const auto &obs: obstacles)
	{
		if (!obs->desc.ycb)
		{
			moveit_msgs::CollisionObject obj_msg;
			if (!getCollisionObjectMsg(*obs, obj_msg, remove)) {
				return false;
			}
			if (!processCollisionObjectMsg(obj_msg, movable)) {
				return false;
			}
		}
		else
		{
			if (!processSTLMesh(*obs, remove, movable)) {
				return false;
			}
		}
	}

	auto cc = movable ? m_cc_m.get() : m_cc_i.get();
	SV_SHOW_INFO(cc->getCollisionWorldVisualization());
	return true;
}

void Robot::ProcessFCLObstacles(std::vector<Object> *obstacles, bool remove)
{
	for (auto &o : *obstacles)
	{
		if (remove) {
			m_cc_i->RemoveFCLMovableObstacle(o.GetFCLObject());
		}
		else {
			m_cc_i->AddFCLMovableObstacle(o.GetFCLObject());
		}
	}
	m_cc_i->SetupFCL();
}

void Robot::ProcessFCLObstacles(const std::vector<Object*> &obstacles, bool remove)
{
	for (const auto &o : obstacles)
	{
		if (remove) {
			m_cc_i->RemoveFCLMovableObstacle(o->GetFCLObject());
		}
		else {
			m_cc_i->AddFCLMovableObstacle(o->GetFCLObject());
		}
	}
	m_cc_i->SetupFCL();
}

bool Robot::SteerAction(
	const smpl::RobotState& to, int steps,
	const smpl::RobotState& from, const comms::ObjectsPoses& start_objs,
	smpl::RobotState& action_end, comms::ObjectsPoses& end_objs,
	double& frac, std::uint32_t& result, int planner)
{
	// auto markers = m_cc_i->getCollisionRobotVisualization(from);
	// for (auto& m : markers.markers) {
	// 	m.ns = "rrt_from_state";
	// }
	// SV_SHOW_INFO(markers);

	result = 0x00000000; // error in action computation

	const double res = DEG5 / 5; // 1 degree
	smpl::collision::MotionInterpolation interp(m_cc_i->robotCollisionModel().get());

	auto planning_variables = m_cc_i->planningVariables();
	m_cc_i->robotMotionCollisionModel()->fillMotionInterpolation(
			from,
			to,
			planning_variables,
			res,
			interp);
	if (planner == 0 && interp.waypointCount() > steps) // RRT
	{
		frac = 0.0;
		return false;
	}

	smpl::RobotState interm;
	bool mov_collision = false;
	int collides_immov_idx = -1;
	std::size_t max_idx = interp.waypointCount() > steps ? steps : interp.waypointCount();
	for (std::size_t i = 0; i < max_idx; i++)
	{
		interp.interpolate(i, interm, planning_variables);
		if (!m_rm->checkJointLimits(interm))
		{
			frac = 0.0;
			return false;
		}

		// check if interpolated waypoint collides with movable obstacles
		if (!mov_collision) {
			mov_collision = !m_cc_m->isStateValid(interm);
		}
		// check if interpolated waypoint collides with immovable obstacles
		if (collides_immov_idx < 0)
		{
			if (!m_cc_i->isStateValid(interm))
			{
				collides_immov_idx = (int)i;
				if (planner == 0) // RRT
				{
					// we cannot reach the 'to' state since there are immovable
					// obstacles in the way; so `steer' fails for RRT
					frac = 0.0;
					return false;
				}
			}
		}

		if (mov_collision && collides_immov_idx >= 0)
		{
			SMPL_DEBUG("Action must be in collision with robot body!");
			frac = 0.0;
			return false;
		}

		if (collides_immov_idx >= 0) {
			// we know that it may be valid up to waypoint collides_immov_idx - 1
			break;
		}
	}

	if (collides_immov_idx == 0)
	{
		// SMPL_ERROR("Start state of SteerAction collides with immovable obstacles!!");
		frac = 0.0;
		return false;
	}
	result |= collides_immov_idx < 0 ? 0x00000001 : 0x00000002; // we have an action to validate, full or partial

	std::size_t action_size = collides_immov_idx < 0 ? max_idx : collides_immov_idx;
	frac = action_size/(double)interp.waypointCount();
	if (!mov_collision)
	{
		// no collision with movable objects along action, no need to simulate
		// whole or partial action is valid
		interp.interpolate(action_size - 1, action_end, planning_variables);
		end_objs = start_objs;
		result |= 0x00000004;
		SMPL_DEBUG("[SteerAction] Planner: %d, Result: %d (no collision with movables)", planner, result);
		// there was no need to simulate
		// if result is 5: full action, 6: partial action

		// markers = m_cc_i->getCollisionRobotVisualization(action_end);
		// for (auto& m : markers.markers) {
		// 	m.ns = "rrt_state";
		// }
		// SV_SHOW_INFO(markers);
		// ROS_WARN("Action SHOULD NOT collide with movable objects!");
		// SV_SHOW_INFO(m_cc_m->getCollisionWorldVisualization());
		// SV_SHOW_INFO(m_cc_i->getCollisionWorldVisualization());

		return true;
	}
	else
	{
		// simulate whole or partial action
		trajectory_msgs::JointTrajectory steer_action;
		steer_action.joint_names = m_rm->getPlanningJoints();
		trajectory_msgs::JointTrajectoryPoint action_pt;
		// starting waypoint
		action_pt.positions = from;
		action_pt.time_from_start = ros::Duration(0.0);
		steer_action.points.push_back(action_pt);

		for (std::size_t i = 1; i < action_size; i++)
		{
			// get intermediate waypoint
			interp.interpolate(i, action_pt.positions, planning_variables);
			// get time to intermediate waypoint
			auto prev_state = steer_action.points.back().positions;
			double duration = profileAction(prev_state, action_pt.positions);

			if ((i < action_size - 1) && duration < 0.3) {
				continue;
			}
			// add intermediate waypoint
			action_pt.time_from_start = steer_action.points.back().time_from_start + ros::Duration(duration);
			steer_action.points.push_back(action_pt);
		}
		// set final waypoint
		interp.interpolate(action_size - 1, action_end, planning_variables);

		// markers = m_cc_i->getCollisionRobotVisualization(action_end);
		// for (auto& m : markers.markers) {
		// 	m.ns = "rrt_state";
		// }
		// SV_SHOW_INFO(markers);
		// ROS_WARN("Action SHOULD collide with movable objects!");
		// SV_SHOW_INFO(m_cc_m->getCollisionWorldVisualization());
		// SV_SHOW_INFO(m_cc_i->getCollisionWorldVisualization());

		profileTrajectoryMoveIt(steer_action);
		int dummy, success;
		std::vector<int> dummy_v;
		// simulate for result
		bool sim_result = m_sim->SimPushes(	{ steer_action }, -1, -1.0, -1.0,
										start_objs,
										dummy, success, end_objs, dummy_v);
		frac = sim_result ? frac : 0.0;
		result |= sim_result ? 0x00000008 : 0x00000063;
		SMPL_DEBUG("[SteerAction] Planner: %d, Result: %d (action simmed)", planner, result);
		// simulated an action
		// success if result is < 100, failure otherwise
		// full action if result = 9, partial if result = 10
		return sim_result;
	}

	SMPL_ERROR("Why are we returning from the very end of Robot::SteerAction?");
	return false;
}

bool Robot::ComputeGraspTraj(const smpl::RobotState& state, trajectory_msgs::JointTrajectory& grasp_traj)
{
	ProcessObstacles({ m_ooi }, true);

	grasp_traj.joint_names.clear();
	grasp_traj.points.clear();

	grasp_traj.joint_names = m_rm->getPlanningJoints();

	trajectory_msgs::JointTrajectoryPoint p;
	p.positions = state;
	p.time_from_start = ros::Duration(0.0);
	grasp_traj.points.push_back(p);

	//////////////////
	// Append grasp //
	//////////////////

	smpl::RobotState pregrasp_state, grasp_state, postgrasp_state, retreat_state;
	if (!getStateNearPose(m_pregrasp_pose, state, pregrasp_state, 1))
	{
		ProcessObstacles({ m_ooi });
		return false;
	}

	if (!getStateNearPose(m_grasp_pose, pregrasp_state, grasp_state, 1))
	{
		ProcessObstacles({ m_ooi });
		return false;
	}

	if (!getStateNearPose(m_postgrasp_pose, grasp_state, postgrasp_state, 1))
	{
		ProcessObstacles({ m_ooi });
		return false;
	}

	Eigen::Affine3d ee_pose = m_rm->computeFK(postgrasp_state);
	double yaw, pitch, roll;
	smpl::angles::get_euler_zyx(ee_pose.rotation(), yaw, pitch, roll);
	ee_pose.translation().x() -= 0.3 * std::cos(yaw);
	ee_pose.translation().y() -= 0.3 * std::sin(yaw);
	if (!getStateNearPose(ee_pose, postgrasp_state, retreat_state, 1))
	{
		ProcessObstacles({ m_ooi });
		return false;
	}

	double pregrasp_time = profileAction(state, pregrasp_state);
	double grasp_time = profileAction(pregrasp_state, grasp_state);
	double postgrasp_time = profileAction(grasp_state, postgrasp_state);
	double retreat_time = profileAction(postgrasp_state, retreat_state);

	p.positions = pregrasp_state;
	p.time_from_start = ros::Duration(pregrasp_time) + grasp_traj.points.back().time_from_start;
	grasp_traj.points.push_back(p);

	p.positions = grasp_state;
	p.time_from_start = ros::Duration(grasp_time) + grasp_traj.points.back().time_from_start;
	grasp_traj.points.push_back(p);

	p.positions = postgrasp_state;
	p.time_from_start = ros::Duration(postgrasp_time) + grasp_traj.points.back().time_from_start;
	grasp_traj.points.push_back(p);

	p.positions = retreat_state;
	p.time_from_start = ros::Duration(retreat_time) + grasp_traj.points.back().time_from_start;
	grasp_traj.points.push_back(p);
	profileTrajectoryMoveIt(grasp_traj);

	ProcessObstacles({ m_ooi });
	return true;
}

bool Robot::CheckGraspTraj(const smpl::RobotState& state, const comms::ObjectsPoses& objects)
{
	trajectory_msgs::JointTrajectory grasp_traj;

	bool success = false;
	if (ComputeGraspTraj(state, grasp_traj))
	{
		success = m_sim->ExecTraj(grasp_traj, objects, 2, m_ooi.desc.id);
		m_sim->RemoveConstraint();
	}

	return success;
}

bool Robot::Init()
{
	m_solve.clear();
	m_grasp_at = -1;

	LatticeState s;
	s.state.insert(s.state.begin(),
		m_start_state.joint_state.position.begin() + 1, m_start_state.joint_state.position.end());

	return true;
}

bool Robot::RandomiseStart()
{
	if (!reinitStartState())
	{
		ROS_ERROR("Failed to randomise start!");
		return false;
	}

	LatticeState s;
	s.state.insert(s.state.begin(),
		m_start_state.joint_state.position.begin() + 1, m_start_state.joint_state.position.end());

	return true;
}

void Robot::ConvertTraj(
	const Trajectory& traj_in,
	trajectory_msgs::JointTrajectory& traj_out)
{
	traj_out.header.frame_id = m_planning_frame;
	traj_out.joint_names.clear();
	traj_out.points.clear();

	// fill joint names header
	auto& variable_names = m_rm->getPlanningJoints();
	for (auto& var_name : variable_names) {
		traj_out.joint_names.push_back(var_name);
	}

	// empty or number of points in the path
	if (!traj_out.joint_names.empty()) {
		traj_out.points.resize(traj_in.size());
	}

	for (size_t pidx = 0; pidx < traj_in.size(); ++pidx)
	{
		auto& point = traj_in[pidx].state;

		auto& p = traj_out.points[pidx];
		p.positions.resize(traj_out.joint_names.size());
		if (point.size() == traj_out.joint_names.size())
		{
			p.positions = point;
			if (pidx == 0) {
				p.time_from_start = ros::Duration(0.0);
			}
			else
			{
				double step_time = profileAction(traj_out.points[pidx-1].positions, p.positions);
				p.time_from_start = traj_out.points[pidx-1].time_from_start + ros::Duration(step_time);
			}
		}
		else
		{
			p.positions.insert(p.positions.begin(), point.begin(), point.begin() + traj_out.joint_names.size());
			p.time_from_start = ros::Duration(point.back());
		}
	}
	traj_out.header.stamp = ros::Time::now();
}

bool Robot::detachObject()
{
	const std::string attached_body_id = "att_obj";
	bool detach_result = m_cc_i->detachObject(attached_body_id)
							|| m_cc_m->detachObject(attached_body_id);
	return detach_result;
}

bool Robot::attachObject(Object *obj, const Eigen::Affine3d &pose)
{
	Eigen::Affine3d obj_pose = Eigen::Translation3d(obj->desc.o_x, obj->desc.o_y, pose.translation().z()) *
						Eigen::AngleAxisd(obj->desc.o_yaw, Eigen::Vector3d::UnitZ()) *
						Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
						Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());
	smpl::collision::CollisionObject* obj_co = obj->smpl_co;

	std::vector<shapes::ShapeConstPtr> shapes;
	smpl::collision::Affine3dVector transforms;
	if (obj->smpl_co)
	{
		for (size_t sidx = 0; sidx < obj->smpl_co->shapes.size(); ++sidx)
		{
			auto transform = pose.inverse() * obj_pose;
			// transform.translation().x() += 0.2;

			switch (obj->smpl_co->shapes.at(sidx)->type)
			{
				case smpl::collision::ShapeType::Box:
				{
					auto box = static_cast<smpl::collision::BoxShape*>(obj->smpl_co->shapes.at(sidx));
					shapes::ShapeConstPtr ao_shape(new shapes::Box(box->size[0], box->size[1], box->size[2]));
					shapes.push_back(std::move(ao_shape));
					break;
				}
				case smpl::collision::ShapeType::Cylinder:
				{
					auto cylinder = static_cast<smpl::collision::CylinderShape*>(obj->smpl_co->shapes.at(sidx));
					shapes::ShapeConstPtr ao_shape(new shapes::Cylinder(cylinder->radius, cylinder->height));
					shapes.push_back(std::move(ao_shape));
					break;
				}
				case smpl::collision::ShapeType::Mesh:
				{
					shapes::ShapeConstPtr ao_shape = MakeROSShape(obj->smpl_co->shapes.at(sidx));
					shapes.push_back(std::move(ao_shape));

					auto itr = YCB_OBJECT_DIMS.find(obj->desc.shape);
					if (itr != YCB_OBJECT_DIMS.end()) {
						transform.translation().z() -= (m_grasp_z - m_table_z);
					}

					break;
				}
				case smpl::collision::ShapeType::Sphere:
				case smpl::collision::ShapeType::Cone:
				case smpl::collision::ShapeType::Plane:
				case smpl::collision::ShapeType::OcTree:
				default:
				{
					ROS_ERROR("Incompatible shape type!");
					return false;
				}
			}
			transforms.push_back(transform);
		}
	}
	else
	{
		ROS_ERROR("Collision object not found!");
		return false;
	}


	const std::string attach_link = "r_gripper_palm_link";
	const std::string attached_body_id = "att_obj";
	bool attach_result = m_cc_i->attachObject(attached_body_id, shapes, transforms, attach_link)
							|| m_cc_m->attachObject(attached_body_id, shapes, transforms, attach_link);
	if (!attach_result)	{
		ROS_ERROR("Failed to attach body to '%s'", attach_link.c_str());
	}

	// auto markers = m_cc_i->getCollisionRobotVisualization(m_postgrasp_state);
	// SV_SHOW_INFO(markers);

	return attach_result;
}

bool Robot::attachAndCheckOOI(const smpl::RobotState& state)
{
	const smpl::urdf::Link* link = smpl::urdf::GetLink(&(m_rm->m_robot_model), "r_gripper_palm_link");
	auto palm_postgrasp_pose = m_rm->computeFKLink(m_postgrasp_state, link);

	if (!attachObject(&m_ooi, palm_postgrasp_pose))
	{
		ROS_ERROR("Failed to attach object.");
		return false;
	}
	else
	{
		// object attached, but are we collision free with it grasped?
		if (!m_cc_i->isStateValid(state))
		{
			ROS_ERROR("Robot state is in collision with attached object.");
			return false;
		}
	}

	return true;
}

bool Robot::planApproach(
	const std::vector<std::vector<double> >& approach_cvecs,
	moveit_msgs::MotionPlanResponse& res,
	const std::vector<Object*>& movable_obstacles,
	bool finalise,
	smpl::RobotState* start_state)
{
	bool have_obs = !movable_obstacles.empty();

	///////////////////
	// Plan approach //
	///////////////////

	detachObject();
	UpdateKDLRobot(0);
	// setGripper(false); // closed
	InitArmPlanner(false);
	bool goal_collides = !m_cc_i->isStateValid(m_pregrasp_state);

	moveit_msgs::MotionPlanRequest req;

	m_ph.param("allowed_planning_time", req.allowed_planning_time, 10.0);
	createJointSpaceGoal(m_pregrasp_state, req);
	req.max_acceleration_scaling_factor = 1.0;
	req.max_velocity_scaling_factor = 1.0;
	req.num_planning_attempts = 1;
	// req.path_constraints;
	if (goal_collides) {
		req.planner_id = "arastar.manip_cbs.joint_distance";
	}
	else {
		req.planner_id = "mhastar.manip_cbs.bfs.joint_distance";
	}

	// set appropriate start state for approach plan to pregrasp state
	moveit_msgs::RobotState orig_start = m_start_state;
	if (start_state != nullptr)
	{
		m_start_state.joint_state.position.erase(
			m_start_state.joint_state.position.begin() + 1,
			m_start_state.joint_state.position.begin() + 1 + start_state->size());
		m_start_state.joint_state.position.insert(
			m_start_state.joint_state.position.begin() + 1,
			start_state->begin(), start_state->end());
	}
	req.start_state = m_start_state;
	// req.trajectory_constraints;
	// req.workspace_parameters;

	// completely unnecessary variable
	moveit_msgs::PlanningScene planning_scene;
	planning_scene.robot_state = m_start_state;

	// SMPL_INFO("Planning to pregrasp state.");
	if (!m_planner->init_planner(planning_scene, req, res))
	{
		// ROS_ERROR("Failed to init planner!");

		if (start_state != nullptr) {
			m_start_state = orig_start;
		}
		return false;
	}

	if (have_obs && finalise) {
		// add to immovable collision checker
		ProcessObstacles(movable_obstacles);
	}

	if (!m_planner->solve_with_constraints(req, res, m_movable_moveit_objs, approach_cvecs))
	{
		// ROS_ERROR("Failed to plan to pregrasp state.");
		if (have_obs && finalise) {
			// remove from immovable collision checker
			ProcessObstacles(movable_obstacles, true);
		}

		if (start_state != nullptr) {
			m_start_state = orig_start;
		}
		return false;
	}
	// SMPL_INFO("Robot found approach plan! # wps = %d", res.trajectory.joint_trajectory.points.size());

	if (have_obs && finalise) {
		// remove from immovable collision checker
		ProcessObstacles(movable_obstacles, true);
	}

	if (start_state != nullptr) {
		m_start_state = orig_start;
	}

	return true;
}

bool Robot::planRetract(
	const std::vector<std::vector<double> >& retract_cvecs,
	moveit_msgs::MotionPlanResponse& res,
	const std::vector<Object*>& movable_obstacles,
	bool finalise)
{
	bool have_obs = !movable_obstacles.empty();

	// setGripper(true); // open
	if (!attachAndCheckOOI(m_postgrasp_state)) {
		detachObject();
		return false;
	}
	InitArmPlanner(false);

	moveit_msgs::MotionPlanRequest req;

	m_ph.param("allowed_planning_time", req.allowed_planning_time, 10.0);
	req.max_acceleration_scaling_factor = 1.0;
	req.max_velocity_scaling_factor = 1.0;
	req.num_planning_attempts = 1;

	// set start state for retract plan to postgrasp state
	moveit_msgs::RobotState orig_start = m_start_state;
	m_start_state.joint_state.position.erase(
		m_start_state.joint_state.position.begin() + 1,
		m_start_state.joint_state.position.begin() + 1 + m_postgrasp_state.size());
	m_start_state.joint_state.position.insert(
		m_start_state.joint_state.position.begin() + 1,
		m_postgrasp_state.begin(), m_postgrasp_state.end());
	req.start_state = m_start_state;

	// set goal state for retract plan to home state
	smpl::RobotState home;
	home.insert(home.begin(),
		orig_start.joint_state.position.begin() + 1, orig_start.joint_state.position.end());
	createJointSpaceGoal(home, req);
	bool goal_collides = !m_cc_i->isStateValid(home);
	if (goal_collides) {
		req.planner_id = "arastar.manip_cbs.joint_distance";
	}
	else {
		req.planner_id = "mhastar.manip_cbs.bfs.joint_distance";
	}

	// add EE path constraint - cannot roll or pitch more than 5 degrees
	addPathConstraint(req.path_constraints);

	// completely unnecessary variable
	moveit_msgs::PlanningScene planning_scene;
	planning_scene.robot_state = m_start_state;
	// SMPL_INFO("Planning to home state with attached body.");
	if (!m_planner->init_planner(planning_scene, req, res))
	{
		// ROS_ERROR("Failed to init planner!");
		m_start_state = orig_start;
		return false;
	}

	if (have_obs && finalise)
	{
		// add to immovable collision checker
		ProcessObstacles(movable_obstacles);
		ProcessFCLObstacles(movable_obstacles);
		m_cc_i->SetFCLObjectOOI(m_ooi.GetFCLObject());
		req.allowed_planning_time *= 2.0;
	}

	if (!m_planner->solve_with_constraints(req, res, m_movable_moveit_objs, retract_cvecs))
	{
		// ROS_ERROR("Failed to plan to home state with attached body.");
		detachObject();
		m_start_state = orig_start;

		if (have_obs && finalise) {
			// remove from immovable collision checker
			ProcessObstacles(movable_obstacles, true);
			ProcessFCLObstacles(movable_obstacles, true);
		}
		return false;
	}
	// SMPL_INFO("Robot found extraction plan! # wps = %d", res.trajectory.joint_trajectory.points.size());
	if (have_obs && finalise) {
		// remove from immovable collision checker
		ProcessObstacles(movable_obstacles, true);
		ProcessFCLObstacles(movable_obstacles, true);
	}

	detachObject();
	m_start_state = orig_start;

	return true;
}

void Robot::getTrajSpheres(
	const trajectory_msgs::JointTrajectory& traj,
	std::set<std::vector<double> >& spheres)
{
	double ox, oy, oz, sx, sy, sz;
	m_sim->GetShelfParams(ox, oy, oz, sx, sy, sz);

	spheres.clear();
	// setGripper(true);
	for (const auto &wp: traj.points)
	{
		auto markers = m_cc_i->getCollisionModelVisualization(wp.positions);
		for (auto& marker : markers)
		{
			double mx = marker.pose.position[0];
			double my = marker.pose.position[1];
			double mz = marker.pose.position[2];
			double mr = boost::get<smpl::visual::Ellipse>(marker.shape).axis_x - RES;
			if (mx < ox && (mx + mr + 0.05 < ox)) {
				continue;
			}
			if (my < oy && (my + mr + 0.05 < oy)) {
				continue;
			}
			if (mz < oz && (mz + mr + 0.05 < oz)) {
				continue;
			}
			if (mz > oz + sz && (mz - mr - 0.05 > oz + sz)) {
				continue;
			}

			spheres.emplace(
				std::initializer_list<double>{
					std::ceil(mx * 100.0) / 100.0,
					std::ceil(my * 100.0) / 100.0,
					std::ceil(mz * 100.0) / 100.0,
					std::ceil(mr * 100.0) / 100.0 });
		}
	}
	// setGripper(false);
}

void Robot::voxeliseTrajectory()
{
	m_traj_voxels.clear();
	std::set<std::vector<double> > spheres;
	double start_time = GetTime();
	getTrajSpheres(m_traj, spheres);

	int sphere_count = 0;
	for (const auto &s : spheres)
	{
		std::unique_ptr<smpl::collision::CollisionShape> shape;
		shape = smpl::make_unique<smpl::collision::SphereShape>(s[3]);

		std::vector<smpl::collision::CollisionShape*> shapes;
		shapes.push_back(shape.get());

		smpl::collision::AlignedVector<Eigen::Affine3d> shape_poses;
		Eigen::Affine3d transform(Eigen::Translation3d(s[0], s[1], s[2]));
		shape_poses.push_back(transform);

		// create the collision object
		smpl::collision::CollisionObject co;
		co.id = std::to_string(sphere_count++);
		co.shapes = std::move(shapes);
		co.shape_poses = std::move(shape_poses);

		if (!smpl::collision::VoxelizeObject(co, m_ngr_res, m_ngr_origin, m_ngr_gmin, m_ngr_gmax, m_traj_voxels)) {
			continue;
		}
	}

	ROS_INFO("Robot trajectory voxelisation took %f seconds", GetTime() - start_time);

	// TODO: make this faster by looping over cells in a sphere octant
	// and computing the rest via symmetry

	// int sphere_count = 0;
	// for (const auto &wp: m_traj.points)
	// {
	// 	auto markers = m_cc_i->getCollisionModelVisualization(wp.positions);
	// 	for (auto& marker : markers)
	// 	{
	// 	}
	// }
}

void Robot::UpdateNGR(bool vis)
{
	voxeliseTrajectory();
	m_grid_ngr->reset();
	for (auto& voxel_list : m_traj_voxels) {
		m_grid_ngr->addPointsToField(voxel_list);
	}

	if (vis) {
		// SV_SHOW_INFO(m_grid_ngr->getBoundingBoxVisualization());
		// SV_SHOW_INFO(m_grid_ngr->getDistanceFieldVisualization());
		SV_SHOW_INFO(m_grid_ngr->getOccupiedVoxelsVisualization());
	}
}

bool Robot::PlanApproachOnly(const std::vector<Object*>& movable_obstacles)
{
	std::vector<std::vector<double> > dummy;
	moveit_msgs::MotionPlanResponse res;
	if (!planApproach(dummy, res, movable_obstacles)) {
		return false;
	}

	m_traj = res.trajectory.joint_trajectory;
	return true;
}

bool Robot::PlanRetrieval(const std::vector<Object*>& movable_obstacles, bool finalise, smpl::RobotState* start_state)
{
	///////////////////
	// Plan approach //
	///////////////////

	// ProcessObstacles({ m_ooi });
	std::vector<std::vector<double> > dummy;
	moveit_msgs::MotionPlanResponse res_a;
	if (!planApproach(dummy, res_a, movable_obstacles, finalise, start_state))
	{
		// ProcessObstacles({ m_ooi }, true);
		return false;
	}
	// ProcessObstacles({ m_ooi }, true);

	m_traj = res_a.trajectory.joint_trajectory;

	//////////////////
	// Append grasp //
	//////////////////

	int grasp_at = m_traj.points.size();

	double grasp_time = profileAction(m_pregrasp_state, m_grasp_state);
	double postgrasp_time = profileAction(m_grasp_state, m_postgrasp_state);
	// double retreat_time = profileAction(m_postgrasp_state, m_pregrasp_state);

	trajectory_msgs::JointTrajectoryPoint p;
	p.positions = m_grasp_state;
	p.time_from_start = ros::Duration(grasp_time) + m_traj.points.back().time_from_start;
	m_traj.points.push_back(p);

	p.positions = m_postgrasp_state;
	p.time_from_start = ros::Duration(postgrasp_time) + m_traj.points.back().time_from_start;
	m_traj.points.push_back(p);

	/////////////////////
	// Plan extraction //
	/////////////////////

	moveit_msgs::MotionPlanResponse res_r;
	if (!planRetract(dummy, res_r, movable_obstacles, finalise)) {
		return false;
	}

	m_grasp_at = grasp_at;
	auto extract_start_time = m_traj.points.back().time_from_start;
	for (size_t i = 1; i < res_r.trajectory.joint_trajectory.points.size(); ++i)
	{
		auto& wp = res_r.trajectory.joint_trajectory.points[i];
		wp.time_from_start += extract_start_time;
		m_traj.points.push_back(wp);
	}
	profileTrajectoryMoveIt(m_traj);

	return true;
}

bool Robot::PlanToHomeState(const std::vector<Object*>& movable_obstacles, smpl::RobotState* start_state)
{
	bool have_obs = !movable_obstacles.empty();
	InitArmPlanner(false);

	moveit_msgs::MotionPlanRequest req;
	moveit_msgs::MotionPlanResponse res;

	m_ph.param("allowed_planning_time", req.allowed_planning_time, 10.0);
	req.max_acceleration_scaling_factor = 1.0;
	req.max_velocity_scaling_factor = 1.0;
	req.num_planning_attempts = 1;

	// set start state for retract plan to latest push end state
	moveit_msgs::RobotState orig_start = m_start_state;
	if (start_state != nullptr)
	{
		m_start_state.joint_state.position.erase(
			m_start_state.joint_state.position.begin() + 1,
			m_start_state.joint_state.position.begin() + 1 + start_state->size());
		m_start_state.joint_state.position.insert(
			m_start_state.joint_state.position.begin() + 1,
			start_state->begin(), start_state->end());
	}
	req.start_state = m_start_state;

	// set goal state for retract plan to home state
	smpl::RobotState home;
	home.insert(home.begin(),
		orig_start.joint_state.position.begin() + 1, orig_start.joint_state.position.end());
	createJointSpaceGoal(home, req);
	bool goal_collides = !m_cc_i->isStateValid(home);
	if (goal_collides) {
		req.planner_id = "arastar.manip_cbs.joint_distance";
	}
	else {
		req.planner_id = "mhastar.manip_cbs.bfs.joint_distance";
	}

	// completely unnecessary variable
	moveit_msgs::PlanningScene planning_scene;
	planning_scene.robot_state = m_start_state;
	// SMPL_INFO("Planning to home state with attached body.");
	if (!m_planner->init_planner(planning_scene, req, res))
	{
		// ROS_ERROR("Failed to init planner!");
		m_start_state = orig_start;
		return false;
	}

	if (have_obs) {
		// add to immovable collision checker
		ProcessObstacles(movable_obstacles);
	}

	std::vector<std::vector<double> > dummy;
	if (!m_planner->solve_with_constraints(req, res, m_movable_moveit_objs, dummy))
	{
		// ROS_ERROR("Failed to plan to home state with attached body.");
		m_start_state = orig_start;

		if (have_obs) {
			// remove from immovable collision checker
			ProcessObstacles(movable_obstacles, true);
		}
		return false;
	}
	// SMPL_INFO("Robot found extraction plan! # wps = %d", res.trajectory.joint_trajectory.points.size());
	if (have_obs) {
		// remove from immovable collision checker
		ProcessObstacles(movable_obstacles, true);
	}
	m_start_state = orig_start;
	m_traj = res.trajectory.joint_trajectory;

	return true;
}

bool Robot::SatisfyPath(HighLevelNode* ct_node, Trajectory** sol_path, int& expands, int& min_f)
{
	expands = 0;
	min_f = 0;
	// CBS TODO: must pick out constraints wrt planning phase:
	// (i) for planning to pregrasp, all constraints with time <= m_grasp_at
	// are active
	// (ii) for planning to home, all constraints with times >= m_grasp_at + 2
	// are active

	// Get relevant constraints - check logic per above
	std::list<std::shared_ptr<Constraint> > approach_constraints, retract_constraints;
	for (auto& constraint : ct_node->m_constraints)
	{
		if (constraint->m_me == ct_node->m_replanned)
		{
			if (constraint->m_time <= m_grasp_at) {
				approach_constraints.push_back(constraint);
			}
			else if (constraint->m_time >= m_grasp_at + 2)
			{
				std::shared_ptr<Constraint> new_constraint(new Constraint());
				new_constraint->m_me = constraint->m_me;
				new_constraint->m_other = constraint->m_other;
				new_constraint->m_time = constraint->m_time - (m_grasp_at + 2);
				new_constraint->m_q = constraint->m_q;
				retract_constraints.push_back(new_constraint);
			}
		}
	}

	std::vector<std::vector<double> > approach_cvecs, retract_cvecs;
	VecConstraints(approach_constraints, approach_cvecs);
	VecConstraints(retract_constraints, retract_cvecs);
	approach_constraints.clear();
	retract_constraints.clear();

	///////////////////
	// Plan approach //
	///////////////////

	moveit_msgs::MotionPlanResponse res_a;
	std::vector<Object*> dummy;
	if (!planApproach(approach_cvecs, res_a, dummy)) {
		return false;
	}

	auto planner_stats = m_planner->getPlannerStats();
	expands = planner_stats["expansions"];
	min_f = planner_stats["min f val"];
	m_traj = res_a.trajectory.joint_trajectory;

	//////////////////
	// Append grasp //
	//////////////////

	int grasp_at = m_traj.points.size();

	double grasp_time = profileAction(m_pregrasp_state, m_grasp_state);
	double postgrasp_time = profileAction(m_grasp_state, m_postgrasp_state);
	// double retreat_time = profileAction(m_postgrasp_state, m_pregrasp_state);

	trajectory_msgs::JointTrajectoryPoint p;
	p.positions = m_grasp_state;
	p.time_from_start = ros::Duration(grasp_time) + m_traj.points.back().time_from_start;
	m_traj.points.push_back(p);

	p.positions = m_postgrasp_state;
	p.time_from_start = ros::Duration(postgrasp_time) + m_traj.points.back().time_from_start;
	m_traj.points.push_back(p);

	/////////////////////
	// Plan extraction //
	/////////////////////

	moveit_msgs::MotionPlanResponse res_r;
	if (!planRetract(retract_cvecs, res_r, dummy)) {
		return false;
	}

	planner_stats = m_planner->getPlannerStats();
	expands += planner_stats["expansions"];
	min_f = std::min(min_f, (int)planner_stats["min f val"]);

	m_grasp_at = grasp_at;
	auto extract_start_time = m_traj.points.back().time_from_start;
	for (size_t i = 0; i < res_r.trajectory.joint_trajectory.points.size(); ++i)
	{
		auto& wp = res_r.trajectory.joint_trajectory.points[i];
		wp.time_from_start += extract_start_time;
		m_traj.points.push_back(wp);
	}

	/////////////////////////
	// Update CBS solution //
	/////////////////////////

	m_solve.clear();
	int t = 0;
	for (const auto &wp: m_traj.points)
	{
		LatticeState s;
		s.state = wp.positions;
		s.coord = Coord(m_rm->jointVariableCount());
		stateToCoord(s.state, s.coord);
		s.t = t;

		m_solve.push_back(s);
		++t;
	}
	*sol_path = &(this->m_solve);

	// SMPL_INFO("Robot has complete plan! m_solve.size() = %d", m_solve.size());
	// visTrajectory(m_solve, "solve_trajectory");
	return true;
}

bool Robot::ComputeGrasps(
	const std::vector<double>& pregrasp_goal)
{
	m_home_state.clear();
	m_home_state.insert(m_home_state.begin(),
		m_start_state.joint_state.position.begin() + 1, m_start_state.joint_state.position.end());
	m_home_pose = m_rm->computeFK(m_home_state);

	m_pregrasp_state.clear();
	m_grasp_state.clear();
	m_postgrasp_state.clear();

	bool success = false;
	UpdateKDLRobot(0);

	smpl::RobotState ee_state;
	m_grasp_z = m_table_z + ((m_distD(m_rng) * 0.05) + 0.025);
	Eigen::Affine3d ee_pose = Eigen::Translation3d(m_ooi.desc.o_x + 0.025 * std::cos(pregrasp_goal[5]), m_ooi.desc.o_y + 0.025 * std::sin(pregrasp_goal[5]), m_grasp_z) *
								Eigen::AngleAxisd(pregrasp_goal[5], Eigen::Vector3d::UnitZ()) *
								Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
								Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());

	auto* vis_name = "grasp_pose";
	SV_SHOW_INFO_NAMED(vis_name, smpl::visual::MakePoseMarkers(
		ee_pose, m_grid_i->getReferenceFrame(), vis_name));

	if (!getStateNearPose(ee_pose, ee_state, m_grasp_state, 1)) {
		return false;
	}
	m_grasp_pose = m_rm->computeFK(m_grasp_state);

	// vis_name = "grasp_state";
	// auto markers = m_cc_i->getCollisionModelVisualization(m_grasp_state);
	// for (auto& marker : markers) {
	// 	marker.ns = vis_name;
	// }
	// SV_SHOW_INFO_NAMED(vis_name, markers);

	// SMPL_INFO("Found grasp state!!");

	// compute pregrasp state
	ee_pose = m_rm->computeFK(m_grasp_state);
	ee_pose.translation().x() = pregrasp_goal[0];
	ee_pose.translation().y() = pregrasp_goal[1];

	vis_name = "pregrasp_pose";
	SV_SHOW_INFO_NAMED(vis_name, smpl::visual::MakePoseMarkers(
		ee_pose, m_grid_i->getReferenceFrame(), vis_name));

	if (getStateNearPose(ee_pose, m_grasp_state, m_pregrasp_state, 1))
	{
		m_pregrasp_pose = m_rm->computeFK(m_pregrasp_state);
		// vis_name = "pregrasp_state";
		// auto markers = m_cc_i->getCollisionModelVisualization(m_pregrasp_state);
		// for (auto& marker : markers) {
		// 	marker.ns = vis_name;
		// }
		// SV_SHOW_INFO_NAMED(vis_name, markers);

		// SMPL_INFO("Found pregrasp state!");
		ee_pose = m_rm->computeFK(m_grasp_state);
		ee_pose.translation().z() += m_grasp_lift;

		// vis_name = "postgrasp_pose";
		// SV_SHOW_INFO_NAMED(vis_name, smpl::visual::MakePoseMarkers(
		// 	ee_pose, m_grid_i->getReferenceFrame(), vis_name));

		// compute postgrasp state
		if (getStateNearPose(ee_pose, m_grasp_state, m_postgrasp_state, 1))
		{
			m_postgrasp_pose = m_rm->computeFK(m_postgrasp_state);
			// vis_name = "postgrasp_state";
			// auto markers = m_cc_i->getCollisionModelVisualization(m_postgrasp_state);
			// for (auto& marker : markers) {
			// 	marker.ns = vis_name;
			// }
			// SV_SHOW_INFO_NAMED(vis_name, markers);

			// SMPL_INFO("Found postgrasp state!!!");

			if (!attachAndCheckOOI(m_postgrasp_state)) {
				detachObject();
				return false;
			}
			detachObject();

			success = true;
		}
	}

	return success;
}

bool Robot::UpdateKDLRobot(int mode)
{
	if (mode == 0) {
		m_robot_config.chain_tip_link = m_chain_tip_link;
	}
	else {
		m_robot_config.chain_tip_link = m_robot_config.push_links.at(mode - 1);
	}

	if (!setupRobotModel() || !m_rm)
	{
		ROS_ERROR("Failed to set up Robot Model");
		return false;
	}

	if(!setReferenceStartState())
	{
		ROS_ERROR("Failed to set start state!");
		return false;
	}

	smpl::RobotState dummy;
	dummy.insert(dummy.begin(),
		m_start_state.joint_state.position.begin() + 1, m_start_state.joint_state.position.end());
	m_rm->computeFK(dummy); // for reasons unknown
}

bool Robot::InitArmPlanner(bool interp)
{
	if (!m_planner_init) {
		m_ph.setParam("robot/interpolate", interp);
		return initPlanner();
	}
	else {
		return createPlanner(interp);
	}
}

bool Robot::planToPoseGoal(
	const moveit_msgs::RobotState& start_state,
	const std::vector<Eigen::Affine3d> &poses,
	trajectory_msgs::JointTrajectory& push_traj,
	double t,
	bool path_constraints)
{
	// create planning problem to push start pose
	InitArmPlanner(false);

	moveit_msgs::MotionPlanRequest req;
	moveit_msgs::MotionPlanResponse res;

	if (poses.size() == 1)
	{
		createPoseGoalConstraint(poses[0]);
		req.goal_constraints.clear();
		req.goal_constraints.resize(1);
		req.goal_constraints[0] = m_goal;
		req.planner_id = "arastar.manip.bfs";
	}
	else
	{
		createMultiPoseGoalConstraint(poses, req);
		req.planner_id = "mhastar.manip";
		for (size_t g = 0; g < poses.size(); ++g) {
			req.planner_id += ".bfs" + std::to_string(g);
		}
	}

	req.allowed_planning_time = t;
	req.max_acceleration_scaling_factor = 1.0;
	req.max_velocity_scaling_factor = 1.0;
	req.num_planning_attempts = 1;
	req.start_state = start_state;
	if (path_constraints)
	{
		addPathConstraint(req.path_constraints);
		req.goal_constraints[0].position_constraints[0].constraint_region.primitives[0].dimensions[0] = 0.01;
		req.goal_constraints[0].position_constraints[0].constraint_region.primitives[0].dimensions[1] = 0.01;
		req.goal_constraints[0].position_constraints[0].constraint_region.primitives[0].dimensions[2] = 0.01;
		req.goal_constraints[0].orientation_constraints[0].absolute_x_axis_tolerance = 2.0 * M_PI;
		req.goal_constraints[0].orientation_constraints[0].absolute_y_axis_tolerance = 2.0 * M_PI; // these do not make a difference inside smpl
		req.goal_constraints[0].orientation_constraints[0].absolute_z_axis_tolerance = 2.0 * M_PI; //
	}

	moveit_msgs::PlanningScene planning_scene; // completely unnecessary variable
	planning_scene.robot_state = start_state;
	// solve for path to push start pose
	if (!m_planner->init_planner(planning_scene, req, res))
	{
		// ROS_ERROR("Failed to init planner!");
		return false;
	}
	if (!m_planner->solve(req, res))
	{
		// ROS_ERROR("Failed to plan.");
		return false;
	}

	push_traj = res.trajectory.joint_trajectory;
	if (push_traj.points.size() <= 1) {
		return false;
	}
	return true;
}

int Robot::computePushStateIK(
	const Eigen::Affine3d &ee_pose,
	const smpl::RobotState &seed,
	smpl::RobotState &solution)
{
	int result = 0;
	// run ik for new state
	if (m_rm->computeIKPos(ee_pose, seed, solution))
	{
		auto markers = m_cc_i->getCollisionRobotVisualization(solution);
		for (auto& m : markers.markers) {
			m.ns = "push_action_state";
		}
		SV_SHOW_INFO(markers);

		if (!m_rm->checkJointLimits(solution))
		{
			result = 3;
			++m_debug_push_info["ik_joint_limit"];
			// return false;
			return result;
		}
		if (!m_cc_i->isStateValid(solution))
		{
			result = 2;
			++m_debug_push_info["obstacle_collision"];
			// return false;
			return result;
		}
	}
	else
	{
		// SMPL_INFO("IK call failed!");
		result = 4;
		++m_debug_push_info["ik_fail"];
		// return false;
		return result;
	}

	return result;
}

int Robot::computePushPoint(
	const double time_start,
	const smpl::RobotState& jnt_positions,
	const Eigen::Affine3d& end_pose,
	trajectory_msgs::JointTrajectory& action)
{
	int result = 0;

	// Variables
	auto q_ = jnt_positions;
	Eigen::Affine3d x_ = m_rm->computeFK(q_), xo_ = x_;

	// Add waypoint to action
	trajectory_msgs::JointTrajectoryPoint action_pt;
	action_pt.positions = q_;
	action_pt.time_from_start = ros::Duration(time_start + m_dt);
	action.points.push_back(action_pt);

	int N = m_push_T / m_dt;
	for (int i = 1; i <= N; ++i)
	{
	    double t = i / (double)N;
		// move waypoint
		xo_.translation().x() = (1.0 - t) * x_.translation().x() + t * end_pose.translation().x();
		xo_.translation().y() = (1.0 - t) * x_.translation().y() + t * end_pose.translation().y();
		xo_.translation().z() = (1.0 - t) * x_.translation().z() + t * end_pose.translation().z();

		result = computePushStateIK(xo_, action_pt.positions, q_);
		if (result == 0)
		{
			action_pt.positions = q_;
			action_pt.time_from_start += ros::Duration(m_dt);
			action.points.push_back(action_pt);
		}
		else
		{
			action.points.pop_back();
			return result;
		}
	}
	// we are done
	// SMPL_INFO("SUCCESS - reached end point");
	visTrajectory(action, "push_action");
	return result;
}

int Robot::computePushPath(
	const double time_start,
	const smpl::RobotState& jnt_positions,
	const std::vector<double>& push,
	const Trajectory* path,
	trajectory_msgs::JointTrajectory& action)
{
	visMAPFPath(path);
	int result = 0;

	// Variables
	auto q_ = jnt_positions;
	Eigen::Affine3d xo_ = m_rm->computeFK(q_);

	// Add first waypoint to action
	trajectory_msgs::JointTrajectoryPoint action_pt;
	action_pt.positions = q_;
	action_pt.time_from_start = ros::Duration(time_start + m_dt);
	action.points.push_back(action_pt);

	// Compute second waypoint = ray-AABB intersection
	xo_.translation().x() = push[0];
	xo_.translation().y() = push[1];
	// Add second waypoint to action
	result = computePushStateIK(xo_, action_pt.positions, q_);
	if (result == 0)
	{
		action_pt.positions = q_;
		action_pt.time_from_start += ros::Duration(m_dt);
		action.points.push_back(action_pt);
	}
	else
	{
		action.points.pop_back();
		return result;
	}

	for (size_t i = 1; i < path->size(); ++i)
	{
		// move waypoint
		xo_.translation().x() += path->at(i).state[0] - path->at(i-1).state[0];
		xo_.translation().y() += path->at(i).state[1] - path->at(i-1).state[1];
		xo_.translation().z() += path->at(i).state[2] - path->at(i-1).state[2];

		result = computePushStateIK(xo_, action_pt.positions, q_);
		if (result == 0)
		{
			action_pt.positions = q_;
			action_pt.time_from_start += ros::Duration(m_dt);
			action.points.push_back(action_pt);
		}
		else
		{
			action.points.pop_back();
			return result;
		}
	}
	// we are done
	// SMPL_INFO("SUCCESS - reached end point");
	visTrajectory(action, "push_action");
	return result;
}

void Robot::profileTrajectoryMoveIt(trajectory_msgs::JointTrajectory& traj)
{
	robot_model_loader::RobotModelLoader robot_model_loader("/robot_description");
	m_moveit_robot_model = robot_model_loader.getModel();
	m_moveit_robot_state.reset(new moveit::core::RobotState(m_moveit_robot_model));
	m_moveit_trajectory_ptr.reset(new robot_trajectory::RobotTrajectory(m_moveit_robot_model, "right_arm"));

	// Moveit Profiling
	m_moveit_trajectory_ptr->setRobotTrajectoryMsg(*m_moveit_robot_state, traj);

	trajectory_processing::IterativeParabolicTimeParameterization iptp;
	bool success = iptp.computeTimeStamps(*m_moveit_trajectory_ptr, m_vel_scale, m_acc_scale);
	moveit_msgs::RobotTrajectory trajectory_msg;
	m_moveit_trajectory_ptr->getRobotTrajectoryMsg(trajectory_msg);
	traj = trajectory_msg.joint_trajectory;
}

void Robot::IdentifyReachableMovables(
	const std::vector<std::shared_ptr<Agent> >& agents,
	std::vector<int>& reachable_ids)
{
	reachable_ids.clear();

	std::vector<Object*> movables;
	for (const auto &a: agents) {
		movables.push_back(a->GetObject());
	}
	ProcessObstacles(movables);

	Eigen::Affine3d obj_pose;
	trajectory_msgs::JointTrajectory traj;

	for (size_t i = 0; i < movables.size(); ++i)
	{
		std::vector<Object*> plan_to = { movables[i] };
		ProcessObstacles(plan_to, true);

		obj_pose = Eigen::Translation3d(movables[i]->desc.o_x, movables[i]->desc.o_y, movables[i]->desc.o_z) *
				Eigen::AngleAxisd(movables[i]->desc.o_yaw, Eigen::Vector3d::UnitZ()) *
				Eigen::AngleAxisd(movables[i]->desc.o_pitch, Eigen::Vector3d::UnitY()) *
				Eigen::AngleAxisd(movables[i]->desc.o_roll, Eigen::Vector3d::UnitX());
		if (planToPoseGoal(m_start_state, { obj_pose }, traj))
		{
			// object i is reachable
			reachable_ids.push_back(movables[i]->desc.id);
		}
		ProcessObstacles(plan_to);
	}

	ProcessObstacles(movables, true);
}

bool Robot::PlanPickPlace(
	std::vector<double> *start_state,
	Agent* object,
	const std::vector<Object*>& other_movables,	const comms::ObjectsPoses& curr_scene,
	comms::ObjectsPoses& result,
	int& pick_at, int& place_at,
	double &plan_time, double &sim_time)
{
	plan_time = 0.0;
	sim_time = 0.0;

	moveit_msgs::RobotState pick_start_state = m_start_state;
	if (start_state != nullptr)
	{
		assert(start_state->size() == m_rm->jointVariableCount());

		pick_start_state.joint_state.position.erase(
			pick_start_state.joint_state.position.begin() + 1,
			pick_start_state.joint_state.position.begin() + 1 + start_state->size());
		pick_start_state.joint_state.position.insert(
			pick_start_state.joint_state.position.begin() + 1,
			start_state->begin(), start_state->end());
	}

	// required info about object being pushed
	const Trajectory* obj_traj = object->SolveTraj();
	// visMAPFPath(obj_traj);
	std::vector<Object*> rearranged_obj = { object->GetObject() };

	double start_time = GetTime();
	// 1. add all objects as obstacles
	// add all movable objects as obstacles into immovable collision checker
	// they should be at their latest positions
	ProcessObstacles(rearranged_obj);
	ProcessObstacles(other_movables);

	// 2. get pregrasps of object to be grasped
	std::vector<Eigen::Affine3d> pregrasps;
	object->GetObject()->GetPregrasps(pregrasps);

	// if (!pregrasps.empty())
	// {
	// 	for (int i = 0; i < pregrasps.size(); ++i)
	// 	{
	// 		visualization_msgs::Marker marker;
	// 		marker.header.frame_id = "base_footprint";
	// 		marker.header.stamp = ros::Time();
	// 		marker.ns = "object_pregrasp_poses";
	// 		marker.id = i;
	// 		marker.action = visualization_msgs::Marker::ADD;
	// 		marker.type = visualization_msgs::Marker::ARROW;

	// 		geometry_msgs::Pose pose;
	// 		tf::poseEigenToMsg(pregrasps[i], pose);
	// 		marker.pose = pose;

	// 		marker.scale.x = 0.1;
	// 		marker.scale.y = 0.015;
	// 		marker.scale.z = 0.015;

	// 		marker.color.a = 0.7; // Don't forget to set the alpha!
	// 		marker.color.r = 0.0;
	// 		marker.color.g = 0.7;
	// 		marker.color.b = 0.6;

	// 		m_vis_pub.publish(marker);

	// 		// marker.id = i + pregrasps.size();
	// 		// marker.type = visualization_msgs::Marker::SPHERE;

	// 		// marker.scale.x = 2.0 * 0.07;
	// 		// marker.scale.y = 2.0 * 0.07;
	// 		// marker.scale.z = 2.0 * 0.1;

	// 		// m_vis_pub.publish(marker);
	// 	}
	// }

	// 3. create multi pose goal
	// 4. call planner to plan approach trajectory
	trajectory_msgs::JointTrajectory pick_traj;
	if (!planToPoseGoal(pick_start_state, pregrasps, pick_traj, 20.0))
	{
		// push_result = 5;
		// ++m_debug_push_info["start_unreachable"];
		// debug_push_end = { -99.0, -99.0 };
		// debug_push = std::make_tuple(debug_push_start, debug_push_end, push_result);

		// remove all movable objects from immovable collision space
		ProcessObstacles(rearranged_obj, true);
		ProcessObstacles(other_movables, true);
		// m_stats["push_plan_time"] += GetTime() - start_time;
		return false;
	}
	ProcessObstacles(rearranged_obj, true);

	// 5. append hardcoded grasping maneuver
	pick_at = pick_traj.points.size(); // TODO: use somewhere

	Eigen::Affine3d pregrasp_pose, grasp_pose, postgrasp_pose, place_pose, retract_pose;
	smpl::RobotState pregrasp_state, grasp_state, postgrasp_state, place_state, retract_state;

	pregrasp_state = pick_traj.points.back().positions;
	pregrasp_pose = m_rm->computeFK(pregrasp_state);

	// compute EE pose at grasp
	Eigen::Affine3d ee_pose = pregrasp_pose;
	double yaw, pitch, roll;
	smpl::angles::get_euler_zyx(ee_pose.rotation(), yaw, pitch, roll);
	ee_pose.translation().x() = object->GetObject()->desc.o_x;
	ee_pose.translation().y() = object->GetObject()->desc.o_y;
	// ee_pose = Eigen::Translation3d(object->GetObject()->desc.o_x, object->GetObject()->desc.o_y, ee_pose.translation().z()) *
	// 							Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
	// 							Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
	// 							Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());
	if (!getStateNearPose(ee_pose, pregrasp_state, grasp_state, 1))
	{
		// remove all movable objects from immovable collision space
		ProcessObstacles(other_movables, true);
		return false;
	}
	grasp_pose = m_rm->computeFK(grasp_state);

	// compute EE pose at postgrasp
	ee_pose = grasp_pose;
	ee_pose.translation().z() += m_grasp_lift;
	if (!getStateNearPose(ee_pose, grasp_state, postgrasp_state, 1))
	{
		// remove all movable objects from immovable collision space
		ProcessObstacles(other_movables, true);
		return false;
	}
	postgrasp_pose = m_rm->computeFK(postgrasp_state);

	double grasp_time = profileAction(pregrasp_state, grasp_state);
	double postgrasp_time = profileAction(grasp_state, postgrasp_state);

	trajectory_msgs::JointTrajectoryPoint p;
	p.positions = grasp_state;
	p.time_from_start = ros::Duration(grasp_time) + pick_traj.points.back().time_from_start;
	pick_traj.points.push_back(p);

	p.positions = postgrasp_state;
	p.time_from_start = ros::Duration(postgrasp_time) + pick_traj.points.back().time_from_start;
	pick_traj.points.push_back(p);

	// 6. attach object to kinematic chain
	const smpl::urdf::Link* link = smpl::urdf::GetLink(&(m_rm->m_robot_model), "r_gripper_palm_link");
	auto palm_postgrasp_pose = m_rm->computeFKLink(postgrasp_state, link);

	if (!attachObject(object->GetObject(), palm_postgrasp_pose))
	{
		ROS_ERROR("Failed to attach object.");
		// remove all movable objects from immovable collision space
		ProcessObstacles(other_movables, true);
		return false;
	}
	else
	{
		auto markers = m_cc_i->getCollisionRobotVisualization(postgrasp_state);
		SV_SHOW_INFO(markers);

		// object attached, but are we collision free with it grasped?
		ProcessObstacles(other_movables, true);
		if (!m_cc_i->isStateValid(postgrasp_state))
		{
			ROS_ERROR("Robot state is in collision with attached object.");
			detachObject();
			return false;
		}
		ProcessObstacles(other_movables);
	}

	// 7. call planner to plan place trajectory
	trajectory_msgs::JointTrajectory place_traj;

	// set start state for place plan to postgrasp state
	moveit_msgs::RobotState place_start_state = m_start_state;
	place_start_state.joint_state.position.erase(
		place_start_state.joint_state.position.begin() + 1,
		place_start_state.joint_state.position.begin() + 1 + postgrasp_state.size());
	place_start_state.joint_state.position.insert(
		place_start_state.joint_state.position.begin() + 1,
		postgrasp_state.begin(), postgrasp_state.end());
	// set place goal pose
	ee_pose = Eigen::Translation3d(obj_traj->back().state.at(0), obj_traj->back().state.at(1), ee_pose.translation().z()) *
								Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ()) * // tolerance is 2*pi anyway
								Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
								Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());

	if (!planToPoseGoal(place_start_state, { ee_pose }, place_traj, 0.1, true))
	{
		// push_result = 5;
		// ++m_debug_push_info["start_unreachable"];
		// debug_push_end = { -99.0, -99.0 };
		// debug_push = std::make_tuple(debug_push_start, debug_push_end, push_result);

		// remove all movable objects from immovable collision space
		ProcessObstacles(other_movables, true);
		detachObject();
		// m_stats["push_plan_time"] += GetTime() - start_time;
		return false;
	}
	detachObject();

	// 8. append hardcoded retract action
	ee_pose = m_rm->computeFK(place_traj.points.back().positions);
	ee_pose.translation().z() -= m_grasp_lift/2.0;
	if (!getStateNearPose(ee_pose, place_traj.points.back().positions, place_state, 1))
	{
		// remove all movable objects from immovable collision space
		ProcessObstacles(other_movables, true);
		return false;
	}
	place_pose = m_rm->computeFK(place_state);

	smpl::angles::get_euler_zyx(place_pose.rotation(), yaw, pitch, roll);
	ee_pose = place_pose;
	ee_pose.translation().x() += 0.1 * std::cos(yaw + M_PI);
	ee_pose.translation().y() += 0.1 * std::sin(yaw + M_PI);
	if (!getStateNearPose(ee_pose, place_state, retract_state, 1))
	{
		// remove all movable objects from immovable collision space
		ProcessObstacles(other_movables, true);
		return false;
	}
	retract_pose = m_rm->computeFK(retract_state);

	double place_time = profileAction(place_traj.points.back().positions, place_state);
	double retract_time = profileAction(place_state, retract_state);

	p.positions = place_state;
	p.time_from_start = ros::Duration(place_time) + place_traj.points.back().time_from_start;
	place_traj.points.push_back(p);

	p.positions = retract_state;
	p.time_from_start = ros::Duration(retract_time) + place_traj.points.back().time_from_start;
	place_traj.points.push_back(p);

	// combine pick and place trajs
	m_traj = pick_traj;
	auto pick_end_time = m_traj.points.back().time_from_start;
	for (size_t i = 1; i < place_traj.points.size(); ++i)
	{
		auto& wp = place_traj.points[i];
		wp.time_from_start += pick_end_time;
		m_traj.points.push_back(wp);
	}
	place_at = pick_at + place_traj.points.size(); // TODO: use somewhere
	profileTrajectoryMoveIt(m_traj);
	plan_time = GetTime() - start_time;

	// 9. simulate placing action
	start_time = GetTime();
	bool success = m_sim->SimPickPlace(m_traj, curr_scene, pick_at, place_at, object->GetID());
	sim_time = GetTime() - start_time;
	return success;
}

// bool Robot::CheckPush(
// 	Agent* object, const State &goal, const std::vector<double>& push,
// 	trajectory_msgs::JointTrajectory &action, trajectory_msgs::JointTrajectory &traj)
// {
// 	PlanPush(nullptr, object, push)
// }

bool Robot::PlanPush(
	std::vector<double> *start_state,
	Agent* object, const std::vector<double>& push,
	const std::vector<Object*>& other_movables,	const comms::ObjectsPoses& curr_scene,
	const double& push_frac,
	comms::ObjectsPoses& result,
	int& push_result,
	std::tuple<State, State, int>& debug_push,
	double &sim_time, bool input)
{
	sim_time = 0.0;

	// required info about object being pushed
	int aidx = getPushIdx(push_frac);
	const Trajectory* obj_traj = object->SolveTraj();
	std::vector<Object*> pushed_obj = { object->GetObject() };
	Eigen::Affine3d push_start_pose, push_end_pose;
	State debug_push_start, debug_push_end;

	moveit_msgs::RobotState push_start_state = m_start_state;
	if (start_state != nullptr)
	{
		assert(start_state->size() == m_rm->jointVariableCount());

		push_start_state.joint_state.position.erase(
			push_start_state.joint_state.position.begin() + 1,
			push_start_state.joint_state.position.begin() + 1 + start_state->size());
		push_start_state.joint_state.position.insert(
			push_start_state.joint_state.position.begin() + 1,
			start_state->begin(), start_state->end());
	}
	smpl::RobotState push_start_joints(push_start_state.joint_state.position.begin() + 1, push_start_state.joint_state.position.end());;
	bool push_found = false;

	double start_time = GetTime();
	++m_stats["plan_push_calls"];

	comms::ObjectsPoses new_scene;
	trajectory_msgs::JointTrajectory new_action;
	bool push_in_db = lookupPushInDB(
						object->GetObject(), obj_traj->back().coord, aidx, curr_scene,
						new_scene, new_action);
	m_stats["push_db_lookup_time"] += GetTime() - start_time;

	if (push_in_db)
	{
		// ROS_INFO("Push found in DB in %f seconds", GetTime() - start_time);
		++m_stats["push_found_in_db"];

		push_start_pose = m_rm->computeFK(new_action.points.front().positions);
		push_end_pose = m_rm->computeFK(new_action.points.at(new_action.points.size() - 2).positions);
		debug_push_start = { push_start_pose.translation().x(), push_start_pose.translation().y() };
		debug_push_end = { push_end_pose.translation().x(), push_end_pose.translation().y() };

		ProcessObstacles(pushed_obj);
		ProcessObstacles(other_movables);

		bool success = true;
		if (!m_cc_i->isStateValid(push_start_joints))
		{
			push_result = 2;
			++m_debug_push_info["obstacle_collision"];
			debug_push = std::make_tuple(debug_push_start, debug_push_end, push_result);

			success = false;
		}

		trajectory_msgs::JointTrajectory push_traj;
		if (success && !planToPoseGoal(push_start_state, { push_start_pose }, push_traj))
		{
			push_result = 5;
			++m_debug_push_info["start_unreachable"];
			debug_push = std::make_tuple(debug_push_start, debug_push_end, push_result);

			success = false;
		}

		// remove all movable objects from immovable collision space
		ProcessObstacles(pushed_obj, true);
		ProcessObstacles(other_movables, true);
		if (success)
		{
			profileTrajectoryMoveIt(push_traj);
			auto t_prev = push_traj.points.back().time_from_start;
			for (size_t i = 1; i < new_action.points.size(); ++i)
			{
				push_traj.points.push_back(new_action.points[i]);
				push_traj.points.back().time_from_start = t_prev + (new_action.points[i].time_from_start - new_action.points[i-1].time_from_start);
				t_prev = push_traj.points.back().time_from_start;
			}

			result = new_scene;
			push_result = 0;
			++m_debug_push_info["sim_success"];
			debug_push = std::make_tuple(debug_push_start, debug_push_end, push_result);
			m_traj = push_traj;
			++m_stats["push_db_successes"];
			m_stats["push_db_total_time"] += GetTime() - start_time;
			m_stats["push_plan_time"] += GetTime() - start_time;
			// ROS_INFO("Push evaluation sans simulation using DB took %f seconds", GetTime() - start_time);
			return true;
		}
		++m_stats["push_db_failures"];
		m_stats["push_db_total_time"] += GetTime() - start_time;
	}

	m_push_trajs.clear();
	m_push_actions.clear();

	// add all movable objects as obstacles into immovable collision checker
	// they should be at their latest positions
	ProcessObstacles(pushed_obj);
	ProcessObstacles(other_movables);

	// get push start pose
	getPushStartPose(push, push_start_pose, input);
	debug_push_start = { push_start_pose.translation().x(), push_start_pose.translation().y() };

	bool failure = false;
	if (m_grid_i->getDistanceFromPoint(
		push_start_pose.translation().x(),
		push_start_pose.translation().y(),
		push_start_pose.translation().z()) <= m_grid_i->resolution())
	{
		push_result = 6;
		++m_debug_push_info["start_inside_obstacle"];
		debug_push_end = { -99.0, -99.0 };
		debug_push = std::make_tuple(debug_push_start, debug_push_end, push_result);

		failure = true;
	}

	// plan path to push start pose with all other objects as obstacles
	trajectory_msgs::JointTrajectory push_traj;
	if (!failure && !planToPoseGoal(push_start_state, { push_start_pose }, push_traj))
	{
		push_result = 5;
		++m_debug_push_info["start_unreachable"];
		debug_push_end = { -99.0, -99.0 };
		debug_push = std::make_tuple(debug_push_start, debug_push_end, push_result);

		failure = true;
	}

	// remove all movable objects from immovable collision space
	ProcessObstacles(pushed_obj, true);
	ProcessObstacles(other_movables, true);
	if (failure)
	{
		m_stats["push_plan_time"] += GetTime() - start_time;
		return false;
	}

	++m_stats["push_start_poses_found"];
	profileTrajectoryMoveIt(push_traj);
	size_t push_traj_approach_size = push_traj.points.size();

	// push action parameters
	push_end_pose = m_rm->computeFK(push_traj.points.back().positions);

	double push_dist, push_at_angle;
	if (!input)
	{
		push_dist = EuclideanDist(obj_traj->front().state, obj_traj->back().state) + 0.05;
		push_at_angle = std::atan2(
				obj_traj->back().state.at(1) - push_end_pose.translation().y(),
				obj_traj->back().state.at(0) - push_end_pose.translation().x());
	}
	else
	{
		std::vector<double> pstart = { push[0], push[1] }, pend = { push[3], push[4] };
		push_dist = EuclideanDist(pstart, pend);
		push_at_angle = std::atan2(
				push[4] - push_end_pose.translation().y(),
				push[3] - push_end_pose.translation().x());
	}

	trajectory_msgs::JointTrajectory push_action;
	// 1. get push action trajectory via inverse kinematics to end point
	// compute push action end pose
	push_end_pose.translation().x() += std::cos(push_at_angle) * push_dist * push_frac + (m_distG(m_rng));
	push_end_pose.translation().y() += std::sin(push_at_angle) * push_dist * push_frac + (m_distG(m_rng));
	// SV_SHOW_INFO_NAMED("push_end_pose", smpl::visual::MakePoseMarkers(
	// 	push_end_pose, m_grid_i->getReferenceFrame(), "push_end_pose"));
	debug_push_end = { push_end_pose.translation().x(), push_end_pose.translation().y() };
	push_result = computePushPoint(
						push_traj.points.back().time_from_start.toSec(),
						push_traj.points.back().positions,
						push_end_pose,
						push_action);
	// // 2. get push action trajectory via inverse kinematics along MAPF path
	// debug_push_end = { obj_traj->back().state.at(0) + std::cos(push[2] + M_PI) * push[3],
	// 			obj_traj->back().state.at(1) + std::sin(push[2] + M_PI) * push[3] };
	// push_result = computePushPath(
	// 					push_traj.points.back().time_from_start.toSec(),
	// 					push_traj.points.back().positions,
	// 					push,
	// 					obj_traj,
	// 					push_action);

	if (push_result > 0 || push_action.points.empty())
	{
		debug_push = std::make_tuple(debug_push_start, debug_push_end, push_result);
		m_stats["push_plan_time"] += GetTime() - start_time;
		return false;
	}

	// compute obtainable push action end pose
	push_end_pose = m_rm->computeFK(push_action.points.back().positions);

	// collision check push action against pushed object
	// ensure that it collides
	{
		bool collides = false;
		ProcessObstacles(pushed_obj);
		for (size_t wp = 1; wp < push_action.points.size(); ++wp)
		{
			if (!m_cc_i->isStateValid(push_action.points[wp].positions))
			{
				collides = true;
				break;
			}
		}

		ProcessObstacles(pushed_obj, true);
		if (!collides)
		{
			push_result = -1;
			++m_debug_push_info["no_collision_ooi"];
			debug_push = std::make_tuple(debug_push_start, debug_push_end, push_result);
			m_stats["push_plan_time"] += GetTime() - start_time;
			return false;
		}

		push_found = true;
		++m_stats["push_actions_found"];
		debug_push = std::make_tuple(debug_push_start, debug_push_end, push_result);

		// append waypoint to retract to push start pose
		push_action.points.push_back(push_action.points.at(0));
		push_action.points.back().time_from_start = push_action.points.at(push_action.points.size() - 2).time_from_start + ros::Duration(1.0);

		// append push_action to push_traj
		auto t_prev = push_traj.points.back().time_from_start;
		for (auto it = push_action.points.begin() + 1; it != push_action.points.end(); ++it)
		{
			auto tdiff = it->time_from_start - t_prev;
			// last two points in push_action must be appended to push_traj
			// they are the final state in the push action and the retraction
			// to the first state
			if ((it < push_action.points.end() - 2) && tdiff.toSec() < 0.3) {
				continue;
			}
			push_traj.points.push_back(*it);
			t_prev = it->time_from_start;
		}

		profileTrajectoryMoveIt(push_traj);
		// store the profiled push_action
		push_action.points.clear();
		push_action.points.insert(push_action.points.begin(), push_traj.points.begin() + push_traj_approach_size - 1, push_traj.points.end());

		m_push_actions.push_back(std::move(push_action));
		m_push_trajs.push_back(std::move(push_traj));
	}

	// reset robot model back to chain tip link
	UpdateKDLRobot(0);

	if (!push_found)
	{
		// we should never enter this if
		m_stats["push_plan_time"] += GetTime() - start_time;
		return false;
	}
	m_stats["push_plan_time"] += GetTime() - start_time;

	start_time = GetTime();
	int pidx, successes;
	std::vector<int> relevant_ids;
	if (!input) {
		m_sim->SimPushes(m_push_actions, object->GetID(), obj_traj->back().state.at(0), obj_traj->back().state.at(1), curr_scene, pidx, successes, result, relevant_ids);
	}
	else {
		m_sim->SimPushes(m_push_actions, object->GetID(), push[3], push[4], curr_scene, pidx, successes, result, relevant_ids);
	}
	m_stats["push_sim_time"] += GetTime() - start_time;
	sim_time += GetTime() - start_time;

	push_result = pidx == -1 ? -2 : 0;
	debug_push = std::make_tuple(debug_push_start, debug_push_end, push_result);
	if (pidx == -1)
	{
		++m_debug_push_info["sim_fail"];
		return false;
	}

	++m_debug_push_info["sim_success"];
	++m_stats["push_sim_successes"];
	m_traj = m_push_trajs.at(pidx);
	// SMPL_INFO("Computed and simulated new push in %f seconds!", GetTime() - start_time);

	addPushToDB(
		object->GetObject(), obj_traj->back().coord, aidx,
		curr_scene, result, relevant_ids, m_push_actions.at(pidx));

	return true;
}

bool Robot::GenMovablePush(
	Object& movable,
	std::vector<double>& push,
	double& to_move_dir, double& to_move_dist,
	double& moved_dir, double& moved_dist,
	Eigen::Affine3d& push_start_pose,
	Eigen::Affine3d& push_result_pose,
	int& push_result)
{
	// add object to be pushed to collision space
	std::vector<Object> movable_v = { movable };
	ProcessObstacles(movable_v);

	// get push start pose
	samplePushStartPose(push, movable, push_start_pose);
	SV_SHOW_INFO_NAMED("push_start_pose", smpl::visual::MakePoseMarkers(
		push_start_pose, m_grid_i->getReferenceFrame(), "push_start_pose"));
	// set/sample push parameters - direction and distance
	to_move_dir = push[2];
	to_move_dist = 0.1 + m_distD(m_rng) * 0.2;
	moved_dir = to_move_dir;
	moved_dist = to_move_dist;
	push_result_pose = push_start_pose;

	// plan path to push start pose
	trajectory_msgs::JointTrajectory push_traj;
	if (!planToPoseGoal(m_start_state, { push_start_pose }, push_traj, 0.5)) {
		push_result = 5;
	}

	// remove the object to pushed from collision space
	ProcessObstacles(movable_v, true);
	if (push_result == 5) {
		return false;
	}

	profileTrajectoryMoveIt(push_traj);
	size_t push_traj_approach_size = push_traj.points.size();
	// actual push start pose achieved by the robot
	push_start_pose = m_rm->computeFK(push_traj.points.back().positions);
	SV_SHOW_INFO_NAMED("push_start_pose", smpl::visual::MakePoseMarkers(
		push_start_pose, m_grid_i->getReferenceFrame(), "push_start_pose"));

	// compute push action end pose based on ee pose achieved at the end of the
	// trajectory to the push start pose
	// translate pose along to_move_dir by to_move_dist
	Eigen::Affine3d push_end_pose = push_start_pose;
	push_end_pose.translation().x() += std::cos(to_move_dir) * (to_move_dist + push[3]);
	push_end_pose.translation().y() += std::sin(to_move_dir) * (to_move_dist + push[3]);
	SV_SHOW_INFO_NAMED("push_end_pose", smpl::visual::MakePoseMarkers(
		push_end_pose, m_grid_i->getReferenceFrame(), "push_end_pose"));

	// get push action trajectory via inverse kinematics
	std::vector<trajectory_msgs::JointTrajectory> push_action(1);
	push_result = computePushPoint(
						push_traj.points.back().time_from_start.toSec(),
						push_traj.points.back().positions,
						push_end_pose,
						push_action.front());

	if (push_result > 0 || push_action.front().points.empty()) {
		return false;
	}

	// collision check push action against pushed object
	// ensure that it collides
	{
		bool collides = false;
		ProcessObstacles(movable_v);
		for (size_t wp = 1; wp < push_action.front().points.size(); ++wp)
		{
			if (!m_cc_i->isStateValid(push_action.front().points[wp].positions))
			{
				collides = true;
				break;
			}
		}

		ProcessObstacles(movable_v, true);
		if (!collides)
		{
			push_result = -1;
			return false;
		}

		// // append waypoint to retract to push start pose
		// push_action.front().points.push_back(push_action.front().points.at(0));
		// push_action.front().points.back().time_from_start = push_action.front().points.at(push_action.front().points.size() - 2).time_from_start + ros::Duration(1.0);

		// append push_action to push_traj
		auto t_prev = push_traj.points.back().time_from_start;
		for (auto it = push_action.front().points.begin() + 1; it != push_action.front().points.end(); ++it)
		{
			auto tdiff = it->time_from_start - t_prev;
			// last point in push_action must be appended to push_traj
			if ((it < push_action.front().points.end() - 1) && tdiff.toSec() < 0.3) {
				continue;
			}
			push_traj.points.push_back(*it);
			t_prev = it->time_from_start;
		}

		profileTrajectoryMoveIt(push_traj);
		// store the profiled push_action
		push_action.front().points.clear();
		push_action.front().points.insert(push_action.front().points.begin(), push_traj.points.begin() + push_traj_approach_size - 1, push_traj.points.end());
	}

	// reset robot model back to chain tip link
	UpdateKDLRobot(0);

	int pidx, successes;
	std::vector<int> relevant_ids;
	comms::ObjectsPoses dummy_o;
	m_sim->SimPushes(push_action, movable.desc.id, movable.desc.o_x, movable.desc.o_y, dummy_o, pidx, successes, dummy_o, relevant_ids);

	push_result = pidx == -1 ? -2 : 0;
	if (pidx == -1)	{
		return false;
	}

	// set moved_dir and moved_dist based on what actually happened in sim
	// HER style
	moved_dir = std::atan2(
				dummy_o.poses[0].xyz[1] - movable.desc.o_y,
				dummy_o.poses[0].xyz[0] - movable.desc.o_x);
	std::vector<double> mstart = { movable.desc.o_x, movable.desc.o_y }, mend = { dummy_o.poses[0].xyz[0], dummy_o.poses[0].xyz[1] };
	moved_dist = EuclideanDist(mstart, mend);
	push_result_pose = Eigen::Translation3d(dummy_o.poses[0].xyz[0], dummy_o.poses[0].xyz[1], dummy_o.poses[0].xyz[2]) *
						Eigen::AngleAxisd(dummy_o.poses[0].rpy[2], Eigen::Vector3d::UnitZ()) *
						Eigen::AngleAxisd(dummy_o.poses[0].rpy[1], Eigen::Vector3d::UnitY()) *
						Eigen::AngleAxisd(dummy_o.poses[0].rpy[0], Eigen::Vector3d::UnitX());

	return true;
}

void Robot::AnimateSolution()
{
	visTrajectory(m_solve, "solve_trajectory");

	size_t pidx = 0;
	while (ros::ok())
	{
		auto& point = m_solve[pidx];
		auto markers = m_cc_i->getCollisionRobotVisualization(point.state);
		for (auto& m : markers.markers) {
			m.ns = "path_animation";
		}
		SV_SHOW_INFO(markers);
		std::this_thread::sleep_for(std::chrono::milliseconds(200));
		pidx++;
		pidx %= m_solve.size();
	}
}

State Robot::GetEEState(const State& state)
{
	Eigen::Affine3d ee_pose = m_rm->computeFK(state);
	State ee = {ee_pose.translation().x(), ee_pose.translation().y(), ee_pose.translation().z(),
				0.0, 0.0, 0.0};
	smpl::angles::get_euler_zyx(ee_pose.rotation(), ee[5], ee[4], ee[3]);
	return ee;
}

void Robot::getPushStartPose(
	const std::vector<double>& push,
	Eigen::Affine3d& push_pose,
	bool input)
{
	UpdateKDLRobot(1); // set to r_gripper_tool_frame

	// z is 3cm above table height
	double z = m_table_z + 0.03;

	// (x, y) is determined by push start location
	double x = push[0];
	double y = push[1];
	if (!input)
	{
		x += std::cos(push[2] + M_PI) * 0.05 + (m_distG(m_rng));
		y += std::sin(push[2] + M_PI) * 0.05 + (m_distG(m_rng));
	}

	push_pose = Eigen::Translation3d(x, y, z) *
				Eigen::AngleAxisd(push[2], Eigen::Vector3d::UnitZ()) *
				Eigen::AngleAxisd(M_PI/6.0, Eigen::Vector3d::UnitY()) *
				Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());

	SV_SHOW_INFO_NAMED("sampled_push_pose", smpl::visual::MakePoseMarkers(
		push_pose, m_grid_i->getReferenceFrame(), "sampled_push_pose"));
}

void Robot::samplePushStartPose(
	std::vector<double>& push, Object& movable,
	Eigen::Affine3d& start_pose)
{
	UpdateKDLRobot(1); // set to r_gripper_tool_frame

	// default values
	push.clear();
	push.resize(4, 0.0);
	push[0] = movable.desc.o_x;
	push[1] = movable.desc.o_y;
	LatticeState from;
	from.state = { 	movable.desc.o_x, movable.desc.o_y, movable.desc.o_z,
						movable.desc.o_roll, movable.desc.o_pitch, movable.desc.o_yaw };

	bool done = false;
	while (!done)
	{
		// sample movement/pushing direction for object
		double move_dir = m_distD(m_rng) * 2 * M_PI;
		push[2] = move_dir;
		// compute ray-AABB point of intersection
		// push = {x, y, move_dir, t} where
		// (x, y) is the point on the AABB
		// t is the distance between the (o_x, o_y) and (x, y)
		movable.GetSE2Push(push, move_dir, from);

		// push start pose (x, y) is determined by push start location from ray-AABB intersection
		// add noise to avoid sampling inside the object
		double x0 = push[0] + std::cos(push[2] + M_PI) * 0.05 + (m_distG(m_rng));
		double y0 = push[1] + std::sin(push[2] + M_PI) * 0.05 + (m_distG(m_rng));

		// z is 3cm above table height
		double z0 = m_table_z + 0.03;

		// check if sampled (x, y, z) is inside collision voxels
		if (m_grid_i->getDistanceFromPoint(x0, y0, z0) <= m_grid_i->resolution()) {
			continue;
		}

		// complete push start pose by sampling rotational dofs
		start_pose = Eigen::Translation3d(x0, y0, z0) *
					Eigen::AngleAxisd(push[2], Eigen::Vector3d::UnitZ()) *
					Eigen::AngleAxisd(M_PI/6.0, Eigen::Vector3d::UnitY()) *
					Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());
		done = true;
	}

	// SV_SHOW_INFO_NAMED("sampled_start_pose", smpl::visual::MakePoseMarkers(
	// 	start_pose, m_grid_i->getReferenceFrame(), "sampled_start_pose"));
}

void Robot::GetRandomState(smpl::RobotState& s)
{
	s.clear();
	s.resize(m_rm->jointVariableCount(), 0.0);

	for(int jidx = 0; jidx < m_rm->jointVariableCount(); jidx++)
	{
		if (m_continuous[jidx]) {
			s.at(jidx) = m_distD(m_rng) * 2 * M_PI;
		}
		else
		{
			auto span = std::fabs(m_max_limits[jidx] - m_min_limits[jidx]);
			s.at(jidx) = m_distD(m_rng) * span + m_min_limits[jidx];
		}
	}
}

bool Robot::reinitStartState()
{
	double x, y;
	double xmin = m_cc->OutsideXMin(), xmax = m_cc->OutsideXMax() - 0.05;
	double ymin = m_cc->OutsideYMin(), ymax = m_cc->OutsideYMax();

	Eigen::Affine3d ee_pose;
	smpl::RobotState s, seed;
	seed.insert(seed.begin(),
		m_start_state.joint_state.position.begin() + 1, m_start_state.joint_state.position.end());
	do
	{
		x = (m_distD(m_rng) * (xmax - xmin)) + xmin;
		y = (m_distD(m_rng) * (ymax - ymin)) + ymin;
		ee_pose = Eigen::Translation3d(x, y, m_table_z + 0.05) *
					Eigen::AngleAxisd((m_distD(m_rng) * M_PI_2) - M_PI_4, Eigen::Vector3d::UnitZ()) *
					Eigen::AngleAxisd(((2 * m_distD(m_rng)) - 1) * DEG5, Eigen::Vector3d::UnitY()) *
					Eigen::AngleAxisd(((2 * m_distD(m_rng)) - 1) * DEG5, Eigen::Vector3d::UnitX());

		if (m_rm->computeIKSearch(ee_pose, seed, s))
		{
			if (m_rm->checkJointLimits(s)) {
				break;
			}
		}
		GetRandomState(seed);
	}
	while (true);

	m_start_state.joint_state.position.erase(
		m_start_state.joint_state.position.begin() + 1,
		m_start_state.joint_state.position.begin() + 1 + s.size());
	m_start_state.joint_state.position.insert(
		m_start_state.joint_state.position.begin() + 1,
		s.begin(), s.end());

	if(!setReferenceStartState())
	{
		ROS_ERROR("Failed to set reinit-ed start state!");
		return false;
	}

	return true;
}

// angles are counterclockwise from 0 to 360 in radians, 0 is the center of bin
// 0, ...
void Robot::coordToState(
	const Coord& coord,
	State& state) const
{
	assert((int)state.size() == m_rm->jointVariableCount() &&
			(int)coord.size() == m_rm->jointVariableCount());

	for (size_t i = 0; i < coord.size(); ++i)
	{
		if (m_continuous[i]) {
			state[i] = coord[i] * m_coord_deltas[i];
		}
		else if (!m_bounded[i]) {
			state[i] = (double)coord[i] * m_coord_deltas[i];
		}
		else {
			state[i] = m_min_limits[i] + coord[i] * m_coord_deltas[i];
		}
	}
}

void Robot::stateToCoord(
	const State& state,
	Coord& coord) const
{
	assert((int)state.size() == m_rm->jointVariableCount() &&
			(int)coord.size() == m_rm->jointVariableCount());

	for (size_t i = 0; i < state.size(); ++i)
	{
		if (m_continuous[i])
		{
			auto pos_angle = smpl::angles::normalize_angle_positive(state[i]);

			coord[i] = (int)((pos_angle + m_coord_deltas[i] * 0.5) / m_coord_deltas[i]);

			if (coord[i] == m_coord_vals[i]) {
				coord[i] = 0;
			}
		}
		else if (!m_bounded[i])
		{
			if (state[i] >= 0.0) {
				coord[i] = (int)(state[i] / m_coord_deltas[i] + 0.5);
			}
			else {
				coord[i] = (int)(state[i] / m_coord_deltas[i] - 0.5);
			}
		}
		else {
			coord[i] = (int)(((state[i] - m_min_limits[i]) / m_coord_deltas[i]) + 0.5);
		}
	}
}

bool Robot::readRobotModelConfig(const ros::NodeHandle &nh)
{
	if (!nh.getParam("group_name", m_robot_config.group_name)) {
		ROS_ERROR("Failed to read 'group_name' from the param server");
		return false;
	}

	std::string config_field;
	if (!nh.getParam("planning_joints", config_field)) {
		ROS_ERROR("Failed to read 'planning_joints' from the param server");
		return false;
	}

	std::stringstream joint_name_stream(config_field);
	while (joint_name_stream.good() && !joint_name_stream.eof()) {
		std::string jname;
		joint_name_stream >> jname;
		if (jname.empty()) {
			continue;
		}
		m_robot_config.planning_joints.push_back(jname);
	}

	config_field.clear();
	if (!nh.getParam("push_links", config_field)) {
		ROS_ERROR("Failed to read 'push_links' from the param server");
		return false;
	}

	joint_name_stream.str("");
	joint_name_stream.clear();
	joint_name_stream.str(config_field);
	while (joint_name_stream.good() && !joint_name_stream.eof()) {
		std::string jname;
		joint_name_stream >> jname;
		if (jname.empty()) {
			continue;
		}
		m_robot_config.push_links.push_back(jname);
	}

	config_field.clear();
	if (!nh.getParam("gripper_joints", config_field)) {
		ROS_ERROR("Failed to read 'gripper_joints' from the param server");
		return false;
	}

	joint_name_stream.str("");
	joint_name_stream.clear();
	joint_name_stream.str(config_field);
	while (joint_name_stream.good() && !joint_name_stream.eof()) {
		std::string jname;
		joint_name_stream >> jname;
		if (jname.empty()) {
			continue;
		}
		m_robot_config.gripper_joints.push_back(jname);
	}

	// only required for generic kdl robot model?
	nh.getParam("kinematics_frame", m_robot_config.kinematics_frame);
	nh.getParam("chain_tip_link", m_robot_config.chain_tip_link);
	m_chain_tip_link = m_robot_config.chain_tip_link;
	return true;
}

bool Robot::setupRobotModel()
{
	if (m_robot_config.kinematics_frame.empty() || m_robot_config.chain_tip_link.empty()) {
		ROS_ERROR("Failed to retrieve param 'kinematics_frame' or 'chain_tip_link' from the param server");
		m_rm = NULL;
		return false;
	}

	m_rm = std::make_unique<smpl::PushingKDLRobotModel>();

	if (!m_rm->init(m_robot_description, m_robot_config.kinematics_frame, m_robot_config.chain_tip_link)) {
		ROS_ERROR("Failed to initialize robot model.");
		m_rm = NULL;
		return false;
	}

	return true;
}

bool Robot::readStartState()
{
	XmlRpc::XmlRpcValue xlist;

	// joint_state
	if (m_ph.hasParam("initial_configuration/joint_state")) {
		m_ph.getParam("initial_configuration/joint_state", xlist);

		if (xlist.getType() != XmlRpc::XmlRpcValue::TypeArray) {
			ROS_WARN("initial_configuration/joint_state is not an array.");
		}

		if (xlist.size() > 0) {
			for (int i = 0; i < xlist.size(); ++i) {
				m_start_state.joint_state.name.push_back(std::string(xlist[i]["name"]));

				if (xlist[i]["position"].getType() == XmlRpc::XmlRpcValue::TypeDouble) {
					m_start_state.joint_state.position.push_back(double(xlist[i]["position"]));
				}
				else {
					ROS_DEBUG("Doubles in the yaml file have to contain decimal points. (Convert '0' to '0.0')");
					if (xlist[i]["position"].getType() == XmlRpc::XmlRpcValue::TypeInt) {
						int pos = xlist[i]["position"];
						m_start_state.joint_state.position.push_back(double(pos));
					}
				}
			}
		}
	}
	else {
		ROS_WARN("initial_configuration/joint_state is not on the param server.");
	}

	// multi_dof_joint_state
	if (m_ph.hasParam("initial_configuration/multi_dof_joint_state")) {
		m_ph.getParam("initial_configuration/multi_dof_joint_state", xlist);

		if (xlist.getType() == XmlRpc::XmlRpcValue::TypeArray) {
			if (xlist.size() != 0) {
				auto &multi_dof_joint_state = m_start_state.multi_dof_joint_state;
				multi_dof_joint_state.joint_names.resize(xlist.size());
				multi_dof_joint_state.transforms.resize(xlist.size());
				for (int i = 0; i < xlist.size(); ++i) {
					multi_dof_joint_state.joint_names[i] = std::string(xlist[i]["joint_name"]);

					Eigen::Quaterniond q;
					smpl::angles::from_euler_zyx(
							(double)xlist[i]["yaw"], (double)xlist[i]["pitch"], (double)xlist[i]["roll"], q);

					geometry_msgs::Quaternion orientation;
					tf::quaternionEigenToMsg(q, orientation);

					multi_dof_joint_state.transforms[i].translation.x = xlist[i]["x"];
					multi_dof_joint_state.transforms[i].translation.y = xlist[i]["y"];
					multi_dof_joint_state.transforms[i].translation.z = xlist[i]["z"];
					multi_dof_joint_state.transforms[i].rotation.w = orientation.w;
					multi_dof_joint_state.transforms[i].rotation.x = orientation.x;
					multi_dof_joint_state.transforms[i].rotation.y = orientation.y;
					multi_dof_joint_state.transforms[i].rotation.z = orientation.z;
				}
			} else {
				ROS_WARN("initial_configuration/multi_dof_joint_state array is empty");
			}
		} else {
			ROS_WARN("initial_configuration/multi_dof_joint_state is not an array.");
		}
	}

	return true;
}

bool Robot::setReferenceStartState()
{
	// Set reference state in the robot planning model...
	// <sbpl_kdl_robot_model/kdl_robot_model.h> <-
	// 	<smpl_urdf_robot_model/smpl_urdf_robot_model.h> <-
	// 		<smpl_urdf_robot_model/robot_state.h>
	smpl::urdf::RobotState reference_state;
	InitRobotState(&reference_state, &m_rm->m_robot_model);
	for (auto i = 0; i < m_start_state.joint_state.name.size(); ++i) {
		auto* var = GetVariable(&m_rm->m_robot_model, &m_start_state.joint_state.name[i]);
		if (var == NULL) {
			ROS_WARN("Failed to do the thing");
			continue;
		}
		// ROS_INFO("Set joint %s to %f", m_start_state.joint_state.name[i].c_str(), m_start_state.joint_state.position[i]);
		SetVariablePosition(&reference_state, var, m_start_state.joint_state.position[i]);
	}
	SetReferenceState(m_rm.get(), GetVariablePositions(&reference_state));

	// Set reference state in the collision scene here...
	setCollisionRobotState();

	return true;
}

bool Robot::setGripper(bool open)
{
	smpl::urdf::RobotState reference_state;
	InitRobotState(&reference_state, &m_rm->m_robot_model);
	for (auto i = 0; i < m_robot_config.gripper_joints.size(); ++i) {
		auto* var = GetVariable(&m_rm->m_robot_model, &m_robot_config.gripper_joints[i]);
		if (var == NULL) {
			ROS_WARN("Failed to do the thing");
			continue;
		}
		// ROS_INFO("Set joint %s to %f", m_start_state.joint_state.name[i].c_str(), m_start_state.joint_state.position[i]);
		double pos = open ? 0.5 : 0.0; // from gripper.urdf.xacro, limit is actually 0.548
		SetVariablePosition(&reference_state, var, pos);

		// Set state in the collision scene here...
		auto& joint_name = m_robot_config.gripper_joints[i];
		if (!m_cc_i->setJointPosition(joint_name, pos) || !m_cc_m->setJointPosition(joint_name, pos)) {
			ROS_ERROR("Failed to set position of joint '%s' to %f", joint_name.c_str(), pos);
			return false;
		}
	}
	SetReferenceState(m_rm.get(), GetVariablePositions(&reference_state));
}

bool Robot::readResolutions(std::vector<double>& resolutions)
{
	std::string disc_string;
	if (!m_ph.getParam("planning/discretization", disc_string))
	{
		SMPL_ERROR("Parameter 'discretization' not found in planning params");
		return false;
	}

	auto disc = ParseMapFromString<double>(disc_string);
	for (size_t vidx = 0; vidx < m_rm->jointVariableCount(); ++vidx)
	{
		auto& vname = m_rm->getPlanningJoints()[vidx];
		std::string joint_name, local_name;
		if (smpl::IsMultiDOFJointVariable(vname, &joint_name, &local_name))
		{
			// adjust variable name if a variable of a multi-dof joint
			auto mdof_vname = joint_name + "_" + local_name;
			auto dit = disc.find(mdof_vname);
			if (dit == end(disc))
			{
				SMPL_ERROR("Discretization for variable '%s' not found in planning parameters", vname.c_str());
				return false;
			}
			resolutions[vidx] = dit->second;
		}
		else
		{
			auto dit = disc.find(vname);
			if (dit == end(disc))
			{
				SMPL_ERROR("Discretization for variable '%s' not found in planning parameters", vname.c_str());
				return false;
			}
			resolutions[vidx] = dit->second;
		}
	}

	return true;
}

double Robot::profileAction(
	const smpl::RobotState& parent, const smpl::RobotState& succ)
{
	double max_time = 0;
	for(int jidx = 0; jidx < m_rm->jointVariableCount(); jidx++)
	{
		auto from_pos = parent[jidx];
		auto to_pos = succ[jidx];
		auto vel = m_rm->velLimit(jidx) * R_SPEED;
		if (vel <= 0.0) {
			continue;
		}
		auto t = 0.0;
		if (m_rm->isContinuous(jidx)) {
			t = smpl::angles::shortest_angle_dist(from_pos, to_pos) / vel;
		} else {
			t = std::fabs(to_pos - from_pos) / vel;
		}

		max_time = std::max(max_time, t);
	}
	return max_time;
}

void Robot::initOccupancyGrids()
{
	double sx, sy, sz, ox, oy, oz, max_distance;

	m_ph.getParam("occupancy_grid/size_x", sx);
	m_ph.getParam("occupancy_grid/size_y", sy);
	m_ph.getParam("occupancy_grid/size_z", sz);
	m_ph.getParam("occupancy_grid/origin_x", ox);
	m_ph.getParam("occupancy_grid/origin_y", oy);
	m_ph.getParam("occupancy_grid/origin_z", oz);
	m_ph.getParam("occupancy_grid/max_dist", max_distance);
	m_ph.getParam("planning_frame", m_planning_frame);

	m_df_i = std::make_shared<smpl::EuclidDistanceMap>(
			ox, oy, oz,
			sx, sy, sz,
			DF_RES,
			max_distance);
	m_df_m = std::make_shared<smpl::EuclidDistanceMap>(
			ox, oy, oz,
			sx, sy, sz,
			DF_RES,
			max_distance);

	bool ref_counted = false;
	m_grid_i = std::make_unique<smpl::OccupancyGrid>(m_df_i, ref_counted);
	m_grid_i->setReferenceFrame(m_planning_frame);
	m_grid_m = std::make_unique<smpl::OccupancyGrid>(m_df_m, ref_counted);
	m_grid_m->setReferenceFrame(m_planning_frame);
	// SV_SHOW_INFO(m_grid_i->getBoundingBoxVisualization());

	bool propagate_negative_distances = true;
	m_df_ngr = std::make_shared<smpl::PropagationDistanceField>(
			ox, oy, oz,
			sx, sy, sz,
			DF_RES,
			max_distance, propagate_negative_distances);

	m_grid_ngr = std::make_unique<smpl::OccupancyGrid>(m_df_ngr, ref_counted);
	m_grid_ngr->setReferenceFrame(m_planning_frame);

	m_ngr_res = m_grid_ngr->resolution();
	m_ngr_origin = Eigen::Vector3d(
			m_grid_ngr->originX(), m_grid_ngr->originY(), m_grid_ngr->originZ());

	m_ngr_gmin = Eigen::Vector3d(
			m_grid_ngr->originX(), m_grid_ngr->originY(), m_grid_ngr->originZ());

	m_ngr_gmax = Eigen::Vector3d(
			m_grid_ngr->originX() + m_grid_ngr->sizeX(),
			m_grid_ngr->originY() + m_grid_ngr->sizeY(),
			m_grid_ngr->originZ() + m_grid_ngr->sizeZ());
}

bool Robot::initCollisionChecker()
{
	smpl::collision::CollisionModelConfig cc_conf;
	if (!smpl::collision::CollisionModelConfig::Load(m_ph, cc_conf)) {
		ROS_ERROR("Failed to load Collision Model Config");
		return false;
	}

	// immovable obstacle collision checker
	m_cc_i = std::make_unique<smpl::collision::CollisionSpace>();
	if (!m_cc_i->init(
			m_grid_i.get(),
			m_robot_description,
			cc_conf,
			m_robot_config.group_name,
			m_robot_config.planning_joints))
	{
		ROS_ERROR("Failed to initialize immovable Collision Space");
		return false;
	}

	if (m_cc_i->robotCollisionModel()->name() == "pr2") {
		// sbpl_collision_checking/types.h
		smpl::collision::AllowedCollisionMatrix acm;
		for (auto& pair : PR2AllowedCollisionPairs) {
			acm.setEntry(pair.first, pair.second, true);
		}
		m_cc_i->setAllowedCollisionMatrix(acm);
	}

	// movable objecet collision checker
	m_cc_m = std::make_unique<smpl::collision::CollisionSpace>();
	if (!m_cc_m->init(
			m_grid_m.get(),
			m_robot_description,
			cc_conf,
			m_robot_config.group_name,
			m_robot_config.planning_joints))
	{
		ROS_ERROR("Failed to initialize movable Collision Space");
		return false;
	}

	if (m_cc_m->robotCollisionModel()->name() == "pr2") {
		// sbpl_collision_checking/types.h
		smpl::collision::AllowedCollisionMatrix acm;
		for (auto& pair : PR2AllowedCollisionPairs) {
			acm.setEntry(pair.first, pair.second, true);
		}
		m_cc_m->setAllowedCollisionMatrix(acm);
	}

	return true;
}

bool Robot::getCollisionObjectMsg(
			const Object& object,
			moveit_msgs::CollisionObject& obj_msg,
			bool remove)
{
	object.GetMoveitObj(obj_msg);
	obj_msg.operation = remove ? moveit_msgs::CollisionObject::REMOVE :
										moveit_msgs::CollisionObject::ADD;

	if (remove) {
		return true;
	}

	obj_msg.header.frame_id = m_planning_frame;
	obj_msg.header.stamp = ros::Time::now();

	return true;
}

/// \brief Process a collision object
/// \param object The collision object to be processed
/// \return true if the object was processed successfully; false otherwise
bool Robot::processCollisionObjectMsg(
	const moveit_msgs::CollisionObject& object,
	bool movable)
{
	if (object.operation == moveit_msgs::CollisionObject::ADD) {
		return addCollisionObjectMsg(object, movable);
	}
	else if (object.operation == moveit_msgs::CollisionObject::REMOVE) {
		return removeCollisionObjectMsg(object, movable);
	}
	// else if (object.operation == moveit_msgs::CollisionObject::APPEND) {
	// 	return AppendCollisionObjectMsg(object);
	// }
	// else if (object.operation == moveit_msgs::CollisionObject::MOVE) {
	// 	return MoveCollisionObjectMsg(object);
	// }
	else {
		return false;
	}
}

bool Robot::processSTLMesh(
	const Object& object, bool remove, bool movable)
{
	auto cc = movable ? m_cc_m.get() : m_cc_i.get();
	if (remove)
	{
		// find the collision object with this name
		auto* _object = findCollisionObject(std::to_string(object.desc.id), movable);
		if (!_object) {
			return false;
		}

		// remove from collision space
		if (!cc->removeObject(_object)) {
			return false;
		}

		// remove all collision shapes belonging to this object
		auto belongs_to_object = [_object](const std::unique_ptr<smpl::collision::CollisionShape>& shape) {
			auto is_shape = [&shape](const smpl::collision::CollisionShape* s) {
				return s == shape.get();
			};
			auto it = std::find_if(
					begin(_object->shapes), end(_object->shapes), is_shape);
			return it != end(_object->shapes);
		};

		auto rit = std::remove_if(
				begin(cc->m_collision_shapes), end(cc->m_collision_shapes),
				belongs_to_object);
		cc->m_collision_shapes.erase(rit, end(cc->m_collision_shapes));

		// remove the object itself
		auto is_object = [_object](const std::unique_ptr<smpl::collision::CollisionObject>& obj) {
			return obj.get() == _object;
		};
		auto rrit = std::remove_if(
				begin(cc->m_collision_objects), end(cc->m_collision_objects), is_object);
		cc->m_collision_objects.erase(rrit, end(cc->m_collision_objects));

		return true;
	}

	moveit_msgs::CollisionObject obj_msg;
	object.GetMoveitObj(obj_msg);
	auto object_mesh = obj_msg.meshes[0];

	auto vertices = new double[3 * object_mesh.vertices.size()];
	auto triangles = new std::uint32_t[3 * object_mesh.triangles.size()];

	for (size_t i = 0; i < object_mesh.vertices.size(); ++i) {
		vertices[3 * i + 0] = object_mesh.vertices[i].x;
		vertices[3 * i + 1] = object_mesh.vertices[i].y;
		vertices[3 * i + 2] = object_mesh.vertices[i].z;
	}

	for (size_t i = 0; i < object_mesh.triangles.size(); ++i) {
		triangles[3 * i + 0] = object_mesh.triangles[i].vertex_indices[0];
		triangles[3 * i + 1] = object_mesh.triangles[i].vertex_indices[1];
		triangles[3 * i + 2] = object_mesh.triangles[i].vertex_indices[2];
	}

	smpl::collision::MeshShape* shape = new smpl::collision::MeshShape;
	shape->vertices = vertices;
	shape->triangles = triangles;
	shape->vertex_count = object_mesh.vertices.size();
	shape->triangle_count = object_mesh.triangles.size();

	Eigen::Affine3d transform;
	tf::poseMsgToEigen(obj_msg.mesh_poses[0], transform);

	std::vector<smpl::collision::CollisionShape*> shapes;
	smpl::collision::AlignedVector<Eigen::Affine3d> shape_poses;

	shapes.push_back(shape);
	shape_poses.push_back(transform);

	auto co = smpl::make_unique<smpl::collision::CollisionObject>();
	co->id = obj_msg.id;
	co->shapes = std::move(shapes);
	co->shape_poses = std::move(shape_poses);

	cc->m_collision_objects.push_back(std::move(co));
	return cc->insertObject(cc->m_collision_objects.back().get());
}

bool Robot::addCollisionObjectMsg(
	const moveit_msgs::CollisionObject& object,
	bool movable)
{
	auto cc = movable ? m_cc_m.get() : m_cc_i.get();

	if (cc->worldCollisionModel()->hasObjectWithName(object.id))
	{
		// find the collision object with this name
		auto* cc_object = findCollisionObject(object.id, movable);
		if (!cc_object) {
			return false;
		}

		Eigen::Affine3d pose;
		if (!object.primitives.empty()) {
			tf::poseMsgToEigen(object.primitive_poses[0], pose);
		}
		else if (!object.meshes.empty()) {
			tf::poseMsgToEigen(object.mesh_poses[0], pose);
		}

		if (cc_object->shape_poses[0].translation() == pose.translation()
			&& cc_object->shape_poses[0].rotation() == pose.rotation()) {
			return true;
		}
		else
		{
			cc_object->shape_poses[0] = pose;
			return cc->moveShapes(cc_object);
		}
	}

	if (object.header.frame_id != cc->getReferenceFrame()) {
		ROS_ERROR("Collision object must be specified in the Collision Space's reference frame (%s)", cc->getReferenceFrame().c_str());
		return false;
	}

	if (!checkCollisionObjectSanity(object)) {
		return false;
	}

	std::vector<smpl::collision::CollisionShape*> shapes;
	smpl::collision::AlignedVector<Eigen::Affine3d> shape_poses;

	for (size_t i = 0; i < object.primitives.size(); ++i) {
		auto& prim = object.primitives[i];

		std::unique_ptr<smpl::collision::CollisionShape> shape;
		switch (prim.type) {
		case shape_msgs::SolidPrimitive::BOX:
			shape = smpl::make_unique<smpl::collision::BoxShape>(
					prim.dimensions[shape_msgs::SolidPrimitive::BOX_X],
					prim.dimensions[shape_msgs::SolidPrimitive::BOX_Y],
					prim.dimensions[shape_msgs::SolidPrimitive::BOX_Z]);
			break;
		case shape_msgs::SolidPrimitive::SPHERE:
			shape = smpl::make_unique<smpl::collision::SphereShape>(
					prim.dimensions[shape_msgs::SolidPrimitive::SPHERE_RADIUS]);
			break;
		case shape_msgs::SolidPrimitive::CYLINDER:
			shape = smpl::make_unique<smpl::collision::CylinderShape>(
					prim.dimensions[shape_msgs::SolidPrimitive::CYLINDER_RADIUS],
					prim.dimensions[shape_msgs::SolidPrimitive::CYLINDER_HEIGHT]);
			break;
		case shape_msgs::SolidPrimitive::CONE:
			shape = smpl::make_unique<smpl::collision::ConeShape>(
					prim.dimensions[shape_msgs::SolidPrimitive::CONE_RADIUS],
					prim.dimensions[shape_msgs::SolidPrimitive::CONE_HEIGHT]);
			break;
		default:
			assert(0);
		}

		cc->m_collision_shapes.push_back(std::move(shape));
		shapes.push_back(cc->m_collision_shapes.back().get());

		auto& prim_pose = object.primitive_poses[i];
		Eigen::Affine3d transform;
		tf::poseMsgToEigen(prim_pose, transform);
		shape_poses.push_back(transform);
	}

	for (size_t i = 0; i < object.meshes.size(); ++i) {
		auto& mesh = object.meshes[i];

		assert(0); // TODO: implement

		auto& mesh_pose = object.mesh_poses[i];
		Eigen::Affine3d transform;
		tf::poseMsgToEigen(mesh_pose, transform);
		shape_poses.push_back(transform);
	}

	// create the collision object
	auto co = smpl::make_unique<smpl::collision::CollisionObject>();
	co->id = object.id;
	co->shapes = std::move(shapes);
	co->shape_poses = std::move(shape_poses);

	cc->m_collision_objects.push_back(std::move(co));
	return cc->insertObject(cc->m_collision_objects.back().get());
}

bool Robot::removeCollisionObjectMsg(
	const moveit_msgs::CollisionObject& object,
	bool movable)
{
	auto cc = movable ? m_cc_m.get() : m_cc_i.get();

	// find the collision object with this name
	auto* _object = findCollisionObject(object.id, movable);
	if (!_object) {
		return false;
	}

	// remove from collision space
	if (!cc->removeObject(_object)) {
		return false;
	}

	// remove all collision shapes belonging to this object
	auto belongs_to_object = [_object](const std::unique_ptr<smpl::collision::CollisionShape>& shape) {
		auto is_shape = [&shape](const smpl::collision::CollisionShape* s) {
			return s == shape.get();
		};
		auto it = std::find_if(
				begin(_object->shapes), end(_object->shapes), is_shape);
		return it != end(_object->shapes);
	};

	auto rit = std::remove_if(
			begin(cc->m_collision_shapes), end(cc->m_collision_shapes),
			belongs_to_object);
	cc->m_collision_shapes.erase(rit, end(cc->m_collision_shapes));

	// remove the object itself
	auto is_object = [_object](const std::unique_ptr<smpl::collision::CollisionObject>& object) {
		return object.get() == _object;
	};
	auto rrit = std::remove_if(
			begin(cc->m_collision_objects), end(cc->m_collision_objects), is_object);
	cc->m_collision_objects.erase(rrit, end(cc->m_collision_objects));

	return true;
}

bool Robot::checkCollisionObjectSanity(
	const moveit_msgs::CollisionObject& object) const
{
	if (object.primitives.size() != object.primitive_poses.size()) {
		ROS_ERROR("Mismatched sizes of primitives and primitive poses");
		return false;
	}

	if (object.meshes.size() != object.mesh_poses.size()) {
		ROS_ERROR("Mismatches sizes of meshes and mesh poses");
		return false;
	}

	// check solid primitive for correct format
	for (auto& prim : object.primitives) {
		switch (prim.type) {
		case shape_msgs::SolidPrimitive::BOX:
		{
			if (prim.dimensions.size() != 3) {
				ROS_ERROR("Invalid number of dimensions for box of collision object '%s' (Expected: %d, Actual: %zu)", object.id.c_str(), 3, prim.dimensions.size());
				return false;
			}
		}   break;
		case shape_msgs::SolidPrimitive::SPHERE:
		{
			if (prim.dimensions.size() != 1) {
				ROS_ERROR("Invalid number of dimensions for sphere of collision object '%s' (Expected: %d, Actual: %zu)", object.id.c_str(), 1, prim.dimensions.size());
				return false;
			}
		}   break;
		case shape_msgs::SolidPrimitive::CYLINDER:
		{
			if (prim.dimensions.size() != 2) {
				ROS_ERROR("Invalid number of dimensions for cylinder of collision object '%s' (Expected: %d, Actual: %zu)", object.id.c_str(), 2, prim.dimensions.size());
				return false;
			}
		}   break;
		case shape_msgs::SolidPrimitive::CONE:
		{
			if (prim.dimensions.size() != 2) {
				ROS_ERROR("Invalid number of dimensions for cone of collision object '%s' (Expected: %d, Actual: %zu)", object.id.c_str(), 2, prim.dimensions.size());
				return false;
			}
		}   break;
		default:
			ROS_ERROR("Unrecognized SolidPrimitive type");
			return false;
		}
	}

	return true;
}

auto Robot::findCollisionObject(const std::string& id, bool movable) const
	-> smpl::collision::CollisionObject*
{
	auto cc = movable ? m_cc_m.get() : m_cc_i.get();

	for (auto& object : cc->m_collision_objects) {
		if (object->id == id) {
			return object.get();
		}
	}
	return nullptr;
}

bool Robot::setCollisionRobotState()
{
	if (m_start_state.joint_state.name.size() != m_start_state.joint_state.position.size()) {
		ROS_ERROR("Joint state contains mismatched number of joint names and joint positions");
		return false;
	}

	for (size_t i = 0; i < m_start_state.joint_state.name.size(); ++i)
	{
		auto& joint_name = m_start_state.joint_state.name[i];
		double joint_position = m_start_state.joint_state.position[i];
		if (!m_cc_i->setJointPosition(joint_name, joint_position)) {
			ROS_ERROR("Failed to set position of joint '%s' to %f", joint_name.c_str(), joint_position);
			return false;
		}
		if (!m_cc_m->setJointPosition(joint_name, joint_position)) {
			ROS_ERROR("Failed to set position of joint '%s' to %f", joint_name.c_str(), joint_position);
			return false;
		}
	}

	auto& multi_dof_joint_state = m_start_state.multi_dof_joint_state;
	for (size_t i = 0; i < multi_dof_joint_state.joint_names.size(); ++i) {
		auto& joint_name = multi_dof_joint_state.joint_names[i];
		auto& joint_transform = multi_dof_joint_state.transforms[i];
		// TODO: Need a way to identify what type of joint this is to extract
		// its condensed set of variables. Might be worth adding a function to
		// CollisionSpace to handle setting any joint from its local transform.
	}

	// TODO: world -> model transform should be handled by multi_dof joint
	// state.

	// // TODO: ProcessAttachedCollisionObject will need to be copied from CollisionSpaceScene
	// auto& attached_collision_objects = m_start_state.attached_collision_objects;
	// for (auto& aco : attached_collision_objects) {
	// 	if (!ProcessAttachedCollisionObject(aco)) {
	// 		ROS_WARN_NAMED(LOG, "Failed to process attached collision object");
	// 		return false;
	// 	}
	// }

	return true;
}

bool Robot::initPlanner()
{
	if (!readPlannerConfig(ros::NodeHandle("~planning"))) {
		ROS_ERROR("Failed to read planner config");
		return false;
	}

	m_planning_params.addParam("discretization", m_planning_config.discretization);
	m_planning_params.addParam("mprim_filename", m_planning_config.mprim_filename);
	m_planning_params.addParam("use_xyz_snap_mprim", m_planning_config.use_xyz_snap_mprim);
	m_planning_params.addParam("use_rpy_snap_mprim", m_planning_config.use_rpy_snap_mprim);
	m_planning_params.addParam("use_xyzrpy_snap_mprim", m_planning_config.use_xyzrpy_snap_mprim);
	m_planning_params.addParam("use_short_dist_mprims", m_planning_config.use_short_dist_mprims);
	m_planning_params.addParam("xyz_snap_dist_thresh", m_planning_config.xyz_snap_dist_thresh);
	m_planning_params.addParam("rpy_snap_dist_thresh", m_planning_config.rpy_snap_dist_thresh);
	m_planning_params.addParam("xyzrpy_snap_dist_thresh", m_planning_config.xyzrpy_snap_dist_thresh);
	m_planning_params.addParam("short_dist_mprims_thresh", m_planning_config.short_dist_mprims_thresh);
	// m_planning_params.addParam("epsilon", 200.0); Epsilon used for ARAStar
	m_planning_params.addParam("return_first_solution", false);
	m_planning_params.addParam("epsilon", 100.0);
	m_planning_params.addParam("search_mode", false);
	m_planning_params.addParam("allow_partial_solutions", false);
	m_planning_params.addParam("target_epsilon", 1.0);
	m_planning_params.addParam("delta_epsilon", 1.0);
	m_planning_params.addParam("improve_solution", false);
	m_planning_params.addParam("bound_expansions", true);
	m_planning_params.addParam("repair_time", 1.0);
	m_planning_params.addParam("bfs_inflation_radius", 0.02);
	m_planning_params.addParam("bfs_cost_per_cell", 100);
	m_ph.getParam("robot/interpolate", m_planning_params.interpolate_path);
	m_ph.getParam("robot/shortcut", m_planning_params.shortcut_path);

	if (!createPlanner(m_planning_params.interpolate_path))
	{
		ROS_ERROR("Failed to create planner.");
		return false;
	}

	m_planner_init = true;
}

bool Robot::readPlannerConfig(const ros::NodeHandle &nh)
{
	if (!nh.getParam("discretization", m_planning_config.discretization)) {
		ROS_ERROR("Failed to read 'discretization' from the param server");
		return false;
	}

	if (!nh.getParam("mprim_filename", m_planning_config.mprim_filename)) {
		ROS_ERROR("Failed to read param 'mprim_filename' from the param server");
		return false;
	}

	if (!nh.getParam("use_xyz_snap_mprim", m_planning_config.use_xyz_snap_mprim)) {
		ROS_ERROR("Failed to read param 'use_xyz_snap_mprim' from the param server");
		return false;
	}

	if (!nh.getParam("use_rpy_snap_mprim", m_planning_config.use_rpy_snap_mprim)) {
		ROS_ERROR("Failed to read param 'use_rpy_snap_mprim' from the param server");
		return false;
	}

	if (!nh.getParam("use_xyzrpy_snap_mprim", m_planning_config.use_xyzrpy_snap_mprim)) {
		ROS_ERROR("Failed to read param 'use_xyzrpy_snap_mprim' from the param server");
		return false;
	}

	if (!nh.getParam("use_short_dist_mprims", m_planning_config.use_short_dist_mprims)) {
		ROS_ERROR("Failed to read param 'use_short_dist_mprims' from the param server");
		return false;
	}

	if (!nh.getParam("xyz_snap_dist_thresh", m_planning_config.xyz_snap_dist_thresh)) {
		ROS_ERROR("Failed to read param 'xyz_snap_dist_thresh' from the param server");
		return false;
	}

	if (!nh.getParam("rpy_snap_dist_thresh", m_planning_config.rpy_snap_dist_thresh)) {
		ROS_ERROR("Failed to read param 'rpy_snap_dist_thresh' from the param server");
		return false;
	}

	if (!nh.getParam("xyzrpy_snap_dist_thresh", m_planning_config.xyzrpy_snap_dist_thresh)) {
		ROS_ERROR("Failed to read param 'xyzrpy_snap_dist_thresh' from the param server");
		return false;
	}

	if (!nh.getParam("short_dist_mprims_thresh", m_planning_config.short_dist_mprims_thresh)) {
		ROS_ERROR("Failed to read param 'short_dist_mprims_thresh' from the param server");
		return false;
	}

	return true;
}

bool Robot::createPlanner(bool interp)
{
	m_planner = std::make_unique<smpl::PlannerInterface>(
										m_rm.get(), m_cc_i.get(), m_grid_i.get(), m_cc_m.get());
	m_planning_params.interpolate_path = interp;

	if (!m_planner->init(m_planning_params)) {
		ROS_ERROR("Failed to initialize Planner Interface");
		return false;
	}

	return true;
}

void Robot::createMultiPoseGoalConstraint(
	const std::vector<Eigen::Affine3d> &poses,
	moveit_msgs::MotionPlanRequest& req)
{
	req.goal_constraints.clear();
	req.goal_constraints.resize(poses.size());
	for (size_t g = 0; g < poses.size(); ++g)
	{
		createPoseGoalConstraint(poses[g]);
		m_goal.orientation_constraints[0].absolute_x_axis_tolerance = DEG5;
		req.goal_constraints[g] = m_goal;
	}
}

void Robot::createPoseGoalConstraint(
	const Eigen::Affine3d& pose)
{
	m_goal.position_constraints.resize(1);
	m_goal.orientation_constraints.resize(1);
	m_goal.position_constraints[0].header.frame_id = m_planning_frame;

	m_goal.position_constraints[0].constraint_region.primitives.resize(1);
	m_goal.position_constraints[0].constraint_region.primitive_poses.resize(1);
	m_goal.position_constraints[0].constraint_region.primitives[0].type = shape_msgs::SolidPrimitive::BOX;
	m_goal.position_constraints[0].constraint_region.primitive_poses[0].position.x = pose.translation().x();
	m_goal.position_constraints[0].constraint_region.primitive_poses[0].position.y = pose.translation().y();
	m_goal.position_constraints[0].constraint_region.primitive_poses[0].position.z = pose.translation().z();

	Eigen::Quaterniond q(pose.rotation());
	tf::quaternionEigenToMsg(q, m_goal.orientation_constraints[0].orientation);

	// set tolerances
	m_goal.position_constraints[0].constraint_region.primitives[0].dimensions.resize(3, 0.05);
	m_goal.orientation_constraints[0].absolute_x_axis_tolerance = 2.0 * M_PI;
	m_goal.orientation_constraints[0].absolute_y_axis_tolerance = 2.0 * M_PI; // these do not make a difference inside smpl
	m_goal.orientation_constraints[0].absolute_z_axis_tolerance = 2.0 * M_PI; // these do not make a difference inside smpl
}

void Robot::createJointSpaceGoal(
	const smpl::RobotState& pose, moveit_msgs::MotionPlanRequest& req)
{
	req.goal_constraints.clear();
	req.goal_constraints.resize(1);

	req.goal_constraints[0].joint_constraints.resize(m_rm->jointVariableCount());
	for (int jidx = 0; jidx < m_rm->jointVariableCount(); ++jidx) {
		req.goal_constraints[0].joint_constraints[jidx].joint_name = m_robot_config.planning_joints.at(jidx);
		req.goal_constraints[0].joint_constraints[jidx].position = pose.at(jidx);
		req.goal_constraints[0].joint_constraints[jidx].tolerance_above = 0.0174533; // 1 degree
		req.goal_constraints[0].joint_constraints[jidx].tolerance_below = 0.0174533; // 1 degree
		req.goal_constraints[0].joint_constraints[jidx].weight = 1.0;
	}
}

void Robot::addPathConstraint(moveit_msgs::Constraints& path_constraints)
{
	path_constraints.position_constraints.resize(1);
	path_constraints.orientation_constraints.resize(1);
	path_constraints.position_constraints[0].header.frame_id = m_planning_frame;

	path_constraints.position_constraints[0].constraint_region.primitives.resize(1);
	path_constraints.position_constraints[0].constraint_region.primitive_poses.resize(1);
	path_constraints.position_constraints[0].constraint_region.primitives[0].type = shape_msgs::SolidPrimitive::BOX;
	path_constraints.position_constraints[0].constraint_region.primitive_poses[0].position.x = 0.0;
	path_constraints.position_constraints[0].constraint_region.primitive_poses[0].position.y = 0.0;
	path_constraints.position_constraints[0].constraint_region.primitive_poses[0].position.z = 0.0;

	Eigen::Quaterniond q;
	smpl::angles::from_euler_zyx(0.0, 0.0, 0.0, q);
	tf::quaternionEigenToMsg(q, path_constraints.orientation_constraints[0].orientation);

	// set tolerances
	path_constraints.position_constraints[0].constraint_region.primitives[0].dimensions.resize(3, 0.0);
	path_constraints.orientation_constraints[0].absolute_x_axis_tolerance = DEG5;
	path_constraints.orientation_constraints[0].absolute_y_axis_tolerance = DEG5;
	path_constraints.orientation_constraints[0].absolute_z_axis_tolerance = 2.0 * M_PI;
}

bool Robot::getStateNearPose(
	const Eigen::Affine3d& pose,
	const smpl::RobotState& seed_state,
	smpl::RobotState& state,
	int N,
	const std::string& ns)
{
	state.clear();
	smpl::RobotState seed;

	if (!seed_state.empty()) {
		seed = seed_state;
	}
	else {
		GetRandomState(seed);
	}

	int tries = 0;
	do
	{
		// if (!ns.empty())
		// {
		// 	auto markers = m_cc_i->getCollisionModelVisualization(seed);
		// 	for (auto& marker : markers) {
		// 		marker.ns = ns + "_seed";
		// 	}
		// 	SV_SHOW_INFO_NAMED(ns + "_seed", markers);

		// 	SV_SHOW_INFO_NAMED(ns + "_pose", smpl::visual::MakePoseMarkers(
		// 		pose, m_grid_i->getReferenceFrame(), ns + "_pose"));
		// }

		if (m_rm->computeIKSearch(pose, seed, state))
		{
			if (m_rm->checkJointLimits(state) && m_cc_i->isStateValid(state)) {
				break;
			}
		}
		GetRandomState(seed);
		state.clear();
		++tries;
	}
	while (tries < N);

	return tries < N;
}


void Robot::displayObjectMarker(const Object& object)
{
	auto marker = object.GetMarker();
	m_vis_pub.publish(marker);
}

void Robot::RunManipulabilityStudy(int N)
{
	double ox, oy, oz, sx, sy, sz;
	m_sim->GetShelfParams(ox, oy, oz, sx, sy, sz);

	moveit_msgs::RobotState start = m_start_state;
	trajectory_msgs::JointTrajectory path;
	Eigen::MatrixXd J, Jprod;

	std::string filename(__FILE__);
	auto found = filename.find_last_of("/\\");
	filename = filename.substr(0, found + 1) + "../../dat/MANIPULABILITY_3cm.csv";

	bool exists = FileExists(filename);
	std::ofstream DATA;
	DATA.open(filename, std::ofstream::out | std::ofstream::app);
	if (!exists)
	{
		DATA << "x,y,successes,yoshikawa,togai\n";
	}

	for (double x0 = ox; x0 <= ox + sx; x0 += RES)
	{
		for (double y0 = oy; y0 <= oy + sy; y0 += RES)
		{
			double yoshikawa = std::numeric_limits<double>::lowest();
			double togai = std::numeric_limits<double>::lowest();
			int plans = 0;
			for (int t = 0; t < N; ++t)
			{
				// goal pose
				Eigen::Affine3d goal_pose = Eigen::Translation3d(x0, y0, m_table_z + 0.03) *
					Eigen::AngleAxisd(m_distD(m_rng) * M_PI * 2, Eigen::Vector3d::UnitZ()) *
					Eigen::AngleAxisd(m_distD(m_rng) * M_PI * 2, Eigen::Vector3d::UnitY()) *
					Eigen::AngleAxisd(m_distD(m_rng) * M_PI * 2, Eigen::Vector3d::UnitX());
				SV_SHOW_INFO_NAMED("study_goal_pose", smpl::visual::MakePoseMarkers(
					goal_pose, m_grid_i->getReferenceFrame(), "study_goal_pose"));

				if (planToPoseGoal(start, { goal_pose }, path, 0.5))
				{
					++plans;

					smpl::RobotState end_state = path.points.back().positions;
					m_rm->computeJacobian(end_state, J);

					Jprod = J * J.transpose();
					double w1 = std::sqrt(Jprod.determinant());

					Eigen::JacobiSVD<Eigen::MatrixXd> svd(J);
					double w2 = svd.singularValues()(svd.singularValues().size()-1)
    								/ svd.singularValues()(0);

					if (w1 > yoshikawa) {
						yoshikawa = w1;
					}
					if (w2 > togai) {
						togai = w2;
					}
				}
			}

			double success = 100.0 * (plans/(double)N);
			DATA << x0 << ',' << y0 << ',' << success << ',' << yoshikawa << ',' << togai << '\n';
		}
	}

	DATA.close();
}

void Robot::RunPushIKStudy(int N)
{
	double ox, oy, oz, sx, sy, sz;
	m_sim->GetShelfParams(ox, oy, oz, sx, sy, sz);

	moveit_msgs::RobotState start = m_start_state;
	Eigen::Affine3d push_start_pose, push_end_pose;
	trajectory_msgs::JointTrajectory path;

	std::string filename(__FILE__);
	auto found = filename.find_last_of("/\\");
	filename = filename.substr(0, found + 1) + "../../dat/PUSHES_IK_3cm.csv";

	bool exists = FileExists(filename);
	std::ofstream DATA;
	DATA.open(filename, std::ofstream::out | std::ofstream::app);
	if (!exists)
	{
		DATA << "x0,y0,yaw0,x1,y1,yaw1,result\n";
	}

	for (double x0 = ox - 0.05; x0 <= ox + sx; x0 += RES)
	{
		for (double y0 = oy - 0.05; y0 <= oy + sy; y0 += RES)
		{
			// 1. random ee yaw at push start pose
			double yaw0 = m_distD(m_rng) * M_PI * 2;
			yaw0 = smpl::angles::normalize_angle(yaw0);

			// 2. push start pose
			push_start_pose = Eigen::Translation3d(x0, y0, m_table_z + 0.03) *
				Eigen::AngleAxisd(yaw0, Eigen::Vector3d::UnitZ()) *
				Eigen::AngleAxisd(M_PI/6.0, Eigen::Vector3d::UnitY()) *
				Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());
			SV_SHOW_INFO_NAMED("push_start_pose", smpl::visual::MakePoseMarkers(
				push_start_pose, m_grid_i->getReferenceFrame(), "push_start_pose"));

			// 3. verify push start pose
			if ((m_grid_i->getDistanceFromPoint(x0, y0, m_table_z + 0.03) <= m_grid_i->resolution())
					|| (!planToPoseGoal(start, { push_start_pose }, path, 2.0)))
			{
				DATA << x0 << ',' << y0 << ',' << yaw0 << ','
						<< -1 << ',' << -1 << ',' << -1 << ',' << 5 << '\n';
				continue;
			}

			auto found_pose = m_rm->computeFK(path.points.back().positions);
			double yaw1, pitch, roll;
			smpl::angles::get_euler_zyx(push_end_pose.rotation(), yaw1, pitch, roll);

			// 4. generate N random push end pose samples to try
			for (int t = 0; t < N; ++t)
			{
				// 5. sample push end pose (x, y) coordinates
				double x1, y1, dist;
				x1 = ox + 0.01 + (m_distD(m_rng) * (sx - 0.02));
				y1 = oy + 0.01 + (m_distD(m_rng) * (sy - 0.02));
				dist = std::sqrt(std::pow(x0 - x1, 2) + std::pow(y0 - y1, 2));

				// 6. set push end pose
				push_end_pose = found_pose;
				push_end_pose.translation().x() = x1;
				push_end_pose.translation().y() = y1;
				SV_SHOW_INFO_NAMED("push_end_pose", smpl::visual::MakePoseMarkers(
					push_end_pose, m_grid_i->getReferenceFrame(), "push_end_pose"));

				// 7. get push action trajectory via inverse kinematics
				trajectory_msgs::JointTrajectory push_action;
				int failure = computePushPoint(
								path.points.back().time_from_start.toSec(),
								path.points.back().positions,
								push_end_pose,
								push_action);
				DATA << x0 << ',' << y0 << ',' << yaw0 << ','
					<< x1 << ',' << y1 << ',' << yaw1 << ',' << failure << '\n';
			}
		}
	}

	DATA.close();
}

void Robot::VisPlane(double z)
{
	double ox, oy, oz, sx, sy, sz;
	m_sim->GetShelfParams(ox, oy, oz, sx, sy, sz);

	visualization_msgs::Marker marker;
	marker.header.frame_id = "base_footprint";
	marker.header.stamp = ros::Time();
	marker.ns = "constraint_plane";
	marker.id = 0;
	marker.action = visualization_msgs::Marker::ADD;
	marker.type = visualization_msgs::Marker::CUBE;

	marker.pose.position.x = ox + sx/2.0;
	marker.pose.position.y = oy + sy/2.0;
	marker.pose.position.z = z;

	marker.scale.x = 2.0 * sx;
	marker.scale.y = 2.0 * sy;
	marker.scale.z = 0.01;

	marker.color.a = 0.5; // Don't forget to set the alpha!
	marker.color.r = 0.86;
	marker.color.g = 0.34;
	marker.color.b = 0.16;

	m_vis_pub.publish(marker);
}

bool Robot::SavePushDebugData(int scene_id)
{
	std::string filename(__FILE__);
	auto found = filename.find_last_of("/\\");
	filename = filename.substr(0, found + 1) + "../../dat/PUSH_DEBUG_DATA.csv";

	bool exists = FileExists(filename);
	std::ofstream STATS;
	STATS.open(filename, std::ofstream::out | std::ofstream::app);
	if (!exists)
	{
		STATS << "UID,"
				<< "PlanPushCalls,PushStartPosesFound,PushActionsFoundToSim,"
				<< "PushPlanTime,PushSimTime,PushSimSuccesses,"
				<< "PushDBLookupTime,PushesFoundInDB,PushDBSuccesses,PushDBFails,PushDBTotalTime,"
				<< "PushSuccessInSim,PushFailInSim,NoCollisionWithObject,"
				<< "IKDidNotReachEnd,IKObstacleCollision,IKJointLimitViolation,"
				<< "IKFail,PushStartPoseUnreachable,PushStartPoseInObstacle\n";
	}

	STATS << scene_id << ','
			<< m_stats["plan_push_calls"] << ','
			<< m_stats["push_start_poses_found"] << ','
			<< m_stats["push_actions_found"] << ','
			<< m_stats["push_plan_time"] << ','
			<< m_stats["push_sim_time"] << ','
			<< m_stats["push_sim_successes"] << ','
			<< m_stats["push_db_lookup_time"] << ','
			<< m_stats["push_found_in_db"] << ','
			<< m_stats["push_db_successes"] << ','
			<< m_stats["push_db_failures"] << ','
			<< m_stats["push_db_total_time"] << ','
			<< m_debug_push_info["sim_success"] << ','
			<< m_debug_push_info["sim_fail"] << ','
			<< m_debug_push_info["no_collision_ooi"] << ','
			<< m_debug_push_info["ik_ended_early"] << ','
			<< m_debug_push_info["obstacle_collision"] << ','
			<< m_debug_push_info["ik_joint_limit"] << ','
			<< m_debug_push_info["ik_fail"] << ','
			<< m_debug_push_info["start_unreachable"] << ','
			<< m_debug_push_info["start_inside_obstacle"] << '\n';
	STATS.close();
}

int Robot::getPushIdx(double push_frac)
{
	if (push_frac == 1.0) {
		return 0;
	}
}

void Robot::addPushToDB(
	Object* o, const Coord &goal, const int& aidx,
	const comms::ObjectsPoses &init_scene,
	const comms::ObjectsPoses &result_scene,
	const std::vector<int> &relevant_ids,
	const trajectory_msgs::JointTrajectory &push_action)
{
	ContPose pose(o->desc.o_x, o->desc.o_y, o->desc.o_z, o->desc.o_roll, o->desc.o_pitch, o->desc.o_yaw);
	ObjectState ooi_state(o->desc.id, o->Symmetric(), pose);

	auto db_key = std::make_tuple(ooi_state, goal, aidx);
	auto db_value = std::make_tuple(init_scene, result_scene, relevant_ids, push_action);

	const auto it = m_valid_sims.find(db_key);
	if (it != m_valid_sims.end()) {
		it->second.push_back(db_value);
	}

	m_valid_sims[db_key] = { db_value };
}

bool Robot::lookupPushInDB(
	Object* o, const Coord &goal, const int& aidx, const comms::ObjectsPoses &curr_scene,
	comms::ObjectsPoses &new_scene,
	trajectory_msgs::JointTrajectory &new_action)
{
	ContPose pose(o->desc.o_x, o->desc.o_y, o->desc.o_z, o->desc.o_roll, o->desc.o_pitch, o->desc.o_yaw);
	ObjectState ooi_state(o->desc.id, o->Symmetric(), pose);

	auto db_key = std::make_tuple(ooi_state, goal, aidx);
	const auto it = m_valid_sims.find(db_key);
	if (it == m_valid_sims.end())
	{
		// have not seen this push before
		return false;
	}

	// 1. relevant objects should currently be in same poses as seen before
	// 2. irrelevant objects should not collide with the push action
	//		- add them to movable collision space
	for (const auto &push : it->second)
	{
		const auto &init_scene = std::get<0>(push);
		const auto &result_scene = std::get<1>(push);
		const auto &relevant_ids = std::get<2>(push);
		const auto &push_action = std::get<3>(push);

		std::vector<Object*> irrelevant_objs;
		bool consistent = true;
		for (size_t i = 0; i < init_scene.poses.size(); ++i)
		{
			if (std::find(relevant_ids.begin(), relevant_ids.end(), init_scene.poses.at(i).id) != relevant_ids.end())
			{
				DiscPose seen(ContPose(init_scene.poses.at(i).xyz[0], init_scene.poses.at(i).xyz[1], init_scene.poses.at(i).xyz[2], init_scene.poses.at(i).rpy[0], init_scene.poses.at(i).rpy[1], init_scene.poses.at(i).rpy[2]));
				DiscPose curr(ContPose(curr_scene.poses.at(i).xyz[0], curr_scene.poses.at(i).xyz[1], curr_scene.poses.at(i).xyz[2], curr_scene.poses.at(i).rpy[0], curr_scene.poses.at(i).rpy[1], curr_scene.poses.at(i).rpy[2]));
				if (seen != curr)
				{
					consistent = false;
					break;
				}
			}
			else
			{
				// m_movables[m_movable_map[curr_scene.poses.at(i).id]]->SetObjectPose(
				// 		curr_scene.poses.at(i).xyz, curr_scene.poses.at(i).rpy);
				irrelevant_objs.push_back(m_movables[m_movable_map[curr_scene.poses.at(i).id]]->GetObject());
			}
		}

		if (consistent)
		{
			ProcessObstacles(irrelevant_objs, false, true);
			for (const auto &push_state : push_action.points)
			{
				if (!m_cc_m->isStateValid(push_state.positions))
				{
					consistent = false;
					break;
				}
			}
			ProcessObstacles(irrelevant_objs, true, true);
		}

		if (consistent)
		{
			new_scene.poses.clear();
			for (size_t i = 0; i < result_scene.poses.size(); ++i)
			{
				if (std::find(relevant_ids.begin(), relevant_ids.end(), result_scene.poses.at(i).id) != relevant_ids.end()) {
					new_scene.poses.push_back(result_scene.poses.at(i));
				}
				else {
					new_scene.poses.push_back(curr_scene.poses.at(i));
				}
			}

			new_action = push_action;
			return true;
		}
	}
	return false;
}

void Robot::visMAPFPath(const Trajectory* path)
{
	visualization_msgs::Marker points, line_strip;
	points.header.frame_id = line_strip.header.frame_id = m_planning_frame;
	points.header.stamp = line_strip.header.stamp = ros::Time::now();
	points.ns = line_strip.ns = "mapf_path";
	points.action = line_strip.action = visualization_msgs::Marker::ADD;
	points.pose.orientation.w = line_strip.pose.orientation.w = 1.0;

	points.id = 0;
	line_strip.id = 1;

	points.type = visualization_msgs::Marker::POINTS;
	line_strip.type = visualization_msgs::Marker::LINE_STRIP;

	// POINTS markers use x and y scale for width/height respectively
	points.scale.x = 0.01;
	points.scale.y = 0.01;

	// LINE_STRIP markers use only the x component of scale, for the line width
	line_strip.scale.x = 0.01;

	// Points are cyan
	points.color.g = 1.0f;
	points.color.b = 1.0f;
	points.color.a = 1.0;

	// Line strip is blue
	line_strip.color.b = 1.0;
	line_strip.color.a = 1.0;

	// Create the vertices for the points and lines
	for (size_t i = 0; i < path->size(); ++i)
	{
		geometry_msgs::Point p;
		p.x = path->at(i).state[0];
		p.y = path->at(i).state[1];
		p.z = path->at(i).state[2];

		points.points.push_back(p);
		line_strip.points.push_back(p);
	}

	m_vis_pub.publish(points);
	m_vis_pub.publish(line_strip);
}

void Robot::visTrajectory(
	const trajectory_msgs::JointTrajectory &traj,
	const std::string &name)
{
	std::vector<smpl::visual::Marker> ma;

	auto cinc = 1.0f / float(traj.points.size());
	for (size_t i = 0; i < traj.points.size(); ++i) {
		auto markers = m_cc_i->getCollisionModelVisualization(traj.points[i].positions);

		for (auto& marker : markers) {
			auto r = 0.1f;
			auto g = cinc * (float)(traj.points.size() - (i + 1));
			auto b = cinc * (float)i;
			marker.color = smpl::visual::Color{ r, g, b, 1.0f };
		}

		for (auto& m : markers) {
			ma.push_back(std::move(m));
		}
	}

	for (size_t i = 0; i < ma.size(); ++i) {
		auto& marker = ma[i];
		marker.ns = name;
		marker.id = i;
	}
	SV_SHOW_INFO_NAMED(name, ma);
}

void Robot::visTrajectory(
	const Trajectory &traj,
	const std::string &name)
{
	std::vector<smpl::visual::Marker> ma;

	auto cinc = 1.0f / float(traj.size());
	for (size_t i = 0; i < traj.size(); ++i) {
		auto markers = m_cc_i->getCollisionModelVisualization(traj[i].state);

		for (auto& marker : markers) {
			auto r = 0.1f;
			auto g = cinc * (float)(traj.size() - (i + 1));
			auto b = cinc * (float)i;
			marker.color = smpl::visual::Color{ r, g, b, 1.0f };
		}

		for (auto& m : markers) {
			ma.push_back(std::move(m));
		}
	}

	for (size_t i = 0; i < ma.size(); ++i) {
		auto& marker = ma[i];
		marker.ns = name;
		marker.id = i;
	}
	SV_SHOW_INFO_NAMED(name, ma);
}

} // namespace clutter
