#include <pushplan/utils/controller.hpp>

#include <smpl/angles.h>

namespace clutter
{

RobotController::RobotController()
{
	// tell the action client that we want to spin a thread by default
	// traj_client_ = new TrajClient("torso_controller/joint_trajectory_action", true);
	m_r_arm_controller = std::make_unique<TrajClient>("r_arm_controller/joint_trajectory_action", true);
	m_l_arm_controller = std::make_unique<TrajClient>("l_arm_controller/joint_trajectory_action", true);
	m_torso_controller = std::make_unique<TrajClient>("torso_controller/joint_trajectory_action", true);
	m_gripper_controller = std::make_unique<GripperClient>("r_gripper_controller/gripper_action", true);
}

void RobotController::InitControllers()
{
	// wait for action server to come up
	while(!m_r_arm_controller->waitForServer(ros::Duration(2.0))) {
		continue;
		//	ROS_INFO("Waiting for the joint_trajectory_action server");
	}
	while(!m_l_arm_controller->waitForServer(ros::Duration(2.0))) {
		continue;
		//	ROS_INFO("Waiting for the joint_trajectory_action server");
	}
	while(!m_torso_controller->waitForServer(ros::Duration(2.0))) {
		ROS_INFO("Waiting for the joint_trajectory_action server");
		continue;
	}
}

void RobotController::RaiseTorso(const double& value)
{
	pr2_controllers_msgs::JointTrajectoryGoal goal;
	goal.trajectory.joint_names.push_back("torso_lift_joint");
	goal.trajectory.points.resize(2);
	for (int i = 0; i < 2; ++i)
	{
		goal.trajectory.points[i].positions.resize(1);
		goal.trajectory.points[i].velocities.resize(1);

		goal.trajectory.points[i].positions[0] = value/(2.0 - i);
		goal.trajectory.points[i].velocities[0] = 0.0;

		goal.trajectory.points[i].time_from_start = ros::Duration(1.0) * (1.0 + (4.0 * i));
	}

	goal.trajectory.header.stamp = ros::Time::now() + ros::Duration(1.0);
	m_torso_controller->sendGoal(goal);
}

void RobotController::MoveLeftArmToRest()
{
	pr2_controllers_msgs::JointTrajectoryGoal goal;
	goal.trajectory.joint_names.push_back("l_shoulder_pan_joint");
	goal.trajectory.joint_names.push_back("l_shoulder_lift_joint");
	goal.trajectory.joint_names.push_back("l_upper_arm_roll_joint");
	goal.trajectory.joint_names.push_back("l_elbow_flex_joint");
	goal.trajectory.joint_names.push_back("l_forearm_roll_joint");
	goal.trajectory.joint_names.push_back("l_wrist_flex_joint");
	goal.trajectory.joint_names.push_back("l_wrist_roll_joint");

	int joints = 7;
	goal.trajectory.points.resize(2);
	for (int i = 0; i < 2; ++i)
	{
		goal.trajectory.points[i].positions.resize(joints);
		goal.trajectory.points[i].velocities.resize(joints);
		for (int j = 0; j < joints; ++j)
		{
			goal.trajectory.points[i].positions[j] = m_REST_LEFT_ARM.at(j)/(2.0 - i);
			goal.trajectory.points[i].velocities[j] = 0.0;
		}
		goal.trajectory.points[i].time_from_start = ros::Duration(1.0) * (1.0 + (4.0 * i));
	}

	goal.trajectory.header.stamp = ros::Time::now() + ros::Duration(1.0);
	m_l_arm_controller->sendGoal(goal);
}

void RobotController::MoveToStartState(const sensor_msgs::JointState& start_state)
{
	pr2_controllers_msgs::JointTrajectoryGoal goal;
	goal.trajectory.joint_names.clear();

	int joints = 0;
	ROS_WARN("Read start state joint name!");
	for (size_t i = 0; i < start_state.name.size(); ++i)
	{
		if (start_state.name.at(i).compare("torso_lift_joint") == 0)
			continue;

		ROS_WARN("joint name %d = %s", i, start_state.name.at(i).c_str());

		goal.trajectory.joint_names.push_back(start_state.name.at(i));
		joints++;
	}

	ROS_WARN("Read start state joint positions!");
	goal.trajectory.points.resize(1);
	for (int i = 0; i < 1; ++i)
	{
		goal.trajectory.points[i].positions.resize(joints);
		goal.trajectory.points[i].velocities.resize(joints);

		int c = 0;
		for (size_t j = 0; j < start_state.name.size(); ++j)
		{
			if (start_state.name.at(j).compare("torso_lift_joint") == 0) {
				c++;
				continue;
			}

			ROS_WARN("joint position %d (wp %d) = %f", j, i, start_state.position.at(j));
			goal.trajectory.points[i].positions[j-c] = start_state.position.at(j)/(2.0 - i);
			goal.trajectory.points[i].velocities[j-c] = 0.0;
		}

		goal.trajectory.points[i].time_from_start = ros::Duration(1.0) * (1.0 + (4.0 * i));
	}

	goal.trajectory.header.stamp = ros::Time::now() + ros::Duration(1.0);
	m_r_arm_controller->sendGoal(goal);
}

void RobotController::ExecArmTrajectory(pr2_controllers_msgs::JointTrajectoryGoal goal)
{
	// When to start the trajectory: 1s from now
	goal.trajectory.header.stamp = ros::Time::now() + ros::Duration(1.0);
	m_r_arm_controller->sendGoal(goal);
}

pr2_controllers_msgs::JointTrajectoryGoal RobotController::PR2TrajFromMsg(const trajectory_msgs::JointTrajectory& trajectory)
{
	pr2_controllers_msgs::JointTrajectoryGoal goal;

	size_t num_joints = trajectory.joint_names.size();
	goal.trajectory.joint_names.clear();
	goal.trajectory.joint_names.resize(num_joints);
	for (size_t i = 0; i < num_joints; ++i) {
		goal.trajectory.joint_names[i] = trajectory.joint_names[i];
	}

	size_t traj_points = trajectory.points.size();
	goal.trajectory.points.resize(traj_points);

	for (size_t i = 0; i < traj_points; ++i)
	{
		goal.trajectory.points[i].positions.resize(num_joints);
		goal.trajectory.points[i].velocities.resize(num_joints);
		goal.trajectory.points[i].accelerations.resize(num_joints);
		for (size_t j = 0; j < num_joints; ++j)
		{
			goal.trajectory.points[i].positions[j] = trajectory.points[i].positions[j]; // smpl::angles::normalize_angle(trajectory.points[i].positions[j]);
			goal.trajectory.points[i].velocities[j] = trajectory.points[i].velocities[j];
			goal.trajectory.points[i].accelerations[j] = trajectory.points[i].accelerations[j];
			
			// if (i > 0) {
			// 	double angle_diff = smpl::angles::shortest_angle_diff(trajectory.points[i].positions[j], trajectory.points[i-1].positions[j]);
			// 	double time_diff = trajectory.points[i].time_from_start.toSec() - trajectory.points[i-1].time_from_start.toSec();
			// 	goal.trajectory.points[i].velocities[j] = angle_diff/time_diff;
			// }
		}
		goal.trajectory.points[i].time_from_start = trajectory.points[i].time_from_start;
	}

	return goal;
}

bool RobotController::OpenGripper()
{
	// open the gripper
	pr2_controllers_msgs::Pr2GripperCommandGoal goal;
	goal.command.position = 0.08;
	goal.command.max_effort = -1.0; // do not limit effort

	ROS_INFO("Send open gripper goal!");
	auto state = m_gripper_controller->sendGoalAndWait(goal);
	auto res = m_gripper_controller->getResult();

	ROS_INFO("Result:");
	if (res) {
		ROS_INFO("  Effort: %f", res->effort);
		ROS_INFO("  Position %f", res->position);
		ROS_INFO("  Reached Goal: %d", res->reached_goal);
		ROS_INFO("  Stalled: %d", res->stalled);
	}

	if (state != actionlib::SimpleClientGoalState::SUCCEEDED) {
		ROS_ERROR("Failed to open gripper (%s)", state.getText().c_str());
		return false;
	}

	return true;
}

bool RobotController::CloseGripper()
{
	pr2_controllers_msgs::Pr2GripperCommandGoal goal;
	goal.command.position = 0.0;
	goal.command.max_effort = 50.0; // gently
	auto state = m_gripper_controller->sendGoalAndWait(goal);
	// if (state != actionlib::SimpleClientGoalState::SUCCEEDED) {
	//     ROS_ERROR("Failed to close gripper (%s)", state.getText().c_str());
	//     return false;
	// }

	ROS_INFO("Result:");
	auto res = m_gripper_controller->getResult();
	if (res) {
		ROS_INFO("  Effort: %f", res->effort);
		ROS_INFO("  Position %f", res->position);
		ROS_INFO("  Reached Goal: %d", res->reached_goal);
		ROS_INFO("  Stalled: %d", res->stalled);
	}

	return true;
}

} // namespace clutter
