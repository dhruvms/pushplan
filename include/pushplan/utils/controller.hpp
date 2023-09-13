#ifndef CONTROLLER_HPP
#define CONTROLLER_HPP

#include <pr2_controllers_msgs/JointTrajectoryAction.h>
#include <pr2_controllers_msgs/Pr2GripperCommandAction.h>
#include <actionlib/client/simple_action_client.h>
#include <sensor_msgs/JointState.h>

#include <memory>

namespace clutter
{

typedef actionlib::SimpleActionClient< pr2_controllers_msgs::JointTrajectoryAction > TrajClient;
typedef actionlib::SimpleActionClient< pr2_controllers_msgs::Pr2GripperCommandAction> GripperClient;

class RobotController
{
public:
	RobotController();

	void InitControllers();
	void RaiseTorso(const double& value);
	void MoveLeftArmToRest();
	void MoveToStartState(const sensor_msgs::JointState& start_state);
	pr2_controllers_msgs::JointTrajectoryGoal PR2TrajFromMsg(
			const trajectory_msgs::JointTrajectory& trajectory);

	//! Sends the command to start a given trajectory
	void ExecArmTrajectory(pr2_controllers_msgs::JointTrajectoryGoal goal);

	// Gripper action functions
	bool OpenGripper();
	bool CloseGripper(double position=0.0);

	//! Returns the current state of the action
	actionlib::SimpleClientGoalState GetArmState(bool left=false) {
		if (left)
			return m_l_arm_controller->getState();
		return m_r_arm_controller->getState();
	};
	actionlib::SimpleClientGoalState GetTorsoState() {
		return m_torso_controller->getState();
	};

private:
	// Action client for the joint trajectory action
	// used to trigger the arm movement action
	std::unique_ptr<TrajClient> m_r_arm_controller;
	std::unique_ptr<TrajClient> m_l_arm_controller;
	std::unique_ptr<TrajClient> m_torso_controller;
	std::unique_ptr<GripperClient> m_gripper_controller;

	std::vector<double> m_REST_LEFT_ARM = {1.5707963267948966, 1.5707963267948966, 0.0, -0.7853981633974483, 0.0, 0.0, 0.0};

};

} // namespace clutter


#endif // CONTROLLER_HPP
