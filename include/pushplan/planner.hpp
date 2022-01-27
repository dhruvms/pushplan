#ifndef PLANNER_HPP
#define PLANNER_HPP

#include <pushplan/types.hpp>
#include <pushplan/agent.hpp>
#include <pushplan/collision_checker.hpp>
#include <pushplan/robot.hpp>
#include <pushplan/bullet_sim.hpp>
#include <comms/ObjectsPoses.h>

#include <ros/ros.h>
#include <std_srvs/Empty.h>

#include <string>
#include <vector>
#include <memory>

namespace clutter
{

class CBS;

class Planner
{
public:
	Planner() : m_num_objs(-1), m_scene_id(-1),
				m_ph("~") {};
	bool Init(const std::string& scene_file, int scene_id, bool ycb);

	bool Plan();
	bool SaveData();
	bool Alive();
	bool Rearrange();
	std::uint32_t RunSim();
	bool TryExtract();
	void AnimateSolution();

	// fcl::CollisionObjectf* GetObject(const LatticeState& s, int priority);
	Agent* GetAgent(const int& id) {
		assert(id > 0); // 0 is robot
		return m_agents.at(m_agent_map[id]).get();
	}

private:
	std::string m_scene_file;
	std::shared_ptr<CollisionChecker> m_cc;
	std::shared_ptr<Robot> m_robot;
	std::shared_ptr<BulletSim> m_sim;
	std::shared_ptr<CBS> m_cbs;

	int m_num_objs, m_scene_id;
	std::vector<std::shared_ptr<Agent> > m_agents;
	std::shared_ptr<Agent> m_ooi;
	std::unordered_map<int, size_t> m_agent_map;
	Coord m_ooi_g;
	State m_ooi_gf;
	std::vector<double> m_goal;

	trajectory_msgs::JointTrajectory m_exec;
	Trajectory m_exec_interm;
	std::vector<trajectory_msgs::JointTrajectory> m_rearrangements;
	comms::ObjectsPoses m_rearranged;

	std::vector<size_t> m_priorities;

	ros::NodeHandle m_ph, m_nh;
	ros::ServiceServer m_simulate, m_animate, m_rearrange;
	std::uint32_t m_violation;

	std::map<std::string, double> m_stats;
	double m_plan_time, m_plan_budget, m_sim_budget, m_total_budget;

	bool runSim(std_srvs::Empty::Request& req, std_srvs::Empty::Response& resp);
	bool animateSolution(std_srvs::Empty::Request& req, std_srvs::Empty::Response& resp);
	bool rearrange(std_srvs::Empty::Request& req, std_srvs::Empty::Response& resp);

	bool setupProblem();
	void updateAgentPositions(
		const comms::ObjectsPoses& result,
		comms::ObjectsPoses& rearranged);
	int cleanupLogs();

	void init_agents(
		bool ycb, std::vector<Object>& obstacles);
	void parse_scene(std::vector<Object>& obstacles);
	void writePlanState(int iter);
	void setupGlobals();
	int armId();

	bool savePlanData();

};

} // namespace clutter


#endif // PLANNER_HPP
