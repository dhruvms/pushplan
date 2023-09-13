#include <pushplan/search/planner.hpp>
#include <pushplan/agents/agent.hpp>
#include <pushplan/search/cbs.hpp>
#include <pushplan/search/cbswp.hpp>
#include <pushplan/search/pbs.hpp>
#include <pushplan/sampling/sampling_planner.hpp>
#include <pushplan/sampling/rrt.hpp>
#include <pushplan/sampling/rrtstar.hpp>
#include <pushplan/sampling/tcrrt.hpp>
#include <pushplan/utils/constants.hpp>
#include <pushplan/utils/discretisation.hpp>
#include <pushplan/utils/geometry.hpp>
#include <pushplan/utils/helpers.hpp>
#include <comms/ObjectPose.h>

#include <smpl/console/console.h>
#include <moveit_msgs/RobotTrajectory.h>
#include <boost/math/distributions/beta.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cstdlib>

#include <gperftools/profiler.h>

namespace clutter
{

Planner::~Planner()
{
	if (m_cc) {
		m_cc->ReleaseRobot();
	}
	if (m_mamo_search)
	{
		m_cbs->ReleaseRobot();
		m_mamo_search->Cleanup();
	}
	m_robot.reset();
}

bool Planner::Init(const std::string& scene_file, int scene_id, bool ycb)
{
	m_scene_file = scene_file;
	m_scene_id = scene_id;

	std::string tables;
	int num_immov, num_mov;

	m_ph.getParam("object_filename", tables);
	m_ph.getParam("objects/num_immov", num_immov);
	m_ph.getParam("objects/num_mov", num_mov);
	m_sim = std::make_shared<BulletSim>(
				tables, ycb,
				m_scene_id, std::string(),
				num_immov, num_mov);

	m_robot = std::make_shared<Robot>();
	m_robot->SetID(0);
	m_robot->SetSim(m_sim.get());

	m_stats["robot_planner_time"] = 0.0;
	m_stats["push_planner_time"] = 0.0;
	m_stats["push_input_time"] = 0.0;
	m_stats["mapf_time"] = 0.0;
	m_stats["sim_time"] = 0.0;

	m_ph.getParam("goal/plan_budget", m_plan_budget);
	m_ph.getParam("goal/sim_budget", m_sim_budget);
	m_ph.getParam("goal/total_budget", m_total_budget);

	m_agent_map.clear();
	m_agents.clear();
	m_ooi = std::make_shared<Agent>();

	std::vector<Object> all_obstacles;
	if (m_scene_id < 0)
	{
		// init agents from simulator
		if (!init_agents(ycb, all_obstacles))
		{
			SMPL_ERROR("No immovable obstacle is graspable - OOI not set!");
			return false;
		}
	}
	else {
		parse_scene(all_obstacles); // TODO: relax ycb = false assumption
	}

	// create collision checker
	m_cc = std::make_shared<CollisionChecker>(this, all_obstacles, m_robot);
	// m_cc->ReinitMovableCC(m_agents);

	m_robot->SetCC(m_cc);
	for (auto& a: m_agents) {
		a->SetCC(m_cc);
	}
	if (!m_robot->Setup())
	{
		SMPL_ERROR("Robot setup failed!");
		return false;
	}
	if (!m_robot->ProcessObstacles(all_obstacles))
	{
		SMPL_ERROR("Robot collision space setup failed!");
		return false;
	}
	m_robot->ProcessFCLObstacles(m_cc->GetObstacles());

	// Not for PR2 experiments
	m_robot->CreateVirtualTable();

	all_obstacles.clear();

	// Ready OOI
	if (m_ooi->Set())
	{
		setupSim(m_sim.get(), m_robot->GetStartState()->joint_state, m_ooi->GetID());

		m_cc->AddObstacle(m_ooi->GetObject());
		m_ooi->SetCC(m_cc);
		m_robot->SetOOI(m_ooi->GetObject());
		m_robot->SetMovables(m_agents);

		// Compute robot grasping states and plan path through them
		// if (m_robot->ComputeGrasps(m_ooi_pregrasps))
		// {
		// 	if (!SetupNGR())
		// 	{
		// 		SMPL_ERROR("Robot failed to plan path through grasp states!");
		// 		return false;
		// 	}
		// }
		// else {
		// 	SMPL_ERROR("Robot failed to compute grasp states!");
		// 	return false;
		// }
		int t = 0, grasp_tries;
		m_ph.getParam("robot/grasping/tries", grasp_tries);
		for (; t < grasp_tries; ++t)
		{
			if (m_robot->ComputeGrasps(m_ooi_pregrasps))
			{
				if (SetupNGR()) {
					break;
				}
			}
		}

		if (t == grasp_tries)
		{
			SMPL_ERROR("Robot failed to compute grasp states!");
			return false;
		}
	}
	m_robot->VizCC();

	setupTorch();

	setupSim(m_sim.get(), m_robot->GetStartState()->joint_state, m_ooi->GetID());
	m_violation = 0x00000008;
	m_distD = std::uniform_real_distribution<double>(0.0, 1.0);
	m_ph.getParam("robot/pushing/input", m_push_input);

	// // For PR2 experiments
	// m_controller = std::make_unique<RobotController>();
	// m_controller->InitControllers();

	return createMAMOSearch();
}

bool Planner::Alive()
{
	// double plan_time = m_stats["robot_planner_time"] + m_stats["push_planner_time"] + m_stats["mapf_time"];
	// if (plan_time > m_plan_budget) {
	// 	return false;
	// }

	// if (m_stats["sim_time"] + m_robot->SimTime() > m_sim_budget) {
	// 	return false;
	// }

	// total_time includes:
	// 	1. time taken to compute grasps
	// 	2. time taken to plan robot path and setup ngr
	// 	3. time taken to sample, plan, simulate pushes
	// 	4. time taken to solve mapf problem
	double total_time = m_stats["robot_planner_time"] + m_stats["push_planner_time"] + m_stats["mapf_time"];
	return total_time < m_total_budget;
}

bool Planner::SetupNGR()
{
	std::vector<Object*> movable_obstacles;
	for (const auto& a: m_agents) {
		movable_obstacles.push_back(a->GetObject());
	}

	m_robot->ProcessObstacles({ m_ooi->GetObject() }, true);
	// if (!m_robot->PlanApproachOnly(movable_obstacles)) {
	m_timer = GetTime();
	if (!m_robot->PlanRetrieval(movable_obstacles))	{
		return false;
	}
	ROS_WARN("Planner::SetupNGR > Robot::PlanRetrieval took %f seconds!", GetTime() - m_timer);
	m_stats["robot_planner_time"] = GetTime() - m_timer;
	m_robot->ProcessObstacles({ m_ooi->GetObject() });

	m_robot->UpdateNGR(false);
	m_exec = m_robot->GetLastPlanProfiled();

	double ox, oy, oz, sx, sy, sz;
	m_sim->GetShelfParams(ox, oy, oz, sx, sy, sz);

	for (auto& a: m_agents)
	{
		a->SetObstacleGrid(m_robot->ObsGrid());
		a->SetNGRGrid(m_robot->NGRGrid());
		a->ResetSolution();
		// if (ALGO == MAPFAlgo::OURS) {
		// 	a->ComputeNGRComplement(ox, oy, oz, sx, sy, sz);
		// }
		// a->Init();
		// a->VisualiseState(a->InitState().coord, "movable_"+std::to_string(a->GetID()), 215);
	}
	// for (auto& a: m_immovables)
	// {
	// 	a->SetObstacleGrid(m_robot->ObsGrid());
	// 	a->SetNGRGrid(m_robot->NGRGrid());
	// 	a->ResetSolution();
	// 	a->Init();
	// 	a->VisualiseState(a->InitState().coord, "immovable_"+std::to_string(a->GetID()), 340);
	// }
	// m_ooi->SetObstacleGrid(m_robot->ObsGrid());
	// m_ooi->SetNGRGrid(m_robot->NGRGrid());
	// m_ooi->ResetSolution();
	// m_ooi->Init();
	// m_ooi->VisualiseState(m_ooi->InitState().coord, "ooi", 50);

	// m_ooi->SetObstacleGrid(m_robot->Grid());
	// if (ALGO == MAPFAlgo::OURS) {
	// 	m_ooi->ComputeNGRComplement(ox, oy, oz, sx, sy, sz);
	// }

	auto ngr_voxels = m_robot->TrajVoxels();
	for (auto itr_list = ngr_voxels->begin(); itr_list != ngr_voxels->end(); ++itr_list)
	{
		for (auto itr = itr_list->begin(); itr != itr_list->end(); ++itr)
		{
			Coord c;
			c.push_back(DiscretisationManager::ContXToDiscX(itr->x()));
			c.push_back(DiscretisationManager::ContYToDiscY(itr->y()));
			c.push_back(DiscretisationManager::ContYToDiscY(itr->z()));
			m_ngr.insert(std::move(c));
		}
	}

	// m_first_traj_success = TryExtract();

	return true;
}

bool Planner::Plan(bool& done)
{
	done = false;
	bool result = m_mamo_search->Solve(m_total_budget);
	bool exec_success = false;
	if (result)
	{
		m_mamo_search->GetRearrangements(m_rearrangements, m_actions);
		exec_success = runSim();
		done = true;
		ROS_ERROR("Execution success: %d", exec_success);
		if (SAVE) {
			writeState("SOLUTION", "lazystats");
		}
	}
	m_mamo_search->SaveStats((int)exec_success);
	m_robot->SavePushDebugData(m_scene_id);

	return done;
}

bool Planner::FinalisePlan(
	const std::vector<ObjectState>& objects,
	std::vector<double>* start_state,
	trajectory_msgs::JointTrajectory& solution)
{
	m_timer = GetTime();

	std::vector<Object*> movable_obstacles;
	for (const auto& pose: objects)
	{
		m_agents.at(m_agent_map[pose.id()])->SetObjectPose(pose.cont_pose());
		movable_obstacles.push_back(m_agents.at(m_agent_map[pose.id()])->GetObject());
	}

	m_robot->ProcessObstacles({ m_ooi->GetObject() }, true);
	if (!m_robot->PlanRetrieval(movable_obstacles, true, start_state))
	{
		m_stats["robot_planner_time"] += GetTime() - m_timer;
		m_robot->ProcessObstacles({ m_ooi->GetObject() });
		return false;
	}
	ROS_WARN("Planner::FinalisePlan > Robot::PlanRetrieval took %f seconds!", GetTime() - m_timer);
	m_robot->ProcessObstacles({ m_ooi->GetObject() });
	solution = m_robot->GetLastPlanProfiled();

	m_stats["robot_planner_time"] += GetTime() - m_timer;
	return true;
}

bool Planner::PlanToHomeState(
	const std::vector<ObjectState>& objects,
	std::vector<double>* start_state,
	trajectory_msgs::JointTrajectory& solution)
{
	m_timer = GetTime();

	std::vector<Object*> movable_obstacles;
	for (const auto& pose: objects)
	{
		m_agents.at(m_agent_map[pose.id()])->SetObjectPose(pose.cont_pose());
		movable_obstacles.push_back(m_agents.at(m_agent_map[pose.id()])->GetObject());
	}

	// m_robot->ProcessObstacles({ m_ooi->GetObject() }, true);
	if (!m_robot->PlanToHomeState(movable_obstacles, start_state))
	{
		m_stats["robot_planner_time"] += GetTime() - m_timer;
		// m_robot->ProcessObstacles({ m_ooi->GetObject() });
		return false;
	}
	// m_robot->ProcessObstacles({ m_ooi->GetObject() });
	solution = m_robot->GetLastPlanProfiled();

	m_stats["robot_planner_time"] += GetTime() - m_timer;
	return true;
}

void Planner::AddGloballyInvalidPush(
	const std::pair<Coord, Coord>& bad_start_goal)
{
	m_invalid_pushes_G.push_back(std::move(bad_start_goal));
}

void Planner::AddLocallyInvalidPush(
	unsigned int state_id, int agent_id, Coord bad_goal, int samples)
{
	const auto it1 = m_invalid_pushes_L.find(state_id);
	if (it1 == m_invalid_pushes_L.end())
	{
		// never seen an invalid push for this state before
		m_invalid_pushes_L[state_id][agent_id][bad_goal] = samples;
	}
	else
	{
		const auto it2 = it1->second.find(agent_id);
		if (it2 == it1->second.end())
		{
			// never seen an invalid push for this object in this state before
			it1->second[agent_id][bad_goal] = samples;
		}
		else
		{
			const auto it3 = it2->second.find(bad_goal);
			if (it3 == it2->second.end())
			{
				// never seen this invalid push for this object in this state before
				it2->second[bad_goal] = samples;
			}
			else
			{
				// have seen this invalid push for this object in this state before
				it3->second = std::max(it3->second, samples);
			}
		}
	}
}

bool Planner::createCBS()
{
	switch (ALGO)
	{
		case MAPFAlgo::VCBS:
		case MAPFAlgo::ECBS:
		{
			m_cbs = std::make_shared<CBS>(m_robot, m_agents, m_scene_id);
			break;
		}
		case MAPFAlgo::CBSWP:
		{
			m_cbs = std::make_shared<CBSwP>(m_robot, m_agents, m_scene_id);
			break;
		}
		case MAPFAlgo::PBS:
		{
			m_cbs = std::make_shared<PBS>(m_robot, m_agents, m_scene_id);
			break;
		}
		default:
		{
			SMPL_ERROR("MAPF Algo type currently not supported!");
			return false;
		}
	}
	m_cbs->SetCC(m_cc);

	return true;
}

bool Planner::createMAMOSearch()
{
	if (!createCBS()) {
		return false;
	}

	m_mamo_search = std::make_unique<MAMOSearch>(this);
	return m_mamo_search->CreateRoot();
}

bool Planner::setupProblem()
{
	// CBS TODO: assign starts and goals to agents

	// int result = cleanupLogs();
	// if (result == -1) {
	// 	SMPL_ERROR("system command errored!");
	// }

	// Set agent current positions and time
	// m_robot->Init();
	// if (!m_robot->RandomiseStart()) {
	// 	return false;
	// }

	for (auto& a: m_agents) {
		a->Init();
	}

	return true;
}

void Planner::setupTorch()
{
	double ox, oy, oz, sx, sy, sz;
	m_sim->GetShelfParams(ox, oy, oz, sx, sy, sz);

	std::vector<double> push_locs;
	for (double dx = -sx/2; dx <= sx/2; dx += RES)
	{
		for (double dy = -sy/2; dy <= sy/2; dy += RES)
		{
			push_locs.push_back(dx);
			push_locs.push_back(dy);
		}
	}
	std::vector<int64_t> push_locs_size = {(int)push_locs.size()/2, 2};
	m_tensoroptions = std::make_shared<at::TensorOptions>(at::TensorOptions(at::ScalarType::Double));
	m_push_locs = std::make_shared<at::Tensor>(torch::from_blob(push_locs.data(), at::IntList(push_locs_size), *m_tensoroptions));
	*m_push_locs = m_push_locs->toType(at::kFloat);

	std::string torch_model_path;
	m_ph.getParam("model_file", torch_model_path);
	m_push_model = std::make_shared<torch::jit::script::Module>(torch::jit::load(torch_model_path));

	for (auto& a: m_agents) {
		a->InitTorch(m_tensoroptions, m_push_locs, m_push_model, ox + sx/2, oy + sy/2, oz, sx/2, sy/2);
	}
}

bool Planner::TryExtract()
{
	m_timer = GetTime();

	if (!setupSim(m_sim.get(), m_robot->GetStartState()->joint_state, m_ooi->GetID()))
	{
		m_stats["sim_time"] += GetTime() - m_timer;
		return false;
	}

	comms::ObjectsPoses rearranged = m_rearranged;
	bool success = true;
	if (!m_sim->ExecTraj(m_exec, rearranged, m_robot->GraspAt(), m_ooi->GetID()))
	{
		SMPL_ERROR("Failed to exec traj!");
		success = false;
	}

	m_sim->RemoveConstraint();

	m_stats["sim_time"] += GetTime() - m_timer;
	return success;
}

std::uint32_t Planner::RunSolution()
{
	std::string soln_suffix;
	int soln_num;
	m_ph.getParam("robot/soln/suffix", soln_suffix);
	m_ph.getParam("robot/soln/num", soln_num);
	for (int i = 1; i <= soln_num; ++i)
	{
		std::string suffix = soln_suffix;
		suffix += std::to_string(i);
		read_solution(suffix);
		RunSim(false);
	}
	// read_solution(soln_suffix);
	// return RunSim(false);

	// ROS_ERROR("SHOULD I EXECUTE???");
	// getchar();
	// return Execute();
}

std::uint32_t Planner::RunSim(bool save)
{
	m_timer = GetTime();
	if (!runSim()) {
		SMPL_ERROR("Simulation failed!");
	}
	m_stats["sim_time"] += GetTime() - m_timer;

	// if (save) {
	// 	writeState("SOLUTION");
	// }

	return m_violation;
}

void Planner::AnimateSolution()
{
	std::vector<Object*> final_objects;
	for (auto& a: m_agents) {
		// a->ResetObject();
		final_objects.push_back(a->GetObject());
	}
	m_robot->ProcessObstacles(final_objects);

	animateSolution();
}

bool Planner::runSim()
{
	setupSim(m_sim.get(), m_robot->GetStartState()->joint_state, m_ooi->GetID());

	// if all executions succeeded, m_violation == 0
	m_violation = 0x00000000;

	comms::ObjectsPoses dummy;
	for (size_t i = 0; i < m_rearrangements.size(); ++i)
	{
		if (m_rearrangements.at(i).points.empty()) {
			continue;
		}

		auto &action_params = m_actions.at(i);
		int pick_at = -1, place_at = -1, oid = -1;
		if (i < m_rearrangements.size() - 1)
		{
			// if any rearrangement traj execuction failed, m_violation == 1
			if (action_params._type == MAMOActionType::PICKPLACE)
			{
				pick_at = action_params._params[0];
				place_at = action_params._params[1];
				oid = action_params._oid;
			}

			if (!m_sim->ExecTraj(m_rearrangements.at(i), dummy, pick_at, place_at, oid))
			{
				SMPL_ERROR("Failed to exec rearrangement!");
				m_violation |= 0x00000001;
			}
		}
		else
		{
			// if all rearrangements succeeded, but extraction failed, m_violation == 4
			// if any rearrangement traj execuction failed, and extraction failed, m_violation == 5
			pick_at = action_params._params[0];
			if (!m_sim->ExecTraj(m_rearrangements.at(i), dummy, pick_at, place_at, m_ooi->GetID()))
			{
				SMPL_ERROR("Failed to exec traj!");
				m_violation |= 0x00000004;
			}
		}
	}

	m_sim->RemoveConstraint();
	m_sim_success = m_violation == 0;

	return m_sim_success;
}

bool Planner::animateSolution()
{
	m_robot->AnimateSolution();
	return true;
}

////////
// IO //
////////

int Planner::cleanupLogs()
{
	std::string files(__FILE__), command;
	auto found = files.find_last_of("/\\");
	files = files.substr(0, found + 1) + "../../dat/txt/*.txt";

	command = "rm " + files;
	// SMPL_WARN("Execute command: %s", command.c_str());
	return system(command.c_str());
}

bool Planner::init_agents(
	bool ycb, std::vector<Object>& obstacles)
{
	auto mov_objs = m_sim->GetMovableObjs();
	auto mov_obj_ids = m_sim->GetMovableObjIDs();
	auto immov_objs = m_sim->GetImmovableObjs();
	auto immov_obj_ids = m_sim->GetImmovableObjIDs();

	obstacles.clear();

	int tables = FRIDGE ? 5 : 1;
	bool ooi_set = false;
	Object o;
	for (size_t i = 0; i < immov_objs->size(); ++i)
	{
		o.desc.id = immov_obj_ids->at(i).first;
		o.desc.shape = immov_obj_ids->at(i).second;
		o.desc.type = i < tables ? -1 : 0; // table or immovable
		o.desc.o_x = immov_objs->at(i).at(0);
		o.desc.o_y = immov_objs->at(i).at(1);
		o.desc.o_z = immov_objs->at(i).at(2);
		o.desc.o_roll = immov_objs->at(i).at(3);
		o.desc.o_pitch = immov_objs->at(i).at(4);
		o.desc.o_yaw = immov_objs->at(i).at(5);
		o.desc.ycb = (bool)immov_objs->at(i).back();
		o.desc.yaw_offset = 0.0;

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

		o.SetupGaussianCost();
		o.CreateCollisionObjects();
		o.CreateSMPLCollisionObject();
		o.GenerateCollisionModels();
		o.InitProperties();

		if (!ooi_set && i >= tables && o.Graspable())
		{
			Eigen::Affine3d ooi_T = Eigen::Translation3d(o.desc.o_x, o.desc.o_y, o.desc.o_z) *
										Eigen::AngleAxisd(o.desc.o_yaw, Eigen::Vector3d::UnitZ()) *
										Eigen::AngleAxisd(o.desc.o_pitch, Eigen::Vector3d::UnitY()) *
										Eigen::AngleAxisd(o.desc.o_roll, Eigen::Vector3d::UnitX());
			o.SetTransform(ooi_T);
			m_ooi_pregrasps.clear();
			o.GetPregrasps(m_ooi_pregrasps);

			m_ooi->SetObject(o);
			ooi_set = true;
			continue;
		}

		obstacles.push_back(o);
	}

	for (size_t i = 0; i < mov_objs->size(); ++i)
	{
		o.desc.id = mov_obj_ids->at(i).first;
		o.desc.shape = mov_obj_ids->at(i).second;
		o.desc.type = 1; // movable
		o.desc.o_x = mov_objs->at(i).at(0);
		o.desc.o_y = mov_objs->at(i).at(1);
		o.desc.o_z = mov_objs->at(i).at(2);
		o.desc.o_roll = mov_objs->at(i).at(3);
		o.desc.o_pitch = mov_objs->at(i).at(4);
		o.desc.o_yaw = mov_objs->at(i).at(5);
		o.desc.ycb = (bool)mov_objs->at(i).back();
		o.desc.yaw_offset = 0.0;

		if (o.desc.ycb)
		{
			auto itr = YCB_OBJECT_DIMS.find(mov_obj_ids->at(i).second);
			if (itr != YCB_OBJECT_DIMS.end())
			{
				o.desc.x_size = itr->second.at(0);
				o.desc.y_size = itr->second.at(1);
				o.desc.z_size = itr->second.at(2);
				o.desc.o_yaw += itr->second.at(3);
			}
			o.desc.mass = -1;
			o.desc.mu = mov_objs->at(i).at(6);
		}
		else
		{
			o.desc.x_size = mov_objs->at(i).at(6);
			o.desc.y_size = mov_objs->at(i).at(7);
			o.desc.z_size = mov_objs->at(i).at(8);
			o.desc.mass = mov_objs->at(i).at(9);
			o.desc.mu = mov_objs->at(i).at(10);
		}
		o.desc.movable = true;
		o.desc.locked = false;

		o.CreateCollisionObjects();
		o.CreateSMPLCollisionObject();
		o.GenerateCollisionModels();
		o.InitProperties();

		std::shared_ptr<Agent> movable(new Agent(o));
		m_agents.push_back(std::move(movable));
		m_agent_map[o.desc.id] = m_agents.size() - 1;
	}

	return ooi_set;
}

void Planner::parse_scene(std::vector<Object>& obstacles)
{
	std::ifstream SCENE;
	SCENE.open(m_scene_file);

	if (SCENE.is_open())
	{
		std::string line;
		while (!SCENE.eof())
		{
			getline(SCENE, line);
			if (line.compare("O") == 0)
			{
				getline(SCENE, line);
				int num_objs = std::stoi(line);
				obstacles.clear();

				for (int i = 0; i < num_objs; ++i)
				{
					Object o;

					getline(SCENE, line);
					std::stringstream ss(line);
					std::string split;

					int count = 0;
					while (ss.good())
					{
						getline(ss, split, ',');
						switch (count) {
							case 0: o.desc.id = std::stoi(split); break;
							case 1: o.desc.shape = std::stoi(split); break;
							case 2: o.desc.type = std::stoi(split); break;
							case 3: o.desc.o_x = std::stof(split); break;
							case 4: o.desc.o_y = std::stof(split); break;
							case 5: o.desc.o_z = std::stof(split); break;
							case 6: o.desc.o_roll = std::stof(split); break;
							case 7: o.desc.o_pitch = std::stof(split); break;
							case 8: o.desc.o_yaw = std::stof(split); break;
							case 9: o.desc.x_size = std::stof(split); break;
							case 10: o.desc.y_size = std::stof(split); break;
							case 11: o.desc.z_size = std::stof(split); break;
							case 12: {
								o.desc.mass = std::stof(split);
								o.desc.locked = o.desc.mass == 0;
								break;
							}
							case 13: o.desc.mu = std::stof(split); break;
							case 14: o.desc.movable = (split.compare("True") == 0); break;
						}
						o.desc.ycb = false;
						o.desc.yaw_offset = 0.0;
						count++;
					}

					o.CreateCollisionObjects();
					o.CreateSMPLCollisionObject();
					o.GenerateCollisionModels();
					o.InitProperties();

					if (o.desc.movable)
					{
						std::shared_ptr<Agent> movable(new Agent(o));
						m_agents.push_back(std::move(movable));
						m_agent_map[o.desc.id] = m_agents.size() - 1;
					}
					else
					{
						o.SetupGaussianCost();
						// o.desc.x_size += RES;
						// o.desc.y_size += RES;
						obstacles.push_back(o);

						// if (o.desc.id > 5)
						// {
						// 	std::shared_ptr<Agent> movable(new Agent(o));
						// 	m_immovables.push_back(std::move(movable));
						// }
					}
				}
			}

			else if (line.compare("ooi") == 0)
			{
				getline(SCENE, line);
				int ooi_idx = std::stoi(line); // object of interest ID

				for (auto itr = obstacles.begin(); itr != obstacles.end(); ++itr)
				{
					if (itr->desc.id == ooi_idx)
					{
						// Eigen::Affine3d ooi_T = Eigen::Translation3d(itr->desc.o_x, itr->desc.o_y, itr->desc.o_z) *
						// 							Eigen::AngleAxisd(itr->desc.o_yaw, Eigen::Vector3d::UnitZ()) *
						// 							Eigen::AngleAxisd(itr->desc.o_pitch, Eigen::Vector3d::UnitY()) *
						// 							Eigen::AngleAxisd(itr->desc.o_roll, Eigen::Vector3d::UnitX());
						// itr->SetTransform(ooi_T);
						// m_ooi_pregrasps.clear();
						// itr->GetPregrasps(m_ooi_pregrasps);

						m_ooi->SetObject(*itr);
						obstacles.erase(itr);
						break;
					}
				}
			}

			else if (line.compare("G") == 0)
			{
				getline(SCENE, line);

				std::stringstream ss(line);
				std::string split;
				std::vector<double> pregrasp_goal;
				while (ss.good())
				{
					getline(ss, split, ',');
					pregrasp_goal.push_back(std::stod(split));
				}

				std::swap(pregrasp_goal[3], pregrasp_goal[5]);
				if (std::fabs(pregrasp_goal[3]) >= 1e-4)
				{
					pregrasp_goal[3] = 0.0;
					pregrasp_goal[4] = 0.0;
					pregrasp_goal[5] = smpl::angles::normalize_angle(pregrasp_goal[5] + M_PI);
				}
				SMPL_INFO("Goal (x, y, z, yaw): (%f, %f, %f, %f)", pregrasp_goal[0], pregrasp_goal[1], pregrasp_goal[2], pregrasp_goal[5]);

				Eigen::Affine3d ooi_T = Eigen::Translation3d(pregrasp_goal[0], pregrasp_goal[1], pregrasp_goal[2]) *
											Eigen::AngleAxisd(pregrasp_goal[5], Eigen::Vector3d::UnitZ()) *
											Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
											Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());
				m_ooi_pregrasps.push_back(std::move(ooi_T));
			}
		}
	}

	SCENE.close();
}

void Planner::read_solution(const std::string &suffix)
{
	std::string filename(__FILE__);
	auto found = filename.find_last_of("/\\");
	filename = filename.substr(0, found + 1) + "../../dat/txt/";

	std::stringstream ss;
	ss << "SOLUTION" << "_" << m_scene_id;
	ss << "_";
	ss << suffix;
	std::string s = ss.str();

	filename += s;
	filename += ".txt";

	std::ifstream SOLUTION;
	SOLUTION.open(filename);

	if (SOLUTION.is_open())
	{
		std::string line;
		while (!SOLUTION.eof())
		{
			getline(SOLUTION, line);
			if (line.compare("SOLUTION") == 0)
			{
				getline(SOLUTION, line);
				int num_trajs = std::stoi(line);

				// read all rearrangement trajs
				m_rearrangements.clear();
				m_actions.clear();
				for (int i = 0; i < num_trajs; ++i)
				{
					getline(SOLUTION, line);
					if (line.compare("TRAJECTORY") != 0)
					{
						SMPL_ERROR("Did not find the start of a new solution trajectory where I expected!");
						break;
					}

					// get MAMOAction params
					getline(SOLUTION, line);
					std::stringstream ssp_action(line);
					std::string split_action;

					MAMOAction action;
					int count = 0;
					while (ssp_action.good())
					{
						getline(ssp_action, split_action, ',');
						int val = std::stoi(split_action);
						switch (count)
						{
							case 0 : {
								action._type = static_cast<MAMOActionType>(val);
								break;
							}
							case 1 :
							case 2 : {
								if (val != -1) {
									action._params.push_back(val);
								}
								break;
							}
							case 3 : {
								if (val != -1) {
									action._oid = val;
								}
								break;
							}
							default : {
								break;
							}
						}
						++count;
					}
					m_actions.push_back(std::move(action));

					getline(SOLUTION, line);
					int traj_points = std::stoi(line);

					trajectory_msgs::JointTrajectory push_traj;
					push_traj.joint_names = m_robot->RobotModel()->getPlanningJoints();
					for (int j = 0; j < traj_points; ++j)
					{
						getline(SOLUTION, line);
						std::stringstream ssp_wp(line);
						std::string split_wp;

						trajectory_msgs::JointTrajectoryPoint p;
						int count = 0;
						while (ssp_wp.good())
						{
							getline(ssp_wp, split_wp, ',');
							if (count < 7) {
								p.positions.push_back(std::stod(split_wp));
							}

							// // For PR2 experiments
							// else if (count < 14) {
							// 	p.velocities.push_back(std::stod(split));
							// }
							// else if (count < 21) {
							// 	p.accelerations.push_back(std::stod(split));
							// }

							else {
								p.time_from_start = ros::Duration(std::stod(split_wp));
							}
							++count;
						}

						push_traj.points.push_back(std::move(p));
					}
					m_rearrangements.push_back(std::move(push_traj));
				}
			}
		}
	}

	SOLUTION.close();
}

bool Planner::SaveData()
{
	auto robot_stats = m_robot->GetStats();

	std::string filename(__FILE__);
	auto found = filename.find_last_of("/\\");
	filename = filename.substr(0, found + 1) + "../../dat/PLANNER.csv";

	bool exists = FileExists(filename);
	std::ofstream STATS;
	STATS.open(filename, std::ofstream::out | std::ofstream::app);
	if (!exists)
	{
		STATS << "UID,"
				<< "PlanSuccess,SimSuccess,SimResult,SimTime,"
				<< "RobotPlanTime,MAPFTime,PushInputTime,PlanPushTime,"
				<< "PlanPushCalls,PushActionsFound,"
				<< "PushSimSuccesses,PushPlanningTime,PushSimTime,"
				<< "CBSCalls,CBSSuccesses,CBSTime,"
				<< "CTNodes,CTDeadends,CTExpansions,LLTime,ConflictDetectionTime\n";
	}

	STATS << m_scene_id << ','
			<< m_plan_success << ',' << m_sim_success << ','
			<< m_violation << ',' << m_stats["sim_time"] << ','
			<< m_stats["robot_planner_time"] << ',' << m_stats["mapf_time"] << ','
			<< m_stats["push_input_time"] << ',' << m_stats["push_planner_time"] << ','
			<< robot_stats["plan_push_calls"] << ',' << robot_stats["push_actions_found"] << ','
			<< robot_stats["push_sim_successes"] << ','
			<< robot_stats["push_plan_time"] << ',' << robot_stats["push_sim_time"] << ','
			<< m_cbs_stats["calls"] << ',' << m_cbs_stats["solved"] << ','
			<< m_cbs_stats["search_time"] << ',' << m_cbs_stats["ct_nodes"] << ','
			<< m_cbs_stats["ct_deadends"] << ',' << m_cbs_stats["ct_expanded"] << ','
			<< m_cbs_stats["ll_time"] << ',' << m_cbs_stats["conflict_time"] << '\n';
	STATS.close();
}

////////////////////
// Sampling-Based //
////////////////////

bool Planner::RunRRT()
{
	int samples, steps, planner;
	double gbias, gthresh, timeout;

	m_ph.getParam("sampling/samples", samples);
	m_ph.getParam("sampling/steps", steps);
	m_ph.getParam("sampling/gbias", gbias);
	m_ph.getParam("sampling/gthresh", gthresh);
	m_ph.getParam("sampling/timeout", timeout);
	m_ph.getParam("sampling/planner", planner);

	switch (planner)
	{
		case 0:
		{
			m_sampling_planner = std::make_shared<sampling::RRT>(
				samples, steps, gbias, gthresh, timeout);
			SMPL_INFO("Run RRT");
			break;
		}
		case 1:
		{
			m_sampling_planner = std::make_shared<sampling::RRTStar>(
				samples, steps, gbias, gthresh, timeout);
			SMPL_INFO("Run RRTStar");
			break;
		}
		case 2:
		{
			int mode, I, J;
			m_ph.getParam("sampling/tcrrt/mode", mode);
			m_ph.getParam("sampling/tcrrt/I", I);
			m_ph.getParam("sampling/tcrrt/J", J);

			m_sampling_planner = std::make_shared<sampling::TCRRT>(
				samples, steps, gbias, gthresh, timeout, mode, I, J);
			SMPL_INFO("Run TCRRT");

			smpl::RobotState pregrasp_state;
			Eigen::Affine3d pregrasp_pose;
			m_robot->GetPregraspState(pregrasp_state);
			m_robot->ComputeFK(pregrasp_state, pregrasp_pose);
			m_robot->VisPlane(pregrasp_pose.translation().z());
			m_sampling_planner->SetConstraintHeight(pregrasp_pose.translation().z());
			break;
		}
		default:
		{
			SMPL_ERROR("Unsupported sampling planner!");
			return false;
		}
	}

	m_sampling_planner->SetRobot(m_robot);
	m_sampling_planner->SetRobotGoalCallback(std::bind(&Robot::GetPregraspState, m_robot.get(), std::placeholders::_1));

	smpl::RobotState start_config;
	m_robot->GetHomeState(start_config);
	comms::ObjectsPoses start_objects = GetStartObjects();
	m_sampling_planner->SetStartState(start_config, start_objects);

	bool plan_success = m_sampling_planner->Solve();
	bool exec_success = false;
	// if (plan_success)
	{
		m_sampling_planner->ExtractTraj(m_exec);
		exec_success = m_sim->ExecTraj(m_exec, start_objects);
	}

	auto rrt_stats = m_sampling_planner->GetStats();
	std::string filename(__FILE__);
	auto found = filename.find_last_of("/\\");
	filename = filename.substr(0, found + 1) + "../../dat/";
	switch (planner)
	{
		case 0:
		{
			filename += "RRT.csv";
			break;
		}

		case 1:
		{
			filename += "RRTStar.csv";
			break;
		}

		case 2:
		{
			int mode;
			m_ph.getParam("sampling/tcrrt/mode", mode);
			filename += "TCRRT_" + std::to_string(mode) + ".csv";
			break;
		}
	}

	// m_stats["
	// m_stats["goal_samples"] = 0.0;
	// m_stats["random_samples"] = 0.0;

	bool exists = FileExists(filename);
	std::ofstream STATS;
	STATS.open(filename, std::ofstream::out | std::ofstream::app);
	if (!exists)
	{
		STATS << "UID,"
				<< "PlanSuccess,ExecSuccess,SimCalls,SimTime,"
				<< "TotalVertices,PlanTime,SolnCost,"
				<< "FirstGoalVertex,FirstSolnTime,"
				<< "GoalSamples,RandomSamples\n";
	}

	STATS << m_scene_id << ','
			<< plan_success << ',' << exec_success << ','
			<< rrt_stats["sims"] << ',' << rrt_stats["sim_time"] << ','
			<< rrt_stats["vertices"] << ',' << rrt_stats["plan_time"] << ','
			<< rrt_stats["soln_cost"] << ',' << rrt_stats["first_goal"] << ','
			<< rrt_stats["first_soln_time"] << ',' << rrt_stats["goal_samples"] << ','
			<< rrt_stats["random_samples"] << '\n';
	STATS.close();

	return false;
}

////////////////
// IK Studies //
////////////////

void Planner::RunStudy(int study)
{
	int N;
	if (study == 0)
	{
		m_ph.getParam("robot/yoshikawa_tries", N);
		m_robot->RunManipulabilityStudy(N);
	}
	else if (study == 1)
	{
		m_ph.getParam("robot/push_ik_ends", N);
		m_robot->RunPushIKStudy(N);
	}
}

///////////////
// Utilities //
///////////////

const std::shared_ptr<Agent>& Planner::GetAgent(const int& id)
{
	assert(id > 0); // 0 is robot
	return m_agents.at(m_agent_map[id]);
}

const std::vector<std::shared_ptr<Agent> >& Planner::GetAllAgents()
{
	return m_agents;
}

bool Planner::Replan()
{
	return m_replan;
}

const std::shared_ptr<CBS>& Planner::GetCBS() const
{
	return m_cbs;
}

const std::shared_ptr<CollisionChecker>& Planner::GetCC() const
{
	return m_cc;
}

const std::shared_ptr<Robot>& Planner::GetRobot() const
{
	return m_robot;
}

const std::vector<std::pair<Coord, Coord> >* Planner::GetGloballyInvalidPushes() const
{
	return &m_invalid_pushes_G;
}

const std::map<Coord, int, coord_compare>* Planner::GetLocallyInvalidPushes(
	unsigned int state_id, int agent_id) const
{
	if (m_invalid_pushes_L.empty()) {
		return nullptr;
	}

	const auto it1 = m_invalid_pushes_L.find(state_id);
	if (it1 != m_invalid_pushes_L.end())
	{
		const auto it2 = it1->second.find(agent_id);
		if (it2 != it1->second.end()) {
			return &(it2->second);
		}
	}

	return nullptr;
}

const int& Planner::GetSceneID() const
{
	return m_scene_id;
}

const int& Planner::GetOoIID() const
{
	return m_ooi->GetID();
}

const std::set<Coord, coord_compare>& Planner::GetNGR() const
{
	return m_ngr;
}

const trajectory_msgs::JointTrajectory& Planner::GetFirstTraj() const
{
	return m_exec;
}

const bool& Planner::GetFirstTrajSuccess() const
{
	return m_first_traj_success;
}

//////////////////////////
// Prioritised Planning //
//////////////////////////

bool Planner::RunPP()
{
	m_ooi->InitPP();
	for (auto& a: m_agents) {
		a->InitPP();
	}

	bool success = false;
	int makespan = -1, flowtime = 0, failure = 0;
	double plan_time = 0.0;

	m_timer = GetTime();
	prioritise();

	int priority = 0;
	for (const auto& p: m_priorities)
	{
		double ll_time = GetTime();
		if (!m_agents.at(p)->PlanPrioritised(priority))
		{
			failure = 1; // low-level planner failed to find a solution
			plan_time = GetTime() - m_timer;
			break;
		}
		if (GetTime() - ll_time > 30.0)
		{
			failure = 2; // low-level planner timed out
			plan_time = GetTime() - m_timer;
			break;
		}
		++priority;
	}

	if (failure == 0 && (GetTime() - m_timer) > 30.0)
	{
		failure = 3; // high-level planner timed out
		plan_time = GetTime() - m_timer;
	}

	// high-level planner did not fail
	if (failure == 0)
	{
		// PP found a solution!
		success = true;
		plan_time = GetTime() - m_timer;
		for (const auto& a: m_agents)
		{
			int move_length = int(a->SolveTraj()->size());
			flowtime += move_length;
			if (move_length > makespan) {
				makespan = move_length;
			}
		}
		writeState("PP");
	}

	std::string filename(__FILE__);
	auto found = filename.find_last_of("/\\");
	filename = filename.substr(0, found + 1) + "../../dat/PP.csv";

	bool exists = FileExists(filename);
	std::ofstream STATS;
	STATS.open(filename, std::ofstream::out | std::ofstream::app);
	if (!exists)
	{
		STATS << "UID,"
				<< "Success,Failure?,PlanTime,"
				<< "Makespan,Flowtime\n";
	}

	STATS << m_scene_id << ','
			<< success << ',' << failure << ','
			<< plan_time << ',' << makespan << ',' << flowtime << '\n';
	STATS.close();

	return true;
}

void Planner::prioritise()
{
	m_priorities.clear();
	m_priorities.resize(m_agents.size());
	std::iota(m_priorities.begin(), m_priorities.end(), 0);

	smpl::RobotState home_state;
	Eigen::Affine3d home_pose;
	m_robot->GetHomeState(home_state);
	m_robot->ComputeFK(home_state, home_pose);
	State robot = { home_pose.translation().x(), home_pose.translation().y() };
	std::vector<double> dists(m_agents.size(), std::numeric_limits<double>::max());
	for (size_t i = 0; i < m_agents.size(); ++i) {
		dists.at(i) = std::min(dists.at(i), EuclideanDist(robot, m_agents.at(i)->InitState().state));
	}
	std::stable_sort(m_priorities.begin(), m_priorities.end(),
		[&dists](size_t i1, size_t i2) { return dists[i1] < dists[i2]; });
}

void Planner::writeState(const std::string& prefix, const std::string& suffix)
{
	std::string filename(__FILE__);
	auto found = filename.find_last_of("/\\");
	filename = filename.substr(0, found + 1) + "../../dat/txt/";

	std::stringstream ss;
	ss << prefix << "_" << m_scene_id << "_" << suffix << "_";
	auto t = std::time(nullptr);
	auto tm = *std::localtime(&t);
	ss << std::put_time(&tm, "%d-%m-%Y-%H-%M-%S");
	ss << "_";
	std::string s = ss.str();

	filename += s;
	filename += ".txt";

	std::ofstream DATA;
	DATA.open(filename, std::ofstream::out);

	DATA << 'O' << '\n';
	int o = m_cc->NumObstacles() + m_agents.size();
	DATA << o << '\n';

	std::string movable = "False";
	auto obstacles = m_cc->GetObstacles();
	for (const auto& obs: *obstacles)
	{
		int id = obs.desc.id == m_ooi->GetID() ? 999 : obs.desc.id;
		DATA << id << ','
				<< obs.Shape() << ','
				<< obs.desc.type << ','
				<< obs.desc.o_x << ','
				<< obs.desc.o_y << ','
				<< obs.desc.o_z << ','
				<< obs.desc.o_roll << ','
				<< obs.desc.o_pitch << ','
				<< obs.desc.o_yaw << ','
				<< obs.desc.x_size << ','
				<< obs.desc.y_size << ','
				<< obs.desc.z_size << ','
				<< obs.desc.mass << ','
				<< obs.desc.mu << ','
				<< movable << '\n';
	}

	movable = "True";
	for (const auto& a: m_agents)
	{
		auto agent_obj = a->GetObject();
		State loc = a->InitState().state;
		if (loc.empty()) {
			loc = { agent_obj->desc.o_x, agent_obj->desc.o_y };
		}
		DATA << agent_obj->desc.id << ','
			<< agent_obj->Shape() << ','
			<< agent_obj->desc.type << ','
			<< loc.at(0) << ','
			<< loc.at(1) << ','
			<< agent_obj->desc.o_z << ','
			<< agent_obj->desc.o_roll << ','
			<< agent_obj->desc.o_pitch << ','
			<< agent_obj->desc.o_yaw << ','
			<< agent_obj->desc.x_size << ','
			<< agent_obj->desc.y_size << ','
			<< agent_obj->desc.z_size << ','
			<< agent_obj->desc.mass << ','
			<< agent_obj->desc.mu << ','
			<< movable << '\n';
	}

	// write solution trajs
	DATA << 'T' << '\n';
	o = m_agents.size();
	DATA << o << '\n';

	for (const auto& a: m_agents)
	{
		auto agent_obj = a->GetObject();
		auto move = a->SolveTraj();
		DATA << agent_obj->desc.id << '\n';
		DATA << move->size() << '\n';
		for (const auto& s: *move) {
			DATA << s.state.at(0) << ',' << s.state.at(1) << '\n';
		}
	}

	DATA << "SOLUTION" << '\n';
	DATA << m_rearrangements.size() << '\n';
	for (size_t i = 0; i < m_rearrangements.size(); ++i)
	{
		auto &action_params = m_actions.at(i);
		int type = -1, pick_at = -1, place_at = -1, oid = -1;
		type = static_cast<int>(action_params._type);

		if (i < m_rearrangements.size() - 1)
		{
			// if any rearrangement traj execuction failed, m_violation == 1
			if (action_params._type == MAMOActionType::PICKPLACE)
			{
				pick_at = action_params._params[0];
				place_at = action_params._params[1];
				oid = action_params._oid;
			}
		}
		else
		{
			pick_at = action_params._params[0];
			oid = m_ooi->GetID();
		}

		DATA << "TRAJECTORY" << '\n';
		DATA << type << ',' << pick_at << ',' << place_at << ',' << oid << '\n';
		DATA << m_rearrangements.at(i).points.size() << '\n';

		for (const auto& p: m_rearrangements.at(i).points)
		{
			DATA 	<< p.positions[0] << ','
					<< p.positions[1] << ','
					<< p.positions[2] << ','
					<< p.positions[3] << ','
					<< p.positions[4] << ','
					<< p.positions[5] << ','
					<< p.positions[6] << ','

					// // For PR2 experiments
					// << p.velocities[0] << ','
					// << p.velocities[1] << ','
					// << p.velocities[2] << ','
					// << p.velocities[3] << ','
					// << p.velocities[4] << ','
					// << p.velocities[5] << ','
					// << p.velocities[6] << ','
					// << p.accelerations[0] << ','
					// << p.accelerations[1] << ','
					// << p.accelerations[2] << ','
					// << p.accelerations[3] << ','
					// << p.accelerations[4] << ','
					// << p.accelerations[5] << ','
					// << p.accelerations[6] << ','

					<< p.time_from_start.toSec() << '\n';
		}
	}

	if (!m_ngr.empty())
	{
		DATA << "NGR" << '\n';
		DATA << m_ngr.size() << '\n';
		for (const auto& p: m_ngr) {
			DATA 	<< p.at(0) << ','
					<< p.at(1) << '\n';
		}
	}

	DATA.close();
}

bool Planner::Execute()
{
	ROS_WARN("moveToStartState");
	moveToStartState();

	for (const auto& traj: m_rearrangements) {
		ROS_WARN("executeTraj PUSH");
		executeTraj(traj, false);
	}

	ROS_WARN("executeTraj GRASP + EXTRACT");
	executeTraj(m_exec);
}

bool Planner::executeTraj(const trajectory_msgs::JointTrajectory& traj, bool grasping)
{
	ROS_WARN("CloseGripper");
	m_controller->CloseGripper();
	int FIX_GRASP_AT = 0;

	trajectory_msgs::JointTrajectory traj_piece;
	traj_piece.header = traj.header;
	traj_piece.joint_names = traj.joint_names;

	traj_piece.points.clear();
	if (!grasping) {
		traj_piece.points.insert(traj_piece.points.begin(), traj.points.begin(), traj.points.end());
	}
	else {
		traj_piece.points.insert(traj_piece.points.begin(), traj.points.begin(), traj.points.begin() + FIX_GRASP_AT);
	}

	ROS_WARN("ExecArmTrajectory");
	m_controller->ExecArmTrajectory(m_controller->PR2TrajFromMsg(traj_piece));
	while (!m_controller->GetArmState().isDone() && ros::ok())
	{
		usleep(50000);
	}

	if (!grasping) {
		return true;
	}

	ROS_WARN("OpenGripper");
	if (!m_controller->OpenGripper()) {
		return false;
	}

	traj_piece.points.clear();
	traj_piece.points.insert(traj_piece.points.begin(), traj.points.begin() + FIX_GRASP_AT, traj.points.begin() + FIX_GRASP_AT + 1);
	// auto skip = traj_piece.points.begin()->time_from_start;
	// traj_piece.points.begin()->time_from_start -= skip;
	// for (auto itr = traj_piece.points.begin(); itr != traj_piece.points.end(); ++itr) {
	// 	itr->time_from_start += ros::Duration(2);
	// }
	ROS_WARN("ExecArmTrajectory");
	m_controller->ExecArmTrajectory(m_controller->PR2TrajFromMsg(traj_piece));
	while (!m_controller->GetArmState().isDone() && ros::ok())
	{
		usleep(50000);
	}

	ROS_WARN("CloseGripper");
	m_controller->CloseGripper();

	traj_piece.points.clear();
	traj_piece.points.insert(traj_piece.points.begin(), traj.points.begin() + FIX_GRASP_AT + 1, traj.points.end());
	// skip = traj_piece.points.begin()->time_from_start;
	// traj_piece.points.begin()->time_from_start -= skip;
	// for (auto itr = traj_piece.points.begin(); itr != traj_piece.points.end(); ++itr) {
	// 	itr->time_from_start += ros::Duration(2);
	// }
	ROS_WARN("ExecArmTrajectory");
	m_controller->ExecArmTrajectory(m_controller->PR2TrajFromMsg(traj_piece));
	while (!m_controller->GetArmState().isDone() && ros::ok())
	{
		usleep(50000);
	}

	return true;
}

void Planner::moveToStartState()
{
	auto start_state = m_robot->GetStartState();
	auto it = std::find(start_state->joint_state.name.begin(), start_state->joint_state.name.end(), "torso_lift_joint");
	// double torso_start = start_state->joint_state.position.at(it - start_state->joint_state.name.begin());
	// ROS_WARN("RaiseTorso");
	// m_controller->RaiseTorso(torso_start);
	// while (!m_controller->GetTorsoState().isDone() && ros::ok()) {
	// 	usleep(50000);
	// }
	// ROS_WARN("RaiseTorso DONE");

	// m_controller->MoveLeftArmToRest();
	// while (!m_controller->GetArmState(true).isDone() && ros::ok()) {
	// 	usleep(50000);
	// }

	ROS_WARN("MoveToStartState");
	m_controller->MoveToStartState(start_state->joint_state);
	ROS_WARN("sent command to MoveToStartState");
	while (!m_controller->GetArmState().isDone() && ros::ok()) {
		usleep(50000);
	}
	ROS_WARN("MoveToStartState DONE");
}

} // namespace clutter
