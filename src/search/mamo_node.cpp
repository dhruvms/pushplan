#include <pushplan/search/mamo_node.hpp>
#include <pushplan/search/planner.hpp>
#include <pushplan/utils/discretisation.hpp>
#include <pushplan/utils/helpers.hpp>

#include <algorithm>

namespace clutter
{

void MAMONode::InitAgents(
	const std::vector<std::shared_ptr<Agent> > &agents,
	const comms::ObjectsPoses& all_objects,
	const std::vector<int> &reachable_ids)
{
	m_all_objects = all_objects;
	for (size_t i = 0; i < agents.size(); ++i)
	{
		if (std::find(reachable_ids.begin(), reachable_ids.end(), agents.at(i)->GetID()) != reachable_ids.end()) {
			this->addAgent(agents.at(i), i);
		}
		else
		{
			// agents.at(i)->SetObjectPose(m_all_objects.poses.at(i).xyz, m_all_objects.poses.at(i).rpy);

			auto o = agents.at(i)->GetObject();
			ContPose pose(o->desc.o_x, o->desc.o_y, o->desc.o_z, o->desc.o_roll, o->desc.o_pitch, o->desc.o_yaw);
			m_all_object_states.emplace_back(o->desc.id, o->Symmetric(), pose);
		}
	}
}

std::vector<double>* MAMONode::GetCurrentStartState()
{
	if (m_action_to_me._oid != -1 && m_action_to_me._type != MAMOActionType::DUMMY)
	{
		// we successfully validated an action to get to this node
		// so we just return the last state on the trajectory for that action
		return &(m_robot_traj.points.back().positions);
	}

	if (m_parent == nullptr) {
		return m_robot->GetHomeState();
	}

	return m_parent->GetCurrentStartState();
}

bool MAMONode::RunMAPF()
{
	if (!m_new_constraints && m_successful_rearranges.empty()) {
		return false;
	}

	for (size_t i = 0; i < m_agents.size(); ++i)
	{
		assert(m_agents.at(i)->GetID() == m_object_states.at(i).id());
		m_agents.at(i)->SetObjectPose(m_object_states.at(i).cont_pose());
		m_agents.at(i)->Init();

		m_agents.at(i)->ResetInvalidPushes(
			m_planner->GetGloballyInvalidPushes(),
			m_planner->GetLocallyInvalidPushes(this->GetObjectsHash(), m_agents.at(i)->GetID()));
		// m_agents.at(i)->UseLearnedCost(!m_expanded_once);
	}

	if (!m_new_constraints)
	{
		const auto &valid_push = m_successful_rearranges.back();

		// add all past hallucinated constraints for this object
		for (const auto& p: m_successful_rearranges_invalidated)
		{
			if (p.first == valid_push.first) {
				m_agents.at(m_agent_map[p.first])->AddHallucinatedConstraint(p.second);
			}
		}

		// add latest hallucinated constraint
		m_agents.at(m_agent_map[valid_push.first])->AddHallucinatedConstraint(valid_push.second);
		m_successful_rearranges_invalidated.push_back(m_successful_rearranges.back());
		m_successful_rearranges.pop_back();
	}

	// set/update/init necessary components
	m_cc->ReinitMovableCC(m_agents);
	m_cbs->Reset();
	m_cbs->AddObjects(m_agents);

	bool result = m_cbs->Solve();
	if (result)
	{
		m_mapf_solution = m_cbs->GetSolution()->m_solution;
		// identifyRelevantMovables();
		for (size_t i = 0; i < m_agents.size(); ++i) {
			m_agents.at(i)->ResetObject(); // in preparation for push evaluation
		}
	}
	m_new_constraints = false;

	return result;
}

void MAMONode::ResetConstraints()
{
	m_new_constraints = true;
	m_mapf_solution.clear();
}

bool MAMONode::tryPickPlace(
	std::vector<MAMOAction> *succ_object_centric_actions,
	std::vector<comms::ObjectsPoses> *succ_objects,
	std::vector<trajectory_msgs::JointTrajectory> *succ_trajs,
	std::vector<std::tuple<State, State, int> > *debug_actions,
	std::vector<std::tuple<State, State, int> > *invalid_action_samples,
	const std::vector<Object*> &movable_obstacles, size_t aid)
{
	const auto& moved = m_mapf_solution.at(aid);

	comms::ObjectsPoses result;
	int pick_at, place_at;
	double plan_time = 0.0, sim_time = 0.0;
	std::tuple<State, State, int> debug_action;
	if (m_robot->PlanPickPlace(this->GetCurrentStartState(), m_agents.at(m_agent_map[moved.first]).get(), movable_obstacles, m_all_objects, result, pick_at, place_at, plan_time, sim_time, debug_action))
	{
		// pick-and-place succeeded!
		MAMOAction action(MAMOActionType::PICKPLACE, moved.first);
		action._params.push_back(pick_at);
		action._params.push_back(place_at);

		succ_object_centric_actions->push_back(action);
		succ_objects->push_back(std::move(result));
		succ_trajs->push_back(m_robot->GetLastPlan());
		debug_actions->push_back(std::move(debug_action));

		m_successful_rearranges.push_back(std::make_pair(moved.first, moved.second.back().coord));
		return true;
	}
	invalid_action_samples->push_back(std::move(debug_action));
	return false;
}

bool MAMONode::tryPush(
	std::vector<MAMOAction> *succ_object_centric_actions,
	std::vector<comms::ObjectsPoses> *succ_objects,
	std::vector<trajectory_msgs::JointTrajectory> *succ_trajs,
	std::vector<std::tuple<State, State, int> > *debug_actions,
	std::vector<std::tuple<State, State, int> > *invalid_action_samples,
	const std::vector<Object*> &movable_obstacles, size_t aid, double *sim_time)
{
	const auto& moved = m_mapf_solution.at(aid);

	// check if we have seen this push before
	int samples = m_agents.at(m_agent_map[moved.first])->InvalidPushCount(moved.second.back().coord);
	if (samples >= SAMPLES) {
		return true; // known invalid push, do not compute
	}
	else if (samples == 0) {
		samples = SAMPLES; // never before seen push, must compute
	}
	else {
		samples = 1; // seen valid push before, lookup from DB
		// samples = SAMPLES - samples;
	}

	// get push location
	std::vector<double> push;
	m_agents.at(m_agent_map[moved.first])->SetSolveTraj(moved.second);
	m_agents.at(m_agent_map[moved.first])->GetSE2Push(push);
	SMPL_INFO("Try pushing object %d", moved.first);

	int p = 0;
	bool globally_invalid = false;
	bool success = true;
	for (; p < samples; ++p)
	{
		// plan to push location
		// m_robot->PlanPush creates the planner internally, because it might
		// change KDL chain during the process
		comms::ObjectsPoses result;
		PushResult push_failure;
		std::tuple<State, State, int> debug_action;
		double sim_time_push = 0.0;
		if (m_robot->PlanPush(this->GetCurrentStartState(), m_agents.at(m_agent_map[moved.first]).get(), push, movable_obstacles, m_all_objects, result, push_failure, debug_action, sim_time_push))
		{
			*sim_time += sim_time_push;

			// valid push found!
			MAMOAction action(MAMOActionType::PUSH, moved.first);
			action._params.clear();

			succ_object_centric_actions->push_back(action); // TODO: currently only one aidx
			succ_objects->push_back(std::move(result));
			succ_trajs->push_back(m_robot->GetLastPlan());
			debug_actions->push_back(std::move(debug_action));
			if (samples == SAMPLES) {
				m_successful_rearranges.push_back(std::make_pair(moved.first, moved.second.back().coord));
			}
			break;
		}
		else
		{
			*sim_time += sim_time_push;

			// assert(samples == SAMPLES);
			// SMPL_INFO("Tried pushing object %d. Return value = %d", moved.first, push_failure);
			switch (push_failure)
			{
				case PushResult::IK_CATCHALL_FAIL : // SMPL_ERROR("Something went wrong in IK."); break;
				case PushResult::IK_JOINT_LIMIT : // SMPL_ERROR("IK hit joint limits."); break;
				case PushResult::IK_OBSTACLE : // SMPL_ERROR("IK hit static obstacle."); break;
				{
					m_planner->AddGloballyInvalidPush(std::make_pair(moved.second.front().coord, moved.second.back().coord));
					globally_invalid = true;
					break;
				}
				// case PushResult::START_INSIDE_OBSTACLE : // SMPL_ERROR("Push start inside object."); break;
				// case PushResult::START_UNREACHABLE : // SMPL_ERROR("Failed to reach push start."); break;
				// case PushResult::NO_OOI_COLLISION : // SMPL_ERROR("Push action did not collide with intended object."); break;
				// case PushResult::FAIL_IN_SIM : // SMPL_ERROR("Valid push failed in simulation!"); break;
				// case PushResult::SUCCESS_IN_SIM : // SMPL_INFO("Push succeeded in simulation!"); break;
				// default: // SMPL_WARN("Unknown push failure cause.");
			}
			invalid_action_samples->push_back(std::move(debug_action));
			if (globally_invalid) {
				break;
			}
		}
	}

	if (samples == SAMPLES) // this was a new push I was considering
	{
		if (globally_invalid) {
			p = SAMPLES * 2; // automatic maximum penalty
		}
		if (p >= SAMPLES) {
			success = false; // failed to find a push, must add new constraint to this state
		}
		if (p > 0)
		{
			// treat this action to have p invalid samples
			m_planner->AddLocallyInvalidPush(
				this->GetObjectsHash(),
				moved.first,
				moved.second.back().coord,
				p);
		}
	}

	return success;
}

void MAMONode::GetSuccs(
	std::vector<MAMOAction> *succ_object_centric_actions,
	std::vector<comms::ObjectsPoses> *succ_objects,
	std::vector<trajectory_msgs::JointTrajectory> *succ_trajs,
	std::vector<std::tuple<State, State, int> > *debug_actions,
	bool *close,
	double *mapf_time, double *get_succs_time, double *sim_time)
{
	*close = false;
	*mapf_time = GetTime();
	if (!this->RunMAPF() && m_successful_rearranges.empty())
	{
		*mapf_time = GetTime() - *mapf_time;
		// I have no more possible successors, please CLOSE me
		*close = true;
		return;
	}
	*mapf_time = GetTime() - *mapf_time;

	std::vector<std::tuple<State, State, int> > invalid_action_samples;
	*get_succs_time = GetTime();
	bool duplicate_successor = false;
	bool something_moved = false;
	for (size_t i = 0; i < m_mapf_solution.size(); ++i)
	{
		const auto& moved = m_mapf_solution.at(i);
		// if (std::find(m_relevant_ids.begin(), m_relevant_ids.end(), m_agents.at(m_agent_map[moved.first])->GetID()) == m_relevant_ids.end())
		// {
		// 	// if something the robot cannot currently reach moved in the MAPF solution,
		// 	// ignore and move on, i.e. no successor generated since the scene cannot and should not change
		// 	continue;
		// }

		// agent does not move in MAPF solution
		if (moved.second.size() == 1 || moved.second.front().coord == moved.second.back().coord) {
			continue;
		}
		something_moved = true;

		// other movables to be considered as obstacles
		std::vector<Object*> movable_obstacles;
		for (size_t i = 0; i < m_agents.size(); ++i)
		{
			m_agents.at(i)->SetObjectPose(m_object_states.at(i).cont_pose());
			if (m_agents.at(i)->GetID() == moved.first) {
				continue; // selected object cannot be obstacle
			}
			movable_obstacles.push_back(m_agents.at(i)->GetObject());
		}

		bool pickplace_result, push_result, graspable = m_agents.at(m_agent_map[moved.first])->GetObject()->Graspable();
		if (graspable) {
			pickplace_result = tryPickPlace(succ_object_centric_actions, succ_objects, succ_trajs, debug_actions, &invalid_action_samples, movable_obstacles, i);
		}
		else
		{
			pickplace_result = false;
			// SMPL_INFO("Object %d is NOT graspable!", moved.first);
		}

		push_result = tryPush(succ_object_centric_actions, succ_objects, succ_trajs, debug_actions, &invalid_action_samples, movable_obstacles, i, sim_time);
		// create duplicate_successor state if any object could not be pushed
		duplicate_successor = duplicate_successor || !push_result;

		if (graspable)
		{
			if (pickplace_result && push_result) {
				m_successful_rearranges.pop_back();
			}
		}
	}

	if (duplicate_successor)
	{
		MAMOAction action(MAMOActionType::DUMMY, -1);
		action._params.clear();
		succ_object_centric_actions->push_back(action);
		succ_objects->push_back(m_all_objects);
		trajectory_msgs::JointTrajectory dummy_traj;
		succ_trajs->push_back(dummy_traj);
		debug_actions->insert(debug_actions->end(),
					invalid_action_samples.begin(), invalid_action_samples.end());
		m_new_constraints = true;
	}
	*get_succs_time = GetTime() - *get_succs_time;

	if (!something_moved) {
		SMPL_WARN("Nothing moved in the MAPF solution, so why did we expand the state?");
	}

	if (!m_expanded_once) {
		m_expanded_once = true;
	}
}

unsigned int MAMONode::ComputeMAMOPriorityOrig()
{
	ComputePriorityFactors();

	if (m_num_objs == 0) {
		return 0;
	}
	return 100 * (m_percent_objs * m_num_objs + m_percent_ngr);
}

void MAMONode::ComputePriorityFactors()
{
	m_percent_ngr = 0.0;
	m_percent_objs = 0.0;
	m_num_objs = 0;

	const auto& ngr = m_planner->GetNGR();
	float ngr_size = float(ngr.size());
	std::set<Coord, coord_compare> agent_voxels;
	std::vector<Coord> intersection;
	for (size_t i = 0; i < m_object_states.size(); ++i)
	{
		m_agents.at(m_agent_map[m_object_states.at(i).id()])->GetVoxels(
					m_object_states.at(i).cont_pose(),
					agent_voxels);

		intersection.clear();
		std::set_intersection(
				ngr.begin(), ngr.end(),
				agent_voxels.begin(), agent_voxels.end(),
				std::back_inserter(intersection));

		if (intersection.size() > 0)
		{
			++m_num_objs;
			m_percent_ngr += (intersection.size()/ngr_size);

			double percent_obj = (intersection.size()/double(agent_voxels.size()));
			auto obj = m_agents.at(m_agent_map[m_object_states.at(i).id()])->GetObject();
			m_percent_objs += percent_obj;

			std::vector<double> priority_props = {percent_obj, obj->Height(), obj->desc.mass, obj->desc.mu};
			m_obj_priority_data.push_back(std::move(priority_props));
		}
	}
}

size_t MAMONode::GetObjectsHash() const
{
	if (m_hash_set_l1) {
		return m_hash_l1;
	}

	size_t hash_val = 0;

	for (const auto &object_state : kall_object_states()) {
		object_state.hash_state(hash_val);
	}

	return hash_val;
}

void MAMONode::SetObjectsHash(const size_t& hash_val)
{
	if (!m_hash_set_l1)
	{
		m_hash_l1 = hash_val;
		m_hash_set_l1 = true;
	}
	else {
		assert(m_hash_l1 == hash_val);
	}
}

// size_t MAMONode::GetObjectsMAPFHash() const
// {
// 	if (m_hash_set_l2) {
// 		return m_hash_l2;
// 	}

// 	size_t hash_val = GetObjectsHash();
// 	for (const auto &movable : m_mapf_solution)
// 	{
// 		boost::hash_combine(hash_val, movable.first);
// 		for (const auto &wp : movable.second) {
// 			boost::hash_combine(hash_val, boost::hash_range(wp.coord.begin(), wp.coord.begin() + 2));
// 		}
// 	}

// 	return hash_val;
// }

// void MAMONode::SetObjectsMAPFHash(const size_t& hash_val)
// {
// 	if (!m_hash_set_l2)
// 	{
// 		m_hash_l2 = hash_val;
// 		m_hash_set_l2 = true;
// 	}
// 	else {
// 		assert(m_hash_l2 == hash_val);
// 	}
// }

// size_t MAMONode::GetObjectsConstraintsHash() const
// {
// 	if (m_hash_set_l2) {
// 		return m_hash_l2;
// 	}

// 	size_t hash_val = GetObjectsHash();
// 	for (const auto &movable : m_mapf_constraints)
// 	{
// 		boost::hash_combine(hash_val, movable.first); // constrained object
// 		for (const auto &c : movable.second)
// 		{
// 			boost::hash_combine(hash_val, boost::hash_range(c.first.begin(), c.first.begin() + 2)); // constrained coord
// 			boost::hash_combine(hash_val, c.second); // number of samples
// 		}
// 	}

// 	return hash_val;
// }

// void MAMONode::SetObjectsConstraintsHash(const size_t& hash_val)
// {
// 	if (!m_hash_set_l2)
// 	{
// 		m_hash_l2 = hash_val;
// 		m_hash_set_l2 = true;
// 	}
// 	else {
// 		assert(m_hash_l2 == hash_val);
// 	}
// }

void MAMONode::SetParent(MAMONode *parent)
{
	m_parent = parent;
}

void MAMONode::SetRobotTrajectory(const trajectory_msgs::JointTrajectory& robot_traj)
{
	m_robot_traj = robot_traj;
}

void MAMONode::AddDebugAction(const std::tuple<State, State, int>& debug_action)
{
	m_debug_actions.push_back(debug_action);
}

void MAMONode::SetPlanner(Planner *planner)
{
	m_planner = planner;
}

void MAMONode::SetCBS(const std::shared_ptr<CBS>& cbs)
{
	m_cbs = cbs;
}

void MAMONode::SetCC(const std::shared_ptr<CollisionChecker>& cc)
{
	m_cc = cc;
}

void MAMONode::SetRobot(const std::shared_ptr<Robot>& robot)
{
	m_robot = robot;
}

void MAMONode::SetEdgeTo(const MAMOAction &action)
{
	m_action_to_me = action;
}

void MAMONode::AddChild(MAMONode* child)
{
	m_children.push_back(child);
}

void MAMONode::RemoveChild(MAMONode* child)
{
	m_children.erase(std::remove(m_children.begin(), m_children.end(), child), m_children.end());
}

size_t MAMONode::num_objects() const
{
	// if (m_agents.size() != m_object_states.size()) {
	// 	throw std::runtime_error("Objects and object states size mismatch!");
	// }
	return m_all_object_states.size();
}

const std::vector<ObjectState>& MAMONode::kobject_states() const
{
	return m_object_states;
}

const std::vector<ObjectState>& MAMONode::kall_object_states() const
{
	return m_all_object_states;
}

trajectory_msgs::JointTrajectory& MAMONode::robot_traj()
{
	return m_robot_traj;
}

const trajectory_msgs::JointTrajectory& MAMONode::krobot_traj() const
{
	return m_robot_traj;
}

const MAMONode* MAMONode::kparent() const
{
	return m_parent;
}

MAMONode* MAMONode::parent()
{
	return m_parent;
}

const std::vector<MAMONode*>& MAMONode::kchildren() const
{
	return m_children;
}

const MAMOAction& MAMONode::kobject_centric_action() const
{
	return m_action_to_me;
}

const std::vector<std::pair<int, Trajectory> >& MAMONode::kmapf_soln() const
{
	return m_mapf_solution;
}

bool MAMONode::has_traj() const
{
	return !m_robot_traj.points.empty();
}

bool MAMONode::has_mapf_soln() const
{
	return !m_mapf_solution.empty();
}

const float& MAMONode::percent_ngr() const
{
	return m_percent_ngr;
}

const std::vector<std::vector<double> >& MAMONode::obj_priority_data() const
{
	return m_obj_priority_data;
}

void MAMONode::addAgent(
	const std::shared_ptr<Agent>& agent,
	const size_t& pidx)
{
	m_agents.push_back(agent);
	m_agent_map[agent->GetID()] = m_agents.size() - 1;
	// SMPL_INFO("Agent ID %d added to MAMONode!", agent->GetID());

	assert(m_agents.back()->GetID() == m_all_objects.poses.at(pidx).id);
	// m_agents.back()->SetObjectPose(m_all_objects.poses.at(pidx).xyz, m_all_objects.poses.at(pidx).rpy);

	auto o = m_agents.back()->GetObject();
	ContPose pose(o->desc.o_x, o->desc.o_y, o->desc.o_z, o->desc.o_roll, o->desc.o_pitch, o->desc.o_yaw);
	m_object_states.emplace_back(o->desc.id, o->Symmetric(), pose);
	m_all_object_states.emplace_back(o->desc.id, o->Symmetric(), pose);
}

// void MAMONode::identifyRelevantMovables()
// {
// 	// may potentially need to SetObjectPose for agents
// 	m_relevant_ids.clear();
// 	m_robot->IdentifyReachableMovables(m_agents, m_relevant_ids);
// }

void MAMONode::resetAgents()
{
	m_agents.clear();
	m_object_states.clear();
	m_all_object_states.clear();
}

void MAMONode::SaveNode(unsigned int my_id, unsigned int parent_id, const std::string &suffix)
{
	// for (int tidx = 0; tidx < 1; tidx += 1)
	{
		std::string filename(__FILE__);
		auto found = filename.find_last_of("/\\");
		filename = filename.substr(0, found + 1) + "../../dat/txt/";

		std::stringstream ss;
		auto t = std::time(nullptr);
		auto tm = *std::localtime(&t);
		ss << std::put_time(&tm, "%d-%m-%Y-%H-%M-%S");
		ss << "_";
		// ss << std::setw(4) << std::setfill('0') << node->m_expand;
		// ss << "_";
		// ss << std::setw(4) << std::setfill('0') << node->m_depth;
		// ss << "_";
		// ss << std::setw(4) << std::setfill('0') << node->m_replanned;
		// ss << "_";
		ss << std::setw(6) << std::setfill('0') << m_planner->GetSceneID();
		ss << "_";
		ss << std::setw(6) << std::setfill('0') << my_id;
		ss << "_";
		ss << std::setw(6) << std::setfill('0') << parent_id;
		ss << "_";
		ss << suffix;

		std::string s = ss.str();

		filename += s;
		filename += ".txt";

		std::ofstream DATA;
		DATA.open(filename, std::ofstream::out);

		const auto &all_agents = m_planner->GetAllAgents();
		DATA << 'O' << '\n';
		int o = m_cc->NumObstacles() + all_agents.size();
		DATA << o << '\n';

		std::string movable;
		auto obstacles = m_cc->GetObstacles();
		int ooi_id = m_planner->GetOoIID();
		for (const auto& obs: *obstacles)
		{
			movable = obs.desc.movable ? "True" : "False";
			int oid = obs.desc.id == ooi_id ? 999 : obs.desc.id;
			DATA << oid << ','
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

		State loc;
		movable = "True";
		for (size_t oidx = 0; oidx < m_agents.size(); ++oidx)
		{
			auto agent_obs = m_agents[oidx]->GetObject();
			comms::ObjectPose agent_pose;
			for (const auto &o : m_all_objects.poses)
			{
				if (o.id == agent_obs->desc.id)
				{
					agent_pose = o;
					break;
				}
			}
			DATA << agent_obs->desc.id << ','
				<< agent_obs->Shape() << ','
				<< agent_obs->desc.type << ','
				<< agent_pose.xyz[0] << ','
				<< agent_pose.xyz[1] << ','
				<< agent_pose.xyz[2] << ','
				<< agent_pose.rpy[0] << ','
				<< agent_pose.rpy[1] << ','
				<< agent_pose.rpy[2] << ','
				<< agent_obs->desc.x_size << ','
				<< agent_obs->desc.y_size << ','
				<< agent_obs->desc.z_size << ','
				<< agent_obs->desc.mass << ','
				<< agent_obs->desc.mu << ','
				<< movable << '\n';
		}

		int xid = 998;
		for (size_t i = 0; i < all_agents.size(); ++i)
		{
			if (m_agent_map.find(all_agents.at(i)->GetID()) == m_agent_map.end())
			{
				auto agent_obs = all_agents.at(i)->GetObject();
				auto agent_pose = m_all_objects.poses.at(i);
				DATA << xid-- << ','
						<< agent_obs->Shape() << ','
						<< agent_obs->desc.type << ','
						<< agent_pose.xyz[0] << ','
						<< agent_pose.xyz[1] << ','
						<< agent_pose.xyz[2] << ','
						<< agent_pose.rpy[0] << ','
						<< agent_pose.rpy[1] << ','
						<< agent_pose.rpy[2] << ','
						<< agent_obs->desc.x_size << ','
						<< agent_obs->desc.y_size << ','
						<< agent_obs->desc.z_size << ','
						<< agent_obs->desc.mass << ','
						<< agent_obs->desc.mu << ','
						<< movable << '\n';
			}
		}

		// write solution trajs
		DATA << 'T' << '\n';
		o = m_mapf_solution.size();
		DATA << o << '\n';
		for (size_t oidx = 0; oidx < m_mapf_solution.size(); ++oidx)
		{
			auto moved = m_mapf_solution.at(oidx);
			DATA << moved.first << '\n';
			DATA << moved.second.size() << '\n';
			for (const auto& wp: moved.second) {
				DATA << wp.state.at(0) << ',' << wp.state.at(1) << '\n';
			}
		}

		if (!m_debug_actions.empty())
		{
			DATA << "PUSHES" << '\n';
			DATA << m_debug_actions.size() << '\n';

			State push_start, push_end;
			int push_result;
			for (const auto& debug_action: m_debug_actions)
			{
				std::tie(push_start, push_end, push_result) = debug_action;
				DATA 	<< push_start.at(0) << ','
						<< push_start.at(1) << ','
						<< push_end.at(0) << ','
						<< push_end.at(1) << ','
						<< push_result << '\n';
			}
		}

		// write invalid goals
		auto invalid_g = m_planner->GetGloballyInvalidPushes();
		DATA << "INVALID_G" << '\n';
		DATA << invalid_g->size() << '\n';
		for (size_t i = 0; i < invalid_g->size(); ++i)
		{
			State f, t;
			f.push_back(DiscretisationManager::DiscXToContX(invalid_g->at(i).first.at(0)));
			f.push_back(DiscretisationManager::DiscYToContY(invalid_g->at(i).first.at(1)));
			t.push_back(DiscretisationManager::DiscXToContX(invalid_g->at(i).second.at(0)));
			t.push_back(DiscretisationManager::DiscYToContY(invalid_g->at(i).second.at(1)));
			DATA << f.at(0) << ',' << f.at(1) << ','
					<< t.at(0) << ',' << t.at(1) << '\n';
		}

		DATA << "INVALID_L" << '\n';
		DATA << m_agents.size() << '\n';
		for (size_t oidx = 0; oidx < m_agents.size(); ++oidx)
		{
			auto invalid_l = m_planner->GetLocallyInvalidPushes(this->GetObjectsHash(), m_agents.at(oidx)->GetID());
			DATA << m_agents.at(oidx)->GetID() << '\n';
			if (invalid_l != nullptr)
			{
				DATA << invalid_l->size() << '\n';
				for (auto it = invalid_l->cbegin(); it != invalid_l->cend(); ++it)
				{
					State s;
					s.push_back(DiscretisationManager::DiscXToContX(it->first.at(0)));
					s.push_back(DiscretisationManager::DiscYToContY(it->first.at(1)));
					DATA << s.at(0) << ',' << s.at(1) << ',' << it->second << '\n';
				}
			}
			else {
				DATA << 0 << '\n';
			}
		}

		const auto& ngr = m_planner->GetNGR();
		DATA << "NGR" << '\n';
		DATA << ngr.size() << '\n';
		for (const auto& p: ngr) {
			DATA 	<< p.at(0) << ','
					<< p.at(1) << '\n';
		}

		DATA.close();
	}
}

} // namespace clutter
