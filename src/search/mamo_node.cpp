#include <pushplan/search/mamo_node.hpp>
#include <pushplan/search/planner.hpp>
#include <pushplan/utils/discretisation.hpp>

#include <algorithm>

namespace clutter
{

void MAMONode::InitAgents(
	const std::vector<std::shared_ptr<Agent> >& agents,
	const comms::ObjectsPoses& all_objects)
{
	m_all_objects = all_objects;
	for (size_t i = 0; i < agents.size(); ++i) {
		this->addAgent(agents.at(i), i);
	}
}

std::vector<double>* MAMONode::GetCurrentStartState()
{
	if (m_oidx != -1 && m_aidx != -1)
	{
		// we successfully validated an action to get to this node
		// so we just return the last state on the trajectory for that action
		return &(m_robot_traj.points.back().positions);
	}

	if (m_parent == nullptr) {
		return nullptr;
	}

	return m_parent->GetCurrentStartState();
}

bool MAMONode::RunMAPF()
{
	for (size_t i = 0; i < m_agents.size(); ++i)
	{
		assert(m_agents.at(i)->GetID() == m_object_states.at(i).id());
		m_agents.at(i)->SetObjectPose(m_object_states.at(i).cont_pose());
		m_agents.at(i)->Init();

		m_agents.at(i)->ResetInvalidPushes(
			m_planner->GetGloballyInvalidPushes(),
			m_planner->GetLocallyInvalidPushes(this->GetConstraintHash(), m_agents.at(i)->GetID()));
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

	return result;
}

void MAMONode::GetSuccs(
	std::vector<std::pair<int, int> > *succ_object_centric_actions,
	std::vector<comms::ObjectsPoses> *succ_objects,
	std::vector<trajectory_msgs::JointTrajectory> *succ_trajs,
	std::vector<std::tuple<State, State, int> > *debug_pushes)
{
	// if (!m_children.empty())
	// {
	// 	// do something else
	// 	m_successful_pushes.back();
	// 	m_successful_pushes.pop_back();
	// }

	bool duplicate_successor = false;
	std::vector<std::tuple<State, State, int> > duplicate_successor_debug_pushes;
	for (size_t i = 0; i < m_mapf_solution.size(); ++i)
	{
		const auto& moved = m_mapf_solution.at(i);
		// if (std::find(m_relevant_ids.begin(), m_relevant_ids.end(), m_agents.at(m_agent_map[moved.first])->GetID()) == m_relevant_ids.end())
		// {
		// 	// if something the robot cannot currently reach moved in the MAPF solution,
		// 	// ignore and move on, i.e. no successor generated since the scene cannot and should not change
		// 	continue;
		// }
		if (moved.second.size() == 1 || moved.second.front().coord == moved.second.back().coord) {
			continue;
		}

		// get push location
		std::vector<double> push;
		m_agents.at(m_agent_map[moved.first])->SetSolveTraj(moved.second);
		m_agents.at(m_agent_map[moved.first])->GetSE2Push(push);

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

		// plan to push location
		// m_robot->PlanPush creates the planner internally, because it might
		// change KDL chain during the process
		comms::ObjectsPoses result;
		int push_failure;
		std::tuple<State, State, int> debug_push;
		if (m_robot->PlanPush(this->GetCurrentStartState(), m_agents.at(m_agent_map[moved.first]).get(), push, movable_obstacles, m_all_objects, 1.0, result, push_failure, debug_push))
		{
			// valid push found!
			succ_object_centric_actions->emplace_back(moved.first, 0); // TODO: currently only one aidx
			succ_objects->push_back(std::move(result));
			succ_trajs->push_back(m_robot->GetLastPlan());
			debug_pushes->push_back(std::move(debug_push));
			// m_successful_pushes.push_back(std::make_pair(moved.first, moved.second.back().coord));
		}
		else
		{
			// SMPL_INFO("Tried pushing object %d. Return value = %d", moved.first, push_failure);
			switch (push_failure)
			{
				case 3: // SMPL_ERROR("Inverse dynamics failed to reach push end."); break;
				case 4: // SMPL_ERROR("Inverse kinematics/dynamics failed (joint limits likely)."); break;
				case 5: // SMPL_ERROR("Inverse kinematics hit static obstacle."); break;
				{
					m_planner->AddGloballyInvalidPush(std::make_pair(moved.second.front().coord, moved.second.back().coord));
					break;
				}
				case 1: // SMPL_ERROR("Push start inside object."); break;
				case 2: // SMPL_ERROR("Failed to reach push start."); break;
				case 6: // SMPL_ERROR("Push action did not collide with intended object."); break;
				case 0: // SMPL_ERROR("Valid push computed! Failed in simulation."); break;
				{
					m_planner->AddLocallyInvalidPush(
						this->GetConstraintHash(),
						moved.first,
						moved.second.back().coord);
					break;
				}
				// case -1: SMPL_INFO("Push succeeded in simulation!"); break;
				default: SMPL_WARN("Unknown push failure cause.");
			}
			duplicate_successor_debug_pushes.push_back(std::move(debug_push));
			if (!duplicate_successor) {
				duplicate_successor = true;
			}
		}
	}

	if (duplicate_successor)
	{
		trajectory_msgs::JointTrajectory dummy_traj;
		succ_object_centric_actions->emplace_back(-1, -1);
		succ_objects->push_back(m_all_objects);
		succ_trajs->push_back(dummy_traj);
		debug_pushes->insert(debug_pushes->end(),
					duplicate_successor_debug_pushes.begin(), duplicate_successor_debug_pushes.end());
	}
}

unsigned int MAMONode::ComputeMAMOPriority()
{
	unsigned int percent_ngr, percent_objs, num_objs;
	computePriorityFactors(percent_ngr, percent_objs, num_objs);
	return (percent_objs/num_objs) + percent_ngr;
}

size_t MAMONode::GetConstraintHash() const
{
	if (m_hash_set_C) {
		return m_hash_C;
	}

	size_t hash_val = 0;

	for (const auto &object_state : kobject_states())
	{
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
	}

	return hash_val;
}

size_t MAMONode::GetSearchHash() const
{
	if (m_hash_set_S) {
		return m_hash_S;
	}

	size_t hash_val = GetConstraintHash();
	for (const auto &movable : m_mapf_solution)
	{
		hash_val ^= std::hash<int>()(movable.first);
		for (const auto &wp : movable.second) {
			boost::hash_combine(hash_val, boost::hash_range(wp.coord.begin(), wp.coord.begin() + 2));
		}
	}

	return hash_val;
}

void MAMONode::SetConstraintHash(const size_t& hash_val)
{
	if (!m_hash_set_C)
	{
		m_hash_C = hash_val;
		m_hash_set_C = true;
	}
	else {
		assert(m_hash_C == hash_val);
	}
}

void MAMONode::SetSearchHash(const size_t& hash_val)
{
	if (!m_hash_set_S)
	{
		m_hash_S = hash_val;
		m_hash_set_S = true;
	}
	else {
		assert(m_hash_S == hash_val);
	}
}

void MAMONode::SetParent(MAMONode *parent)
{
	m_parent = parent;
}

void MAMONode::SetRobotTrajectory(const trajectory_msgs::JointTrajectory& robot_traj)
{
	m_robot_traj = robot_traj;
}

void MAMONode::AddDebugPush(const std::tuple<State, State, int>& debug_push)
{
	m_debug_pushes.push_back(debug_push);
	m_have_debug_pushes = true;
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

void MAMONode::SetEdgeTo(int oidx, int aidx)
{
	m_oidx = oidx;
	m_aidx = aidx;
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
	if (m_agents.size() != m_object_states.size()) {
		throw std::runtime_error("Objects and object states size mismatch!");
	}
	return m_object_states.size();
}

const std::vector<ObjectState>& MAMONode::kobject_states() const
{
	return m_object_states;
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

std::pair<int, int> MAMONode::object_centric_action() const
{
	return std::make_pair(m_oidx, m_aidx);
}

const std::vector<std::pair<int, Trajectory> >& MAMONode::kmapf_soln() const
{
	return m_mapf_solution;
}

bool MAMONode::has_traj() const
{
	return !m_robot_traj.points.empty();
}

void MAMONode::addAgent(
	const std::shared_ptr<Agent>& agent,
	const size_t& pidx)
{
	m_agents.push_back(agent);
	m_agent_map[agent->GetID()] = m_agents.size() - 1;

	assert(m_agents.back()->GetID() == m_all_objects.poses.at(pidx).id);
	m_agents.back()->SetObjectPose(m_all_objects.poses.at(pidx).xyz, m_all_objects.poses.at(pidx).rpy);

	auto o = m_agents.back()->GetObject();
	ContPose pose(o->desc.o_x, o->desc.o_y, o->desc.o_z, o->desc.o_roll, o->desc.o_pitch, o->desc.o_yaw);
	m_object_states.emplace_back(o->desc.id, o->Symmetric(), pose);
}

void MAMONode::identifyRelevantMovables()
{
	// may potentially need to SetObjectPose for agents
	m_relevant_ids.clear();
	m_robot->IdentifyReachableMovables(m_agents, m_relevant_ids);
}

void MAMONode::computePriorityFactors(
	unsigned int &percent_ngr, unsigned int &percent_objs, unsigned int &num_objs)
{
	percent_ngr = 0;
	percent_objs = 0;
	num_objs = 0;

	const auto& ngr = m_planner->GetNGR();
	float ngr_size = float(ngr.size());
	std::set<Coord, coord_compare> agent_voxels;
	std::vector<Coord> intersection;
	for (size_t i = 0; i < m_mapf_solution.size(); ++i)
	{
		const auto& moved = m_mapf_solution.at(i);
		if (moved.second.size() == 1 || moved.second.front().coord == moved.second.back().coord) {
			continue;
		}

		m_agents.at(m_agent_map[moved.first])->GetVoxels(
					m_object_states.at(m_agent_map[moved.first]).cont_pose(),
					agent_voxels);

		intersection.clear();
		std::set_intersection(
				ngr.begin(), ngr.end(),
				agent_voxels.begin(), agent_voxels.end(),
				std::back_inserter(intersection));

		if (intersection.size() > 0)
		{
			++num_objs;
			percent_ngr += (intersection.size()/ngr_size) * 100;
			percent_objs += (intersection.size()/agent_voxels.size()) * 100;
		}
	}
}

void MAMONode::resetAgents()
{
	m_agents.clear();
	m_object_states.clear();
}

void MAMONode::SaveNode(unsigned int my_id,	unsigned int parent_id)
{
	// for (int tidx = 0; tidx < 1; tidx += 1)
	{
		std::string filename(__FILE__);
		auto found = filename.find_last_of("/\\");
		filename = filename.substr(0, found + 1) + "../../dat/txt/";

		std::stringstream ss;
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

		auto t = std::time(nullptr);
		auto tm = *std::localtime(&t);
		ss << std::put_time(&tm, "%d-%m-%Y-%H-%M-%S");

		std::string s = ss.str();

		filename += s;
		filename += ".txt";

		std::ofstream DATA;
		DATA.open(filename, std::ofstream::out);

		DATA << 'O' << '\n';
		int o = m_cc->NumObstacles() + m_agents.size();
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
			auto agent_pose = m_all_objects.poses.at(oidx);
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

		if (!m_debug_pushes.empty())
		{
			DATA << "PUSHES" << '\n';
			DATA << m_debug_pushes.size() << '\n';

			State push_start, push_end;
			int push_result;
			for (const auto& debug_push: m_debug_pushes)
			{
				std::tie(push_start, push_end, push_result) = debug_push;
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
			auto invalid_l = m_planner->GetLocallyInvalidPushes(this->GetConstraintHash(), m_agents.at(oidx)->GetID());
			DATA << m_agents.at(oidx)->GetID() << '\n';
			if (invalid_l != nullptr)
			{
				DATA << invalid_l->size() << '\n';
				for (const auto& c: *invalid_l)
				{
					State s;
					s.push_back(DiscretisationManager::DiscXToContX(c.at(0)));
					s.push_back(DiscretisationManager::DiscYToContY(c.at(1)));
					DATA << s.at(0) << ',' << s.at(1) << '\n';
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
