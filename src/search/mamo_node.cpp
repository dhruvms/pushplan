#include <pushplan/search/mamo_node.hpp>
#include <pushplan/search/planner.hpp>

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

bool MAMONode::RunMAPF(
	unsigned int my_state_id,
	unsigned int parent_id)
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
		auto debug_push_ptr = m_have_debug_push ? &m_debug_push : nullptr;
		m_cbs->WriteLastSolution(debug_push_ptr, my_state_id, parent_id);
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
		m_agents.at(m_agent_map[moved.first])->GetSE2Push(push);

		// other movables to be considered as obstacles
		std::vector<Object*> movable_obstacles;
		for (const auto& a: m_agents)
		{
			if (a->GetID() == moved.first) {
				continue; // selected object cannot be obstacle
			}
			movable_obstacles.push_back(a->GetObject());
		}

		// plan to push location
		// m_robot->PlanPush creates the planner internally, because it might
		// change KDL chain during the process
		comms::ObjectsPoses result;
		int push_failure;
		std::tuple<State, State, int> debug_push;
		if (m_robot->PlanPush(this->GetCurrentStartState(), m_agents.at(m_agent_map[moved.first]).get(), push, movable_obstacles, m_all_objects, result, push_failure, debug_push, 1.0))
		{
			// valid push found!
			succ_object_centric_actions->emplace_back(moved.first, 0); // TODO: currently only one aidx
			succ_objects->push_back(std::move(result));
			succ_trajs->push_back(m_robot->GetLastPlan());
		}
		else
		{
			SMPL_INFO("Tried pushing object %d. Return value = %d", moved.first, push_failure);
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

			trajectory_msgs::JointTrajectory dummy_traj;
			succ_object_centric_actions->emplace_back(-1, -1);
			succ_objects->push_back(m_all_objects);
			succ_trajs->push_back(dummy_traj);
		}
		debug_pushes->push_back(debug_push);
	}
}

size_t MAMONode::GetConstraintHash() const
{
	if (m_hash_set_C) {
		return m_hash_C;
	}

	size_t hash_val = 0;

	for (const auto &object_state : kobject_states())
	{
		const auto &disc_pose = object_state.disc_pose();
		hash_val ^= std::hash<int>()(disc_pose.x()) ^ std::hash<int>()(disc_pose.y());

		bool p = disc_pose.pitch() != 0, r = disc_pose.roll() != 0;
		if (!object_state.symmetric() || p || r)
		{
			hash_val ^= std::hash<int>()(disc_pose.yaw());
			if (p) {
				hash_val ^= std::hash<int>()(disc_pose.pitch());
			}
			if (r) {
				hash_val ^= std::hash<int>()(disc_pose.roll());
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
	hash_val ^= std::hash<int>()(m_oidx);
	hash_val ^= std::hash<int>()(m_aidx);

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

void MAMONode::SetDebugPush(const std::tuple<State, State, int>& debug_push)
{
	m_debug_push = debug_push;
	m_have_debug_push = true;
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

void MAMONode::resetAgents()
{
	m_agents.clear();
	m_object_states.clear();
}

} // namespace clutter
