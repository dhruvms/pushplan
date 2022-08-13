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

void MAMONode::RunMAPF(unsigned int my_state_id)
{
	auto locally_invalid_pushes = m_planner->GetLocallyInvalidPushes(this->GetConstraintHash());
	for (size_t i = 0; i < m_agents.size(); ++i)
	{
		assert(m_agents.at(i)->GetID() == m_object_states.at(i).id());
		m_agents.at(i)->SetObjectPose(m_object_states.at(i).cont_pose());
		m_agents.at(i)->Init();

		m_agents.at(i)->ResetInvalidPushes(m_planner->GetGloballyInvalidPushes());
		if (locally_invalid_pushes != nullptr)
		{
			const auto it = locally_invalid_pushes->find(m_agents.at(i)->GetID());
			if (it != locally_invalid_pushes->end()) {
				m_agents.at(i)->SetLocallyInvalidPushes(it->second);
			}
		}
	}

	m_cc->InitMovableCC(m_agents);

	m_cbs->Reset();
	m_cbs->AddObjects(m_agents);
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

void MAMONode::addAgent(
	const std::shared_ptr<Agent>& agent,
	const size_t& pidx)
{
	m_agents.push_back(agent);

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
