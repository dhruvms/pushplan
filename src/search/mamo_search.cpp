#include <pushplan/search/mamo_search.hpp>
#include <pushplan/search/planner.hpp>

#include <limits>
#include <stdexcept>

namespace clutter
{

void MAMOSearch::CreateRoot()
{
	m_root_node = new MAMONode;
	m_root_node->SetPlanner(m_planner);
	m_root_node->SetCBS(m_planner->GetCBS());
	m_root_node->SetCC(m_planner->GetCC());
	m_root_node->SetRobot(m_planner->GetRobot());
	m_root_node->SetEdgeTo(-1, -1);
	m_root_node->InitAgents(m_planner->GetAllAgents(), m_planner->GetStartObjects()); // inits required fields for hashing

	m_search_nodes.push_back(m_root_node);
	m_root_id = m_hashtable.GetStateIDForceful(m_root_node);
	m_root_search = getSearchState(m_root_id);
}

bool MAMOSearch::Solve()
{
	m_root_search->g = 0;
	m_root_search->leaf = true;
	m_root_search->m_OPEN_h = m_OPEN.push(m_root_search);

	while (!m_OPEN.empty())
	{
		auto next = m_OPEN.top();
		if (done(next))
		{
			auto node = m_hashtable.GetState(next->state_id);
			next->g += node->robot_traj().points.size();

			m_solved_node = node;
			m_solved_search = next;
			extractRearrangements();
			m_grasp_at = m_planner->GetRobot()->GraspAt();

			return true;
		}
		// expand(next);
	}
	return false;
}

void MAMOSearch::GetRearrangements(std::vector<trajectory_msgs::JointTrajectory>& rearrangements, int& grasp_at)
{
	rearrangements.clear();
	rearrangements = m_rearrangements;
	grasp_at = m_grasp_at;
}

// MAMOSearch::expand(MAMOSearchState *state)
// {
// 	if (!state->leaf) {
// 		throw std::runtime_error("Trying to expand non-leaf MAMO search state!");
// 	}

// 	auto node = m_hashtable.GetState(state->state_id);

// 	if (valid_action)
// 	{
// 		// 1. run MAPF
// 		node->RunMAPF(state->state_id);

// 		// 2. init successors
// 	}
// }

bool MAMOSearch::done(MAMOSearchState *state)
{
	auto node = m_hashtable.GetState(state->state_id);
	if (node->kparent() != nullptr && node->object_centric_action() == std::make_pair(-1, -1)) {
		return false;
	}

	auto start_state = node->GetCurrentStartState();
	return m_planner->FinalisePlan(node->kobject_states(), start_state, node->robot_traj());
}

void MAMOSearch::extractRearrangements()
{
	m_rearrangements.clear();
	for (MAMOSearchState *state = m_solved_search; state; state = state->bp)
	{
		auto node = m_hashtable.GetState(state->state_id);
		m_rearrangements.push_back(node->krobot_traj());
	}
	std::reverse(m_rearrangements.begin(), m_rearrangements.end());
}

MAMOSearchState* MAMOSearch::getSearchState(unsigned int state_id)
{
	if (m_search_states.size() <= state_id)	{
		m_search_states.resize(state_id + 1, nullptr);
	}

	auto& state = m_search_states[state_id];
	if (state == nullptr) {
		state = createSearchState(state_id);
	}

	return state;
}

MAMOSearchState* MAMOSearch::createSearchState(unsigned int state_id)
{
	auto state = new MAMOSearchState;
	state->state_id = state_id;
	initSearchState(state);
}

void MAMOSearch::initSearchState(MAMOSearchState *state)
{
	state->g = std::numeric_limits<unsigned int>::max();
	state->leaf = false;
	state->bp = nullptr;
}

} // namespace clutter
