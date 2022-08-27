#include <pushplan/search/mamo_search.hpp>
#include <pushplan/search/planner.hpp>

#include <limits>
#include <stdexcept>

namespace clutter
{

bool MAMOSearch::CreateRoot()
{
	m_root_node = new MAMONode;
	m_root_node->SetPlanner(m_planner);
	m_root_node->SetCBS(m_planner->GetCBS());
	m_root_node->SetCC(m_planner->GetCC());
	m_root_node->SetRobot(m_planner->GetRobot());

	m_root_node->InitAgents(m_planner->GetAllAgents(), m_planner->GetStartObjects()); // inits required fields for hashing
	m_root_node->SetEdgeTo(-1, -1);

	// 1. run MAPF
	bool mapf_solved = m_root_node->RunMAPF();
	if (!mapf_solved) {
		return false;
	}

	m_root_id = m_hashtable.GetStateIDForceful(m_root_node);
	m_root_search = getSearchState(m_root_id);
	m_search_nodes.push_back(m_root_node);

	m_root_node->SaveNode(m_root_id, 0);

	return true;
}

bool MAMOSearch::Solve()
{
	m_root_search->g = 0;
	m_root_search->m_OPEN_h = m_OPEN.push(m_root_search);

	while (!m_OPEN.empty())
	{
		auto next = m_OPEN.top();
		if (done(next))
		{
			SMPL_INFO("Final plan found!");

			auto node = m_hashtable.GetState(next->state_id);
			auto parent_id = next->bp == nullptr ? 0 : next->bp->state_id;
			node->RunMAPF();
			next->g += node->robot_traj().points.size();

			m_solved_node = node;
			m_solved_search = next;
			extractRearrangements();

			return true;
		}
		expand(next);
	}
	return false;
}

void MAMOSearch::GetRearrangements(std::vector<trajectory_msgs::JointTrajectory>& rearrangements, int& grasp_at)
{
	rearrangements.clear();
	rearrangements = m_rearrangements;
	grasp_at = m_grasp_at;
}

bool MAMOSearch::expand(MAMOSearchState *state)
{
	// SMPL_WARN("Expand %d, g-value = %d", state->state_id, state->g);
	auto node = m_hashtable.GetState(state->state_id);

	// 2. generate appropriate successor states
	std::vector<std::pair<int, int> > succ_object_centric_actions;
	std::vector<comms::ObjectsPoses> succ_objects;
	std::vector<trajectory_msgs::JointTrajectory> succ_trajs;
	std::vector<std::tuple<State, State, int> > debug_pushes;
	node->GetSuccs(&succ_object_centric_actions, &succ_objects, &succ_trajs, &debug_pushes);

	assert(succ_object_centric_actions.size() == succ_objects.size());
	assert(succ_objects.size() == succ_trajs.size());
	assert(succ_trajs.size() == succ_object_centric_actions.size());

	// 3. add successor states to search graph
	createSuccs(node, state, &succ_object_centric_actions, &succ_objects, &succ_trajs, &debug_pushes);
	state->closed = true;
	m_OPEN.erase(state->m_OPEN_h);

	return true;
}

bool MAMOSearch::done(MAMOSearchState *state)
{
	auto node = m_hashtable.GetState(state->state_id);
	if (node->kparent() != nullptr && node->object_centric_action() == std::make_pair(-1, -1)) {
		return false;
	}

	bool mapf_done = true;
	const auto& mapf_soln = node->kmapf_soln();
	for (size_t i = 0; i < mapf_soln.size(); ++i)
	{
		const auto& moved = mapf_soln.at(i);
		if (moved.second.size() == 1 || moved.second.front().coord == moved.second.back().coord) {
			continue;
		}
		mapf_done = false;
		break;
	}

	if (mapf_done) {
		return true;
	}

	auto start_state = node->GetCurrentStartState();
	return m_planner->FinalisePlan(node->kobject_states(), start_state, m_exec_traj);
}

void MAMOSearch::extractRearrangements()
{
	m_rearrangements.clear();
	m_rearrangements.push_back(std::move(m_exec_traj));
	ROS_WARN("Solution path state ids (from goal to start):");
	for (MAMOSearchState *state = m_solved_search; state; state = state->bp)
	{
		auto node = m_hashtable.GetState(state->state_id);
		if (node->has_traj())
		{
			ROS_WARN("\t%u", state->state_id);
			m_rearrangements.push_back(node->krobot_traj());
		}
	}
	std::reverse(m_rearrangements.begin(), m_rearrangements.end());
	m_grasp_at = m_planner->GetRobot()->GraspAt();
}

void MAMOSearch::createSuccs(
	MAMONode *parent_node,
	MAMOSearchState *parent_search_state,
	std::vector<std::pair<int, int> > *succ_object_centric_actions,
	std::vector<comms::ObjectsPoses> *succ_objects,
	std::vector<trajectory_msgs::JointTrajectory> *succ_trajs,
	std::vector<std::tuple<State, State, int> > *debug_pushes)
{
	size_t num_succs = succ_object_centric_actions->size();
	bool duplicate_successor = succ_object_centric_actions->back() == std::make_pair(-1, -1);

	unsigned int parent_g = parent_search_state->g;
	for (size_t i = 0; i < num_succs; ++i)
	{
		auto succ = new MAMONode;
		succ->SetPlanner(m_planner);
		succ->SetCBS(m_planner->GetCBS());
		succ->SetCC(m_planner->GetCC());
		succ->SetRobot(m_planner->GetRobot());

		succ->InitAgents(m_planner->GetAllAgents(), succ_objects->at(i)); // inits required fields for hashing
		succ->SetEdgeTo(succ_object_centric_actions->at(i).first, succ_object_centric_actions->at(i).second);

		// run MAPF
		bool mapf_solved = succ->RunMAPF();
		if (!mapf_solved)
		{
			delete succ;
			continue;
		}

		// check if we have visited this mamo state before
		unsigned int old_id, old_g;
		MAMOSearchState *prev_search_state = nullptr;
		if (m_hashtable.Exists(succ))
		{
			old_id = m_hashtable.GetStateID(succ);
			prev_search_state = getSearchState(old_id);
			old_g = prev_search_state->g;
		}

		unsigned int succ_cost = (duplicate_successor && i == num_succs - 1) ? 1000 : succ_trajs->at(i).points.size();
		unsigned int succ_g = parent_g + succ_cost;
		if (prev_search_state != nullptr && (prev_search_state->closed || old_g <= succ_g))
		{
			// previously visited this mamo state with a better value, move on
			delete succ;
			continue;
		}
		else
		{
			succ->SetRobotTrajectory(succ_trajs->at(i));
			succ->SetParent(parent_node);
			if (duplicate_successor && i == num_succs - 1)
			{
				for (auto it = debug_pushes->begin() + i; it != debug_pushes->end(); ++it) {
					succ->AddDebugPush(*it);
				}
			}
			else {
				succ->AddDebugPush(debug_pushes->at(i));
			}

			if (prev_search_state != nullptr)
			{
				// update search state in open list
				auto old_succ = m_hashtable.GetState(old_id);
				old_succ->parent()->RemoveChild(old_succ);

				m_hashtable.UpdateState(succ);
				prev_search_state->bp = parent_search_state;
				prev_search_state->g = succ_g;
				m_OPEN.update(prev_search_state->m_OPEN_h);
				// SMPL_WARN("Update existing successor %d, g-value = %d (previously %d)", old_id, prev_search_state->g, old_g);
			}
			else
			{
				// add search state to open list
				unsigned int succ_id = m_hashtable.GetStateIDForceful(succ);
				auto succ_search_state = getSearchState(succ_id);
				succ_search_state->bp = parent_search_state;
				succ_search_state->g = succ_g;
				succ_search_state->m_OPEN_h = m_OPEN.push(succ_search_state);

				// SMPL_WARN("Generate %d, g-value = %d", succ_id, succ_search_state->g);
				succ->SaveNode(succ_id, parent_search_state->state_id);
			}
			parent_node->AddChild(succ);
			m_search_nodes.push_back(succ);
		}
	}
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
	state->closed = false;
	state->bp = nullptr;
}

} // namespace clutter
