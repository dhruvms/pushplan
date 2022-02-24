#include <pushplan/agent.hpp>
#include <pushplan/geometry.hpp>
#include <pushplan/cbs_nodes.hpp>
#include <pushplan/robot.hpp>
#include <pushplan/conflicts.hpp>

#include <smpl/console/console.h>

#include <iostream>
#include <algorithm>

namespace clutter
{

bool Agent::Setup()
{
	m_orig_o = m_objs.back();
}

void Agent::ResetObject()
{
	m_objs.back().o_x = m_orig_o.o_x;
	m_objs.back().o_y = m_orig_o.o_y;
}

bool Agent::SetObjectPose(
	const std::vector<double>& xyz,
	const std::vector<double>& rpy)
{
	m_objs.back().o_x = xyz.at(0);
	m_objs.back().o_y = xyz.at(1);
	m_objs.back().o_z = xyz.at(2);

	m_objs.back().o_roll = rpy.at(0);
	m_objs.back().o_pitch = rpy.at(1);
	m_objs.back().o_yaw = rpy.at(2);
}

bool Agent::Init()
{
	m_objs.back().o_x = m_orig_o.o_x;
	m_objs.back().o_y = m_orig_o.o_y;

	m_init.t = 0;
	m_init.hc = 0;
	m_init.state.clear();
	m_init.state.push_back(m_objs.back().o_x);
	m_init.state.push_back(m_objs.back().o_y);
	m_init.state.push_back(m_objs.back().o_z);
	m_init.state.push_back(m_objs.back().o_roll);
	m_init.state.push_back(m_objs.back().o_pitch);
	m_init.state.push_back(m_objs.back().o_yaw);
	ContToDisc(m_init.state, m_init.coord);

	if (!m_focal) {
		m_focal = std::make_unique<Focal>(this, 1.0); // make A* search object
	}
	this->reset();
	this->SetStartState(m_init);
	this->SetGoalState(m_init.coord);

	return true;
}

bool Agent::SatisfyPath(HighLevelNode* ct_node, Robot* robot, Trajectory** sol_path)
{
	m_solve.clear();
	LatticeState s = m_init;
	// OOI stays in place during approach
	for (int i = 0; i < robot->GraspAt() + 2; ++i)
	{
		s.t = i;
		m_solve.push_back(s);
	}

	// OOI tracks robot EE during extraction
	auto* r_traj = robot->GetLastTraj();
	for (int i = robot->GraspAt() + 2; i < r_traj->size(); ++i)
	{
		++s.t;
		s.state = robot->GetEEState(r_traj->at(i).state);
		ContToDisc(s.state, s.coord);
		m_solve.push_back(s);
	}
	*sol_path = &(this->m_solve);

	return true;
}

bool Agent::SatisfyPath(HighLevelNode* ct_node, Trajectory** sol_path, int& expands, int& min_f)
{
	m_solve.clear();
	expands = 0;
	min_f = 0;
	// collect agent constraints
	m_constraints.clear();
	for (auto& constraint : ct_node->m_constraints)
	{
		if (constraint->m_me == ct_node->m_replanned) {
			m_constraints.push_back(constraint);
		}
	}
	m_cbs_solution = &(ct_node->m_solution);
	m_cbs_id = ct_node->m_replanned;
	m_max_time = ct_node->m_makespan;

	std::vector<int> solution;
	int solcost;
	bool result = m_focal->replan(&solution, &solcost);

	if (result)
	{
		convertPath(solution);
		*sol_path = &(this->m_solve);
		expands = m_focal->get_n_expands();
		min_f = m_focal->get_min_f();
	}

	return result;
}

// As long as I am not allowed to be in this location at some later time,
// I have not reached a valid goal state
// Conversely, if I can remain in this location (per existing constraints),
// I am at a valid goal state (since states in collision with immovable obstacles
// or out of bounds will never enter OPEN)
bool Agent::IsGoal(int state_id)
{
	assert(state_id >= 0);
	LatticeState* s = getHashEntry(state_id);
	assert(s);

	bool constrained = false;
	for (const auto& constraint : m_constraints)
	{
		if (constraint->m_q.coord == s->coord) {
			if (constraint->m_time >= s->t) {
				constrained = true;
				break;
			}
		}
	}

	// LatticeState dummy = *s;
	// dummy.t = std::max(m_max_time, s->t);
	bool conflict = goalConflict(*s);

	// if (!constrained && !conflict) {
	// 	SMPL_INFO("Agent %d goal->t = %d", GetID(), s->t);
	// }

	return !constrained && !conflict;
	// return !constrained;
}

void Agent::GetSuccs(
	int state_id,
	std::vector<int>* succ_ids,
	std::vector<unsigned int>* costs)
{
	assert(state_id >= 0);
	succ_ids->clear();
	costs->clear();

	LatticeState* parent = getHashEntry(state_id);
	assert(parent);
	m_closed.push_back(parent);

	if (IsGoal(state_id)) {
		SMPL_WARN("We are expanding the goal state (???)");
		return;
	}

	for (int dx = -1; dx <= 1; ++dx)
	{
		for (int dy = -1; dy <= 1; ++dy)
		{
			// ignore ordinal directions for 4-connected grid
			if (GRID == 4 && std::abs(dx * dy) == 1) {
				continue;
			}

			generateSuccessor(parent, dx, dy, succ_ids, costs);
		}
	}
}

unsigned int Agent::GetGoalHeuristic(int state_id)
{
	// TODO: RRA* informed backwards Dijkstra's heuristic
	// TODO: Try penalising distance to shelf edge?
	assert(state_id >= 0);
	LatticeState* s = getHashEntry(state_id);
	assert(s);

	double dist = EuclideanDist(s->coord, m_goal);
	return (dist * COST_MULT);
}

unsigned int Agent::GetConflictHeuristic(int state_id)
{
	assert(state_id >= 0);
	LatticeState* s = getHashEntry(state_id);
	assert(s);

	return (s->hc * COST_MULT);
}

unsigned int Agent::GetGoalHeuristic(const LatticeState& s)
{
	// TODO: RRA* informed backwards Dijkstra's heuristic
	double dist = EuclideanDist(s.coord, m_goal);
	return (dist * COST_MULT);
}

const std::vector<Object>* Agent::GetObject(const LatticeState& s)
{
	m_objs.back().o_x = s.state.at(0);
	m_objs.back().o_y = s.state.at(1);
	return &m_objs;
}

void Agent::GetSE2Push(std::vector<double>& push)
{
	push.clear();
	push.resize(3, 0.0); // (x, y, yaw)

	double move_dir = std::atan2(
					m_solve.back().state.at(1) - m_solve.front().state.at(1),
					m_solve.back().state.at(0) - m_solve.front().state.at(0));
	push.at(0) = m_orig_o.o_x + std::cos(move_dir + M_PI) * (m_objs.back().x_size + 0.05);
	push.at(1) = m_orig_o.o_y + std::sin(move_dir + M_PI) * (m_objs.back().x_size + 0.05);
	push.at(2) = move_dir;

	if (m_objs.back().shape == 0)
	{
		// get my object rectangle
		std::vector<State> rect;
		State o = {m_orig_o.o_x, m_orig_o.o_y};
		GetRectObjAtPt(o, m_objs.back(), rect);

		// find rectangle side away from push direction
		push.at(0) = m_orig_o.o_x + std::cos(move_dir + M_PI) * 0.5;
		push.at(1) = m_orig_o.o_y + std::sin(move_dir + M_PI) * 0.5;
		State p = {push.at(0), push.at(1)}, intersection;
		double op = EuclideanDist(o, p);
		int side = 0;
		for (; side <= 3; ++side)
		{
			LineLineIntersect(o, p, rect.at(side), rect.at((side + 1) % 4), intersection);
			if (PointInRectangle(intersection, rect))
			{
				if (EuclideanDist(intersection, o) + EuclideanDist(intersection, p) <= op + 1e-6) {
					break;
				}
			}
		}

		// compute push point on side
		intersection.at(0) += std::cos(move_dir + M_PI) * 0.08;
		intersection.at(1) += std::sin(move_dir + M_PI) * 0.08;

		// update push
		push.at(0) = intersection.at(0);
		push.at(1) = intersection.at(1);
	}
}

int Agent::generateSuccessor(
	const LatticeState* parent,
	int dx, int dy,
	std::vector<int>* succs,
	std::vector<unsigned int>* costs)
{
	LatticeState child;
	child.t = parent->t + 1;
	child.hc = parent->hc;
	child.coord = parent->coord;
	child.coord.at(0) += dx;
	child.coord.at(1) += dy;
	DiscToCont(child.coord, child.state);
	child.state.insert(child.state.end(), parent->state.begin() + 2, parent->state.end());

	UpdatePose(child);
	if (m_cc->OutOfBounds(child) || m_cc->ImmovableCollision(this->GetFCLObject())) {
		return -1;
	}

	for (const auto& constraint : m_constraints)
	{
		if (child.t == constraint->m_time)
		{
			// Conflict type 1: robot-object conflict
			if (constraint->m_me == constraint->m_other)
			{
	 			if (m_cbs_solution->at(0).second.size() <= constraint->m_time)
				{
					// This should never happen - the constraint would only have existed
					// if this object and the robot had a conflict at that time
					SMPL_WARN("How did this robot-object conflict happen with a small robot traj?");
					continue;
				}

				// successor is invalid if I collide in state 'child'
				// with the robot configuration at the same time
				if (m_cc->RobotObjectCollision(
							this, child,
							m_cbs_solution->at(0).second.at(constraint->m_time), constraint->m_time))
				{
					return -1;
				}
			}
			// Conflict type 2: object-object conflict
			else
			{
				// successor is invalid if I collide in state 'child'
				// with the constraint->m_other object in state constraint->m_q
				if (m_cc->ObjectObjectCollision(this, constraint->m_other, constraint->m_q)) {
					return -1;
				}
			}
		}
	}

	child.hc += conflictHeuristic(child);

	int succ_state_id = getOrCreateState(child);
	LatticeState* successor = getHashEntry(succ_state_id);

	// For ECBS (the state of the object is collision checked with the state of all agents
	// and the resulting number of collisions is used as part of the cost function
	// therefore not a hard constraint (like constraints).)

	succs->push_back(succ_state_id);
	costs->push_back(cost(parent, successor));

	return succ_state_id;
}

int Agent::conflictHeuristic(const LatticeState& state)
{
	int hc = 0;
	UpdatePose(state);

	switch (LLHC)
	{
		case LowLevelConflictHeuristic::BINARY:
		{
			std::vector<LatticeState> other_poses;
			std::vector<int> other_ids;

			for (const auto& agent_traj: *m_cbs_solution)
			{
				if (agent_traj.first == m_cbs_id || agent_traj.first == 0) {
					continue;
				}

				other_ids.push_back(agent_traj.first);
				if (agent_traj.second.size() <= state.t) {
					other_poses.push_back(agent_traj.second.back());
				}
				else {
					other_poses.push_back(agent_traj.second.at(state.t));
				}
			}
			bool conflict = m_cc->ObjectObjectsCollision(this, other_ids, other_poses);

			if (!conflict)
			{
				if (m_cbs_solution->at(0).second.size() <= state.t) {
					conflict = conflict ||
						m_cc->RobotObjectCollision(this, state, m_cbs_solution->at(0).second.back(), state.t);
				}
				else {
					conflict = conflict ||
						m_cc->RobotObjectCollision(this, state, m_cbs_solution->at(0).second.at(state.t), state.t);
				}
			}

			hc = (int)conflict;
			break;
		}
		case LowLevelConflictHeuristic::COUNT:
		{
			LatticeState other_pose;
			for (const auto& other_agent: *m_cbs_solution)
			{
				if (other_agent.first == m_cbs_id || other_agent.first == 0) {
					continue;
				}

				// other agent trajectory is shorter than current state's time
				// so we only collision check against the last state along the
				// other agent trajectory
				if (other_agent.second.size() <= state.t) {
					other_pose = other_agent.second.back();
				}
				else {
					// if the other agent has a trajectory longer than current state's time
					// we collision check against the current state's time
					other_pose = other_agent.second.at(state.t);
				}
				if (m_cc->ObjectObjectCollision(this, other_agent.first, other_pose)) {
					++hc;
				}
			}

			// same logic for robot
			if (m_cbs_solution->at(0).second.size() <= state.t) {
				other_pose = m_cbs_solution->at(0).second.back();
			}
			else {
				other_pose = m_cbs_solution->at(0).second.at(state.t);
			}
			if (m_cc->RobotObjectCollision(this, state, other_pose, state.t)) {
				++hc;
			}
			break;
		}
		default:
		{
			SMPL_ERROR("Unknown conflict heuristic type!");
		}
	}

	return hc;
}

bool Agent::goalConflict(const LatticeState& state)
{
	UpdatePose(state);

	LatticeState other_pose;
	for (const auto& other_agent: *m_cbs_solution)
	{
		if (other_agent.first == m_cbs_id || other_agent.first == 0) {
			continue;
		}

		// other agent trajectory is shorter than current state's time
		// so we only collision check against the last state along the
		// other agent trajectory
		if (other_agent.second.size() <= state.t)
		{
			other_pose = other_agent.second.back();
			if (m_cc->ObjectObjectCollision(this, other_agent.first, other_pose)) {
				return true;
			}
		}
		else
		{
			// if the other agent has a trajectory longer than current state's time
			// we collision check against all states in that trajectory beyond
			// the current state's time
			for (int t = state.t; t < (int)other_agent.second.size(); ++t)
			{
				other_pose = other_agent.second.at(t);
				if (m_cc->ObjectObjectCollision(this, other_agent.first, other_pose)) {
					return true;
				}
			}
		}
	}

	// same logic for robot
	if (m_cbs_solution->at(0).second.size() <= state.t)
	{
		if (m_cc->RobotObjectCollision(this, state, m_cbs_solution->at(0).second.back(), state.t)) {
			return true;
		}
	}
	else
	{
		for (int t = state.t; t < (int)m_cbs_solution->at(0).second.size(); ++t)
		{
			other_pose = m_cbs_solution->at(0).second.at(t);
			if (m_cc->RobotObjectCollision(this, state, other_pose, state.t)) {
				return true;
			}
		}
	}

	return false;
}

unsigned int Agent::cost(
	const LatticeState* s1,
	const LatticeState* s2)
{
	double dist = EuclideanDist(s1->coord, s2->coord);
	dist = dist == 0.0 ? 1.0 : dist;
	return (dist * COST_MULT);
}

bool Agent::convertPath(
	const std::vector<int>& idpath)
{
	Trajectory opath; // vector of LatticeState

	if (idpath.empty()) {
		return true;
	}

	if (idpath[0] == m_goal_id)
	{
		SMPL_ERROR("Cannot extract a non-trivial path starting from the goal state");
		return false;
	}

	LatticeState state;

	// attempt to handle paths of length 1...do any of the sbpl planners still
	// return a single-point path in some cases?
	if (idpath.size() == 1)
	{
		auto state_id = idpath[0];

		if (state_id == m_goal_id)
		{
			auto* entry = getHashEntry(m_start_id);
			if (!entry)
			{
				SMPL_ERROR("Failed to get state entry for state %d", m_start_id);
				return false;
			}
			state = *entry;
			opath.push_back(state);
		}
		else
		{
			auto* entry = getHashEntry(state_id);
			if (!entry)
			{
				SMPL_ERROR("Failed to get state entry for state %d", state_id);
				return false;
			}
			state = *entry;
			opath.push_back(state);
		}
	}
	else
	{
		// grab the first point
		auto* entry = getHashEntry(idpath[0]);
		if (!entry)
		{
			SMPL_ERROR("Failed to get state entry for state %d", idpath[0]);
			return false;
		}
		state = *entry;
		opath.push_back(state);
	}

	// grab the rest of the points
	for (size_t i = 1; i < idpath.size(); ++i)
	{
		auto prev_id = idpath[i - 1];
		auto curr_id = idpath[i];

		if (prev_id == m_goal_id)
		{
			SMPL_ERROR("Cannot determine goal state predecessor state during path extraction");
			return false;
		}

		auto* entry = getHashEntry(curr_id);
		if (!entry)
		{
			SMPL_ERROR("Failed to get state entry state %d", curr_id);
			return false;
		}
		state = *entry;
		opath.push_back(state);
	}
	m_solve = std::move(opath);
	return true;
}

} // namespace clutter
