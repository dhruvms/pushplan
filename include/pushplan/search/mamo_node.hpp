#ifndef MAMO_NODE_HPP
#define MAMO_NODE_HPP

#include <pushplan/agents/object.hpp>
#include <pushplan/search/cbs.hpp>
#include <pushplan/utils/types.hpp>

#include <boost/heap/fibonacci_heap.hpp>
#include <comms/ObjectsPoses.h>
#include <smpl/heap/intrusive_heap.h>
#include <trajectory_msgs/JointTrajectory.h>

#include <vector>
#include <utility>
#include <memory>
#include <stdexcept>

namespace clutter
{

class Planner;

class MAMONode
{
public:
	MAMONode() :
		m_parent(nullptr), m_oidx(-1), m_aidx(-1),
		m_hash_set_C(false), m_hash_set_S(false) {} ;
	MAMONode(int oidx, int aidx) :
		m_parent(nullptr), m_oidx(oidx), m_aidx(oidx),
		m_hash_set_C(false), m_hash_set_S(false) {} ;

	void InitAgents(
		const std::vector<std::shared_ptr<Agent> >& agents,
		const comms::ObjectsPoses& all_objects);
	std::vector<double>* GetCurrentStartState();
	bool RunMAPF();
	void GetSuccs(
		std::vector<std::pair<int, int> > *succ_object_centric_actions,
		std::vector<comms::ObjectsPoses> *succ_objects,
		std::vector<trajectory_msgs::JointTrajectory> *succ_trajs,
		std::vector<std::tuple<State, State, int> > *debug_pushes);

	size_t GetConstraintHash() const;
	size_t GetSearchHash() const;
	void SetConstraintHash(const size_t& hash_val);
	void SetSearchHash(const size_t& hash_val);

	void SetParent(MAMONode *parent);
	void SetRobotTrajectory(const trajectory_msgs::JointTrajectory& robot_traj);
	void SetDebugPush(const std::tuple<State, State, int>& debug_push);

	void SetPlanner(Planner *planner);
	void SetCBS(const std::shared_ptr<CBS>& cbs);
	void SetCC(const std::shared_ptr<CollisionChecker>& cc);
	void SetRobot(const std::shared_ptr<Robot>& robot);
	void SetEdgeTo(int oidx, int aidx);
	void AddChild(MAMONode* child);
	void RemoveChild(MAMONode* child);

	size_t num_objects() const;
	const std::vector<ObjectState>& kobject_states() const;
	trajectory_msgs::JointTrajectory& robot_traj();
	const trajectory_msgs::JointTrajectory& krobot_traj() const;
	const MAMONode* kparent() const;
	MAMONode* parent();
	const std::vector<MAMONode*>& kchildren() const;
	std::pair<int, int> object_centric_action() const;
	const std::vector<std::pair<int, Trajectory> >& kmapf_soln() const;
	bool has_traj() const;

private:
	comms::ObjectsPoses m_all_objects; // all object poses at node
	std::vector<std::shared_ptr<Agent> > m_agents; // pointers to relevant objects
	std::unordered_map<int, size_t> m_agent_map;
	std::vector<ObjectState> m_object_states; // current relevant object states
	int m_oidx, m_aidx; // object-to-move id, action-to-use id
	trajectory_msgs::JointTrajectory m_robot_traj; // robot trajectory from parent to this node
	std::tuple<State, State, int> m_debug_push; // push start and end for viz purposes
	bool m_have_debug_push = false;

	MAMONode *m_parent; // parent node in tree
	std::vector<MAMONode*> m_children; // children nodes in tree

	std::vector<std::pair<int, Trajectory> > m_mapf_solution; // mapf solution found at this node
	std::vector<int> m_relevant_ids;

	Planner *m_planner;
	std::shared_ptr<CBS> m_cbs;
	std::shared_ptr<CollisionChecker> m_cc;
	std::shared_ptr<Robot> m_robot;

	size_t m_hash_C, m_hash_S;
	bool m_hash_set_C, m_hash_set_S;

	void addAgent(
		const std::shared_ptr<Agent>& agent,
		const size_t& pidx);
	void identifyRelevantMovables();

	void resetAgents();
};

////////////////////////
// Equality operators //
////////////////////////

struct EqualsConstraint
{
	bool operator()(MAMONode *a, MAMONode *b) const
	{
		if (a->num_objects() != b->num_objects()) {
			return false;
		}

		return std::is_permutation(a->kobject_states().begin(), a->kobject_states().end(), b->kobject_states().begin());
	}
};

struct EqualsSearch
{
	bool operator()(MAMONode *a, MAMONode *b) const
	{
		EqualsConstraint checkObjects;
		if (!checkObjects(a, b)) {
			return false;
		}
		return a->kmapf_soln() == b->kmapf_soln();
	}
};

///////////////////////////
// Hash Function Structs //
///////////////////////////

struct HashConstraint {
	size_t operator()(MAMONode *hashable_node) const
	{
		auto hash_val = hashable_node->GetConstraintHash();
		hashable_node->SetConstraintHash(hash_val);
		return hash_val;
	}
};

struct HashSearch {
	size_t operator()(MAMONode *hashable_node) const
	{
		auto hash_val = hashable_node->GetSearchHash();
		hashable_node->SetSearchHash(hash_val);
		return hash_val;
	}
};

} // namespace clutter


#endif // MAMO_NODE_HPP
