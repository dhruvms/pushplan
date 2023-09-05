#ifndef MAMO_NODE_HPP
#define MAMO_NODE_HPP

#include <pushplan/agents/object.hpp>
#include <pushplan/search/cbs.hpp>
#include <pushplan/utils/constants.hpp>
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

struct MAMOAction
{
	MAMOActionType _type;
	int _oid;
	std::vector<int> _params;

	MAMOAction() {};
	MAMOAction(MAMOActionType t, int o) : _type(t), _oid(o) {};
};

class MAMONode
{
public:
	MAMONode() :
		m_parent(nullptr),
		m_hash_set_l1(false), m_hash_set_l2(false) {} ;

	void InitAgents(
		const std::vector<std::shared_ptr<Agent> > &agents,
		const comms::ObjectsPoses& all_objects,
		const std::vector<int> &reachable_ids);
	std::vector<double>* GetCurrentStartState();
	bool RunMAPF();
	void GetSuccs(
		std::vector<MAMOAction> *succ_object_centric_actions,
		std::vector<comms::ObjectsPoses> *succ_objects,
		std::vector<trajectory_msgs::JointTrajectory> *succ_trajs,
		std::vector<std::tuple<State, State, int> > *debug_actions,
		bool *close,
		double *mapf_time, double *get_succs_time, double *sim_time);
	unsigned int ComputeMAMOPriorityOrig();
	void ComputePriorityFactors();
	void SaveNode(
		unsigned int my_id, unsigned int parent_id,
		const std::string &suffix=std::string());

	size_t GetObjectsHash() const;
	void SetObjectsHash(const size_t& hash_val);
	// size_t GetObjectsMAPFHash() const;
	// void SetObjectsMAPFHash(const size_t& hash_val);
	// size_t GetObjectsConstraintsHash() const;
	// void SetObjectsConstraintsHash(const size_t& hash_val);

	void SetParent(MAMONode *parent);
	void SetRobotTrajectory(const trajectory_msgs::JointTrajectory& robot_traj);
	void AddDebugAction(const std::tuple<State, State, int>& debug_action);

	void SetPlanner(Planner *planner);
	void SetCBS(const std::shared_ptr<CBS>& cbs);
	void SetCC(const std::shared_ptr<CollisionChecker>& cc);
	void SetRobot(const std::shared_ptr<Robot>& robot);
	void SetEdgeTo(const MAMOAction &action);
	void AddChild(MAMONode* child);
	void RemoveChild(MAMONode* child);

	size_t num_objects() const;
	const std::vector<ObjectState>& kobject_states() const;
	const std::vector<ObjectState>& kall_object_states() const;
	trajectory_msgs::JointTrajectory& robot_traj();
	const trajectory_msgs::JointTrajectory& krobot_traj() const;
	const MAMONode* kparent() const;
	MAMONode* parent();
	const std::vector<MAMONode*>& kchildren() const;
	const MAMOAction& kobject_centric_action() const;
	const std::vector<std::pair<int, Trajectory> >& kmapf_soln() const;
	bool has_traj() const;
	bool has_mapf_soln() const;
	const float& percent_ngr() const;
	const std::vector<std::vector<double> >& obj_priority_data() const;

	void ResetConstraints();

private:
	comms::ObjectsPoses m_all_objects; // all object poses at node
	std::vector<std::shared_ptr<Agent> > m_agents; // pointers to relevant objects
	std::unordered_map<int, size_t> m_agent_map;
	std::vector<ObjectState> m_object_states, m_all_object_states; // current relevant object states
	MAMOAction m_action_to_me; // used action details
	trajectory_msgs::JointTrajectory m_robot_traj; // robot trajectory from parent to this node
	std::vector<std::tuple<State, State, int> > m_debug_actions; // push start and end for viz purposes

	MAMONode *m_parent; // parent node in tree
	std::vector<MAMONode*> m_children; // children nodes in tree

	std::vector<std::pair<int, Trajectory> > m_mapf_solution; // mapf solution found at this node
	std::list<std::pair<int, Coord> > m_successful_rearranges, m_successful_rearranges_invalidated;
	std::vector<int> m_relevant_ids;
	bool m_new_constraints = true;
	bool m_expanded_once = false;

	Planner *m_planner;
	std::shared_ptr<CBS> m_cbs;
	std::shared_ptr<CollisionChecker> m_cc;
	std::shared_ptr<Robot> m_robot;

	size_t m_hash_l1, m_hash_l2;
	bool m_hash_set_l1, m_hash_set_l2;
	float m_percent_ngr, m_percent_objs;
	unsigned int m_num_objs;
	std::vector<std::vector<double> > m_obj_priority_data;

	void addAgent(
		const std::shared_ptr<Agent>& agent,
		const size_t& pidx);
	// void identifyRelevantMovables();

	void resetAgents();

	bool tryPickPlace(
		std::vector<MAMOAction> *succ_object_centric_actions,
		std::vector<comms::ObjectsPoses> *succ_objects,
		std::vector<trajectory_msgs::JointTrajectory> *succ_trajs,
		std::vector<std::tuple<State, State, int> > *debug_actions,
		std::vector<std::tuple<State, State, int> > *invalid_action_samples,
		const std::vector<Object*> &movable_obstacles, size_t aid);
	bool tryPush(
		std::vector<MAMOAction> *succ_object_centric_actions,
		std::vector<comms::ObjectsPoses> *succ_objects,
		std::vector<trajectory_msgs::JointTrajectory> *succ_trajs,
		std::vector<std::tuple<State, State, int> > *debug_actions,
		std::vector<std::tuple<State, State, int> > *invalid_action_samples,
		const std::vector<Object*> &movable_obstacles, size_t aid, double *sim_time);
};

////////////////////////
// Equality operators //
////////////////////////

struct EqualsObjects
{
	bool operator()(MAMONode *a, MAMONode *b) const
	{
		if (a->num_objects() != b->num_objects()) {
			return false;
		}

		return std::is_permutation(a->kall_object_states().begin(), a->kall_object_states().end(), b->kall_object_states().begin());
	}
};

// struct EqualsObjectsMAPF
// {
// 	bool operator()(MAMONode *a, MAMONode *b) const
// 	{
// 		EqualsObjects checkObjects;
// 		if (!checkObjects(a, b)) {
// 			return false;
// 		}
// 		return a->kmapf_soln() == b->kmapf_soln();
// 	}
// };

// struct EqualsObjectsConstraints
// {
// 	bool operator()(MAMONode *a, MAMONode *b) const
// 	{
// 		EqualsObjects checkObjects;
// 		if (!checkObjects(a, b)) {
// 			return false;
// 		}
// 		return a->kmapf_constraints() == b->kmapf_constraints();
// 	}
// };

///////////////////////////
// Hash Function Structs //
///////////////////////////

struct HashObjects {
	size_t operator()(MAMONode *hashable_node) const
	{
		auto hash_val = hashable_node->GetObjectsHash();
		hashable_node->SetObjectsHash(hash_val);
		return hash_val;
	}
};

// struct HashObjectsMAPF {
// 	size_t operator()(MAMONode *hashable_node) const
// 	{
// 		auto hash_val = hashable_node->GetObjectsMAPFHash();
// 		hashable_node->SetObjectsMAPFHash(hash_val);
// 		return hash_val;
// 	}
// };

// struct HashObjectsConstraints {
// 	size_t operator()(MAMONode *hashable_node) const
// 	{
// 		auto hash_val = hashable_node->GetObjectsConstraintsHash();
// 		hashable_node->SetObjectsConstraintsHash(hash_val);
// 		return hash_val;
// 	}
// };

} // namespace clutter


#endif // MAMO_NODE_HPP
