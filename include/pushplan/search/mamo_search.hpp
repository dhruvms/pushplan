#ifndef MAMO_SEARCH_HPP
#define MAMO_SEARCH_HPP


#include <pushplan/search/mamo_node.hpp>
#include <pushplan/utils/hash_manager.hpp>

#include <boost/heap/fibonacci_heap.hpp>

#include <unordered_map>

namespace clutter
{

struct MAMOSearchState
{
	unsigned int state_id;
	// unsigned int g, h, f;
	unsigned int priority, actions, noops;
	bool closed;
	MAMOSearchState *bp;

	struct OPENCompare
	{
		// lower => higher priority
		bool operator()(const MAMOSearchState *p, const MAMOSearchState *q) const
		{
			if (p->actions == q->actions)
			{
				if (p->noops == q->noops)
				{
					if (p->priority == q->priority)
					{
						return rand() % 2;
					}
					return p->priority > q->priority;
				}
				return p->noops < q->noops;
			}
			return p->actions < q->actions;
		}
	};
	boost::heap::fibonacci_heap<MAMOSearchState*, boost::heap::compare<MAMOSearchState::OPENCompare> >::handle_type m_OPEN_h;
};

class Planner;

class MAMOSearch
{
public:
	MAMOSearch() = default;
	MAMOSearch(Planner *planner) : m_planner(planner) {};

	bool CreateRoot();
	bool Solve();
	void GetRearrangements(std::vector<trajectory_msgs::JointTrajectory>& rearrangements, int& grasp_at);
	void SaveStats();

private:
	Planner *m_planner = nullptr;

	std::vector<MAMONode*> m_search_nodes;
	std::vector<MAMOSearchState*> m_search_states;
	boost::heap::fibonacci_heap<MAMOSearchState*, boost::heap::compare<MAMOSearchState::OPENCompare> > m_OPEN;

	MAMONode *m_root_node = nullptr, *m_solved_node = nullptr;
	MAMOSearchState *m_root_search = nullptr, *m_solved_search = nullptr;
	bool m_solved = false;
	unsigned int m_root_id;
	HashManager<HashObjectsMAPF, EqualsObjectsMAPF> m_hashtable;
	std::map<std::string, double> m_stats;

	std::vector<trajectory_msgs::JointTrajectory> m_rearrangements;
	trajectory_msgs::JointTrajectory m_exec_traj;
	int m_grasp_at;

	bool expand(MAMOSearchState *state);
	bool done(MAMOSearchState *state);
	void extractRearrangements();
	void createSuccs(
		MAMONode *parent_node,
		MAMOSearchState *parent_search_state,
		std::vector<std::pair<int, int> > *succ_object_centric_actions,
		std::vector<comms::ObjectsPoses> *succ_objects,
		std::vector<trajectory_msgs::JointTrajectory> *succ_trajs,
		std::vector<std::tuple<State, State, int> > *debug_pushes);

	MAMOSearchState *getSearchState(unsigned int state_id);
	MAMOSearchState *createSearchState(unsigned int state_id);
	void initSearchState(MAMOSearchState *state);
};

} // namespace clutter


#endif // MAMO_SEARCH_HPP
