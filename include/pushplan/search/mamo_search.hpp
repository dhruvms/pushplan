#ifndef MAMO_SEARCH_HPP
#define MAMO_SEARCH_HPP

#include <pushplan/search/mamo_node.hpp>
#include <pushplan/utils/hash_manager.hpp>

#include <boost/heap/fibonacci_heap.hpp>
#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions/exponential.hpp>

#include <unordered_map>

namespace clutter
{

struct MAMOSearchState
{
	unsigned int state_id;
	unsigned int actions, noops;
	double priority;
	bool closed, try_finalise, force_done, true_cost;
	MAMOSearchState *bp;
	MAMOAction action_to_me;

	struct OPENCompare
	{
		// return true if q goes before p
		bool operator()(const MAMOSearchState *p, const MAMOSearchState *q) const
		{
			if (p->actions == q->actions)
			{
				if (p->priority == q->priority)
				{
					if (p->noops == q->noops) {
						return rand() % 2;
					}
					return p->noops > q->noops;
				}
				return p->priority > q->priority;
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
	bool Solve(double budget=300.0);
	void GetRearrangements(
		std::vector<trajectory_msgs::JointTrajectory> &rearrangements,
		std::vector<MAMOAction> &actions);
	void SaveStats(int exec_success=-1);
	void SaveNBData();
	void Cleanup();

private:
	Planner *m_planner = nullptr;

	std::vector<MAMONode*> m_search_nodes;
	std::vector<MAMOSearchState*> m_search_states;
	boost::heap::fibonacci_heap<MAMOSearchState*, boost::heap::compare<MAMOSearchState::OPENCompare> > m_OPEN;

	MAMONode *m_root_node = nullptr, *m_solved_node = nullptr;
	MAMOSearchState *m_root_search = nullptr, *m_solved_search = nullptr;
	bool m_solved = false;
	unsigned int m_root_id;
	HashManager<HashObjects, EqualsObjects> m_hashtable;
	std::map<std::string, double> m_stats;
	double m_timer;

	std::vector<trajectory_msgs::JointTrajectory> m_rearrangements;
	std::vector<MAMOAction> m_actions;
	trajectory_msgs::JointTrajectory m_exec_traj, m_home_traj;

	bool expand(MAMOSearchState *state);
	bool evaluate(MAMOSearchState *state);
	bool done(MAMOSearchState *state);
	void extractRearrangements();
	void createSuccs(
		MAMONode *parent_node,
		MAMOSearchState *parent_search_state,
		std::vector<MAMOAction> *succ_object_centric_actions,
		std::vector<comms::ObjectsPoses> *succ_objects,
		std::vector<trajectory_msgs::JointTrajectory> *succ_trajs,
		std::vector<std::tuple<State, State, int> > *debug_pushes);

	MAMOSearchState *getSearchStateForceful();
	MAMOSearchState *getSearchState(unsigned int state_id);
	MAMOSearchState *createSearchState(unsigned int state_id);
	void initSearchState(MAMOSearchState *state);

	const double solvable_prior = 0.3084358524;
	boost::math::beta_distribution<> D_percent_ngr;
	boost::math::exponential_distribution<> D_num_objs;
	boost::math::exponential_distribution<> D_noops;
	boost::math::exponential_distribution<> D_odata;
	double computeMAMOPriority(MAMOSearchState *state);
	void createDists();
};

} // namespace clutter


#endif // MAMO_SEARCH_HPP
