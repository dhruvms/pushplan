#ifndef MAMO_SEARCH_HPP
#define MAMO_SEARCH_HPP


#include <pushplan/search/mamo_node.hpp>
#include <pushplan/utils/hash_manager.hpp>

#include <boost/heap/fibonacci_heap.hpp>

#include <unordered_map>

namespace clutter
{

class Planner;

class MAMOSearch
{
public:
	MAMOSearch() = default;
	MAMOSearch(Planner *planner) : m_planner(planner) {};

	void CreateRoot();
	bool Solve();
	void GetRearrangements(std::vector<trajectory_msgs::JointTrajectory>& rearrangements, int& grasp_at);

private:
	Planner *m_planner = nullptr;

	std::vector<MAMONode*> m_search_nodes;
	std::vector<MAMOSearchState*> m_search_states;
	boost::heap::fibonacci_heap<MAMOSearchState*, boost::heap::compare<MAMOSearchState::OPENCompare> > m_OPEN;

	MAMONode *m_root_node = nullptr, *m_solved_node = nullptr;
	MAMOSearchState *m_root_search = nullptr, *m_solved_search = nullptr;
	unsigned int m_root_id;
	HashManager<HashSearch, EqualsSearch> m_hashtable;

	std::vector<trajectory_msgs::JointTrajectory> m_rearrangements;
	int m_grasp_at;

	bool done(MAMOSearchState *state);
	void extractRearrangements();

	MAMOSearchState *getSearchState(unsigned int state_id);
	MAMOSearchState *createSearchState(unsigned int state_id);
	void initSearchState(MAMOSearchState *state);
};

} // namespace clutter


#endif // MAMO_SEARCH_HPP
