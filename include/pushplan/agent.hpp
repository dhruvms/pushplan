#ifndef AGENT_HPP
#define AGENT_HPP

#include <pushplan/types.hpp>
#include <pushplan/movable.hpp>

#include <memory>

namespace clutter
{

class Agent : public Movable
{
public:
	Agent() {};
	Agent(const Object& o) {
		m_objs.push_back(o);
	};
	void SetObject(const Object& o) {
		m_objs.push_back(o);
	}

	bool Init() override;

	bool AtGoal(const LatticeState& s, bool verbose=false) override;
	void Step(int k) override;

	unsigned int GetGoalHeuristic(int state_id) override;
	unsigned int GetGoalHeuristic(const LatticeState& s) override;

	const std::vector<Object>* GetObject(const LatticeState& s) override;
	using Movable::GetObject;

private:
	int generateSuccessor(
		const LatticeState* parent,
		int dx, int dy,
		std::vector<int>* succs,
		std::vector<unsigned int>* costs) override;
	unsigned int cost(
		const LatticeState* s1,
		const LatticeState* s2);
	bool convertPath(
		const std::vector<int>& idpath) override;
};

} // namespace clutter


#endif // AGENT_HPP
