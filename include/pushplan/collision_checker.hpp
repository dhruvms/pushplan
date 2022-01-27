#ifndef COLLISION_CHECKER_HPP
#define COLLISION_CHECKER_HPP

#include <pushplan/types.hpp>

#include <boost/functional/hash.hpp>
#include <fcl/config.h>
#include <fcl/broadphase/broadphase_bruteforce.h>
#include <fcl/broadphase/broadphase_spatialhash.h>
#include <fcl/broadphase/broadphase_SaP.h>
#include <fcl/broadphase/broadphase_SSaP.h>
#include <fcl/broadphase/broadphase_interval_tree.h>
#include <fcl/broadphase/broadphase_dynamic_AABB_tree.h>
#include <fcl/broadphase/broadphase_dynamic_AABB_tree_array.h>
#include <fcl/broadphase/default_broadphase_callbacks.h>
#include <fcl/broadphase/detail/sparse_hash_table.h>
#include <fcl/broadphase/detail/spatial_hash.h>
#include <fcl/geometry/geometric_shape_to_BVH_model.h>
// #include <fcl/broadphase/broadphase.h>

#include <vector>
#include <random>
#include <unordered_map>
#include <utility>
#include <iostream>

namespace std
{

class PairHash
{
public:
	// id is returned as hash function
	size_t operator()(const pair<int, int>& s) const
	{
		size_t seed = 0;
		boost::hash_combine(seed, s.first);
		boost::hash_combine(seed, s.second);
		return seed;
	}
};

inline
bool operator==(const pair<int, int>& a, const pair<int, int>& b)
{
	return (a.first == b.first && a.second == b.second) ||
				(a.first == b.second && a.second == b.first);
}

inline
ostream& operator<<(ostream& os, unordered_map<pair<int, int>, int, PairHash> const& s)
{
	os << "[" << s.size() << "] { ";
	for (pair<pair<int, int>, int> i : s)
		os << "(" << i.first.first << ", " << i.first.second << ", " << i.second << ") ";
	return os << "}\n";
}

} // namespace std

namespace clutter
{

class Planner;
class Agent;

class CollisionChecker
{
public:
	CollisionChecker(Planner* planner, const std::vector<Object>& obstacles);

	void AddObstacle(const Object& o) {
		m_obstacles.push_back(o);
	};

	void UpdateTraj(const int& priority, const Trajectory& traj);

	bool OutOfBounds(const LatticeState& s);
	bool ImmovableCollision(const State& s, fcl::CollisionObjectf* o);
	bool ImmovableCollision(const LatticeState& s, fcl::CollisionObjectf* o);
	bool FCLCollision(Agent* a1, Agent* a2);
	bool FCLCollision(Agent* a1, const int& a2_id, const LatticeState& a2_q);

	State GetRandomStateOutside(fcl::CollisionObjectf* o);

	double GetTableHeight() { return m_obstacles.at(m_base_loc).o_z + m_obstacles.at(0).z_size; };
	double OutsideXMin() { return m_obstacles.at(m_base_loc).o_x - (2 * m_obstacles.at(m_base_loc).x_size); };
	double OutsideYMin() { return m_obstacles.at(m_base_loc).o_y - (0.67 * m_obstacles.at(m_base_loc).y_size); };
	double OutsideXMax() { return m_obstacles.at(m_base_loc).o_x - m_obstacles.at(m_base_loc).x_size; };
	double OutsideYMax() { return m_obstacles.at(m_base_loc).o_y + (0.67 * m_obstacles.at(m_base_loc).y_size); };

	int NumObstacles() { return (int)m_obstacles.size(); };
	const std::vector<Object>* GetObstacles() { return &m_obstacles; };


private:
	Planner* m_planner = nullptr;

	std::vector<Object> m_obstacles;
	size_t m_base_loc;
	std::vector<Trajectory> m_trajs;

	std::random_device m_dev;
	std::mt19937 m_rng;
	std::uniform_real_distribution<double> m_distD;

	fcl::BroadPhaseCollisionManagerf* m_fcl_immov = nullptr;

	void cleanupChildren(std::vector<int>& check);

};

} // namespace clutter


#endif // COLLISION_CHECKER_HPP
