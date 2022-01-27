#include <pushplan/collision_checker.hpp>
#include <pushplan/planner.hpp>
#include <pushplan/geometry.hpp>
#include <default_broadphase_callbacks.h>

#include <smpl/console/console.h>
#include <fcl/broadphase/broadphase_dynamic_AABB_tree.h>

namespace clutter
{

CollisionChecker::CollisionChecker(Planner* planner, const std::vector<Object>& obstacles)
:
m_planner(planner),
m_obstacles(obstacles),
m_rng(m_dev())
{
	m_fcl_immov = new fcl::DynamicAABBTreeCollisionManagerf();

	// preprocess immovable obstacles
	for (size_t i = 0; i != m_obstacles.size(); ++i)
	{
		if (m_obstacles.at(i).id == 1) {
			m_base_loc = i;
			continue;
		}
		m_fcl_immov->registerObject(m_obstacles.at(i).GetFCLObject());
	}

	m_distD = std::uniform_real_distribution<double>(0.0, 1.0);

	initMovableCollisionChecker();
}

void CollisionChecker::initMovableCollisionChecker(){
	/*
	Inits EECBS Collision Checker which is capable of collision
	checking with multiple objects at the same time
	*/
	m_fcl_mov = new fcl::DynamicAABBTreeCollisionManagerf(); // Used for Collision checking with all other movable agents (EECBS)
	auto all_agents = m_planner->GetAllAgents();

	for(int i = 0; i < all_agents.size(); i++){
		m_fcl_mov->registerObject(all_agents[i]->GetFCLObject());
	}

	return;
}

void CollisionChecker::UpdateTraj(const int& priority, const Trajectory& traj)
{
	if (int(m_trajs.size()) <= priority) {
		m_trajs.resize(priority + 1, {});
	}

	m_trajs.at(priority) = traj;
}

bool CollisionChecker::OutOfBounds(const LatticeState& s)
{
	bool oob = s.state.at(0) <= (m_obstacles.at(m_base_loc).o_x - m_obstacles.at(m_base_loc).x_size);
	oob = oob || s.state.at(0) >= (m_obstacles.at(m_base_loc).o_x + m_obstacles.at(m_base_loc).x_size);
	oob = oob || s.state.at(1) <= (m_obstacles.at(m_base_loc).o_y - m_obstacles.at(m_base_loc).y_size);
	oob = oob || s.state.at(1) >= (m_obstacles.at(m_base_loc).o_y + m_obstacles.at(m_base_loc).y_size);

	return oob;
}

bool CollisionChecker::ImmovableCollision(const State& s, fcl::CollisionObjectf* o)
{
	LatticeState ls;
	ls.state = s;
	return this->ImmovableCollision(ls, o);
}

bool CollisionChecker::ImmovableCollision(const LatticeState& s, fcl::CollisionObjectf* o)
{
	// double start_time = GetTime(), time_taken;

	fcl::Transform3f pose;
	pose.setIdentity();
	fcl::Vector3f T(o->getTranslation());
	// T.setValue(s.state.at(0), s.state.at(1), T[2]);
	T << s.state.at(0), s.state.at(1), T[2];
	pose.translation() = T;

	o->setTransform(pose);
	o->computeAABB();

	m_fcl_immov->setup();
	fcl::DefaultCollisionData<float> collision_data;
	m_fcl_immov->collide(o, &collision_data, fcl::DefaultCollisionFunction);

	// time_taken = GetTime() - start_time;
	// SMPL_INFO("Immovable collision check: %f seconds.", time_taken);

	return collision_data.result.isCollision();
}

// called by CBS::findConflicts
bool CollisionChecker::FCLCollision(Agent* a1, Agent* a2)
{
	fcl::CollisionRequestf request;
	fcl::CollisionResultf result;
	fcl::collide(a1->GetFCLObject(), a2->GetFCLObject(), request, result);
	return result.isCollision();
}

// called by Agent::generateSuccessor
bool CollisionChecker::FCLCollision(Agent* a1, const int& a2_id, const LatticeState& a2_q)
{
	Agent* a2 = m_planner->GetAgent(a2_id);
	a2->UpdatePose(a2_q);

	fcl::CollisionRequestf request;
	fcl::CollisionResultf result;
	fcl::collide(a1->GetFCLObject(), a2->GetFCLObject(), request, result);
	return result.isCollision();
}

State CollisionChecker::GetRandomStateOutside(fcl::CollisionObjectf* o)
{
	State g(2, 0.0);
	State gmin(2, 0.0), gmax(2, 0.0);

	gmin.at(0) = OutsideXMin();
	gmax.at(0) = OutsideXMax();

	gmin.at(1) = OutsideYMin();
	gmax.at(1) = OutsideYMax();

	do
	{
		g.at(0) = (m_distD(m_rng) * (gmax.at(0) - gmin.at(0))) + gmin.at(0);
		g.at(1) = (m_distD(m_rng) * (gmax.at(1) - gmin.at(1))) + gmin.at(1);
	}
	while (ImmovableCollision(g, o));

	return g;
}

bool CollisionChecker::FCLCollisionMultipleAgents(
	Agent* a1,
	const std::vector<int>& all_agent_ids,
	const std::vector<LatticeState>& all_agent_poses){
	/*
	EECBS stats
	Collision checks between a1 and all the other agents (at the current timestep)
	*/

	m_fcl_mov->unregisterObject(a1->GetFCLObject());

	for(int i = 0; i < all_agent_ids.size(); i++){
		Agent* agent = m_planner->GetAgent(all_agent_ids[i]);
		agent->UpdatePose(all_agent_poses[i]);
		m_fcl_mov->update(agent->GetFCLObject());
	}

	m_fcl_mov->setup();
	fcl::DefaultCollisionData<float> collision_data;
	m_fcl_mov->collide(a1->GetFCLObject(), 
		&collision_data, fcl::DefaultCollisionFunction);

	m_fcl_mov->registerObject(a1->GetFCLObject());

	return collision_data.result.isCollision();

}

} // namespace clutter
