#include <pushplan/collision_checker.hpp>
#include <pushplan/planner.hpp>
#include <pushplan/geometry.hpp>
#include <pushplan/constants.hpp>
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
	m_fcl_immov = new fcl::DynamicAABBTreeCollisionManager();
	m_fcl_mov = new fcl::DynamicAABBTreeCollisionManager();

	// preprocess immovable obstacles
	for (size_t i = 0; i != m_obstacles.size(); ++i)
	{
		if (m_obstacles.at(i).id == 1) {
			m_base_loc = i;
			continue;
		}
		LatticeState s;
		s.state.push_back(m_obstacles.at(i).o_x);
		s.state.push_back(m_obstacles.at(i).o_y);
		m_obstacles.at(i).UpdatePose(s);
		m_fcl_immov->registerObject(m_obstacles.at(i).GetFCLObject());
	}

	m_distD = std::uniform_real_distribution<double>(0.0, 1.0);
}

void CollisionChecker::InitMovableSet(std::vector<Agent>* agents)
{
	for (size_t i = 0; i != agents->size(); ++i) {
		m_fcl_mov->registerObject(agents->at(i).GetFCLObject());
	}
}

void CollisionChecker::AddToMovableSet(Agent* agent)
{
	m_fcl_mov->registerObject(agent->GetFCLObject());
}

void CollisionChecker::UpdateTraj(const int& priority, const Trajectory& traj)
{
	if (int(m_trajs.size()) <= priority) {
		m_trajs.resize(priority + 1, {});
	}

	m_trajs.at(priority) = traj;
}

bool CollisionChecker::ImmovableCollision(const State& s, fcl::CollisionObject* o)
{
	LatticeState ls;
	ls.state = s;
	return this->ImmovableCollision(ls, o);
}

bool CollisionChecker::ImmovableCollision(const LatticeState& s, fcl::CollisionObject* o)
{
	// double start_time = GetTime(), time_taken;

	fcl::Transform3f pose;
	pose.setIdentity();
	fcl::Vec3f T(o->getTranslation());
	T.setValue(s.state.at(0), s.state.at(1), T[2]);
	pose.setTranslation(T);

	o->setTransform(pose);
	o->computeAABB();

	m_fcl_immov->setup();
	fcl::DefaultCollisionData collision_data;
	m_fcl_immov->collide(o, &collision_data, fcl::DefaultCollisionFunction);

	// time_taken = GetTime() - start_time;
	// SMPL_INFO("Immovable collision check: %f seconds.", time_taken);

	return collision_data.result.isCollision();
}

// IsStateValid should only be called for priority > 1
bool CollisionChecker::IsStateValid(
	const LatticeState& s, fcl::CollisionObject* o1, const int& priority)
{
	// double start_time = GetTime(), time_taken;

	m_fcl_mov->unregisterObject(o1);
	auto o1_new = m_planner->GetObject(s, priority); // updates pose

	LatticeState robot;
	// Check against movables' FCL manager
	for (int p = 0; p < priority; ++p)
	{
		for (const auto& s2: m_trajs.at(p))
		{
			if (s.t == s2.t)
			{
				if (p == 0)
				{
					robot = s2; // store for later
					break;
				}
				else
				{
					auto o2 = m_planner->GetObject(s2, p);
					m_fcl_mov->update(o2);
				}
			}
		}
	}
	m_fcl_mov->setup();
	fcl::DefaultCollisionData collision_data;
	m_fcl_mov->collide(o1, &collision_data, fcl::DefaultCollisionFunction);
	bool collision = collision_data.result.isCollision();

	// time_taken = GetTime() - start_time;
	// SMPL_INFO("Movable collision check: %f seconds.", time_taken);

	// start_time = GetTime();
	// double start_time = GetTime(), time_taken;

	if (!collision && !robot.state.empty())
	{
		if (!CC_2D) {
			collision = collision || m_planner->CheckRobotCollision(robot, priority);
		}
		else
		{
			auto o1_obj = m_planner->GetObject(priority)->back();
			State o1_loc = {s.state.at(0), s.state.at(1)};
			std::vector<State> o1_rect;
			bool rect_o1 = false;

			// preprocess rectangle once only
			if (o1_obj.Shape() == 0)
			{
				GetRectObjAtPt(o1_loc, o1_obj, o1_rect);
				rect_o1 = true;
			}

			auto robot_2d = m_planner->Get2DRobot(robot);

			if (!checkCollisionObjSet(o1_obj, o1_loc, rect_o1, o1_rect, robot_2d))
			{
				if (!CC_3D) {
					collision = true;
				}
				else {
					collision = collision || m_planner->CheckRobotCollision(robot, priority);
				}
			}
		}
	}


	// time_taken = GetTime() - start_time;
	// SMPL_INFO("Robot collision check: %f seconds.", time_taken);

	m_fcl_mov->registerObject(o1_new);
	return !collision;
}

bool CollisionChecker::UpdateConflicts(
	const LatticeState& s, const int& priority)
{
	auto o1 = m_planner->GetObject(s, priority); // updates pose
	int id1 = m_planner->GetID(priority);

	LatticeState robot;
	bool done = false;
	for (int p = 0; p < priority; ++p)
	{
		for (const auto& s2: m_trajs.at(p))
		{
			if (s.t == s2.t)
			{
				if (p == 0)
				{
					robot = s2; // store for later
					break;
				}
				else
				{
					auto o2 = m_planner->GetObject(s2, p);

					fcl::CollisionRequest request;
					fcl::CollisionResult result;
					fcl::collide(o1, o2, request, result);
					if (result.isCollision())
					{
						int id2 = m_planner->GetID(p);
						int priority2 = p == 1 ? 0 : p;
						updateConflicts(id1, priority, id2, priority2, s.t);

						done = true;
						break;
					}
				}
			}
		}
		if (done) {
			break;
		}
	}

	if (!done)
	{
		if (!CC_2D)
		{
			if (m_planner->CheckRobotCollision(robot, priority)) {
				updateConflicts(id1, priority, 100, 0, robot.t);
			}
		}
		else
		{
			auto o1_obj = m_planner->GetObject(priority)->back();
			State o1_loc = {s.state.at(0), s.state.at(1)};
			std::vector<State> o1_rect;
			bool rect_o1 = false;

			// preprocess rectangle once only
			if (o1_obj.Shape() == 0)
			{
				GetRectObjAtPt(o1_loc, o1_obj, o1_rect);
				rect_o1 = true;
			}

			auto robot_2d = m_planner->Get2DRobot(robot);

			if (!checkCollisionObjSet(o1_obj, o1_loc, rect_o1, o1_rect, robot_2d))
			{
				if (!CC_3D) {
					updateConflicts(id1, priority, 100, 0, robot.t);
				}
				else
				{
					if (m_planner->CheckRobotCollision(robot, priority)) {
						updateConflicts(id1, priority, 100, 0, robot.t);
					}
				}
			}
		}
	}

	return true;
}

auto CollisionChecker::GetConflictsOf(int pusher) const ->
std::unordered_map<std::pair<int, int>, int, std::PairHash>
{
	auto pushed = m_conflicts;
	pushed.clear();
	for (const auto& c: m_conflicts)
	{
		if (c.first.second == pusher) {
			pushed[c.first] = c.second;
		}
	}

	return pushed;
}

void CollisionChecker::getChildrenOf(int pusher, std::vector<int>& children) const
{
	children.clear();
	for (const auto& c: m_conflicts)
	{
		if (c.first.second == pusher) {
			children.push_back(c.first.first);
		}
	}
}

void CollisionChecker::GetDescendentsOf(int pusher, std::unordered_set<int>& descendents) const
{
	std::vector<int> to_check, children;
	to_check.push_back(pusher);

	while (!to_check.empty())
	{
		int check = to_check.back();
		to_check.pop_back();

		getChildrenOf(check, children);
		to_check.insert(to_check.end(), children.begin(), children.end());
		for (const auto& c: children) {
			descendents.insert(c);
		}
	}
}

State CollisionChecker::GetRandomStateOutside(fcl::CollisionObject* o)
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

bool CollisionChecker::updateConflicts(
	int id1, int p1,
	int id2, int p2, int t)
{
	auto key = std::make_pair(id1, id2);
	auto search = m_conflicts.find(key);
	if (search != m_conflicts.end())
	{
		if (search->second > t) {
			m_conflicts[key] = t;
		}
	}
	else {
		m_conflicts.emplace(key, t);
	}
}

bool CollisionChecker::checkCollisionObjSet(
	const Object& o1, const State& o1_loc,
	bool rect_o1, const std::vector<State>& o1_rect,
	const std::vector<Object>* a2_objs)
{
	State o2_loc;
	bool rect_o2;
	std::vector<State> o2_rect;

	for (const auto& ao: *a2_objs)
	{
		rect_o2 = false;
		o2_loc = {ao.o_x, ao.o_y};
		if (ao.Shape() == 0)
		{
			GetRectObjAtPt(o2_loc, ao, o2_rect);
			rect_o2 = true;
		}

		if (rect_o1)
		{
			if (rect_o2)
			{
				if (rectRectCollision(o1_rect, o2_rect)) {
					return false;
				}
			}
			else
			{
				if (rectCircCollision(o1_rect, ao, o2_loc)) {
					return false;
				}
			}
		}
		else
		{
			if (rect_o2)
			{
				if (rectCircCollision(o2_rect, o1, o1_loc)) {
					return false;
				}
			}
			else
			{
				if (circCircCollision(o1, o1_loc, ao, o2_loc)) {
					return false;
				}
			}
		}
	}
	// SMPL_WARN("collision! objects ids %d and %d (movable) collide at time %d", o1.id, m_planner->GetObject(p)->id, s.t);
	// std::cout << o1.id << ',' << other_obj->id << ',' << s.t << std::endl;

	return true;
}

bool CollisionChecker::rectRectCollision(
	const std::vector<State>& r1, const std::vector<State>& r2)
{
	return RectanglesIntersect(r1, r2);
}

bool CollisionChecker::rectCircCollision(
	const std::vector<State>& r1, const Object& c1, const State& c1_loc)
{
	return (PointInRectangle(c1_loc, r1) ||
			LineSegCircleIntersect(c1_loc, c1.x_size, r1.at(0), r1.at(1)) ||
			LineSegCircleIntersect(c1_loc, c1.x_size, r1.at(1), r1.at(2)) ||
			LineSegCircleIntersect(c1_loc, c1.x_size, r1.at(2), r1.at(3)) ||
			LineSegCircleIntersect(c1_loc, c1.x_size, r1.at(3), r1.at(0)));
}

bool CollisionChecker::circCircCollision(
	const Object& c1, const State& c1_loc,
	const Object& c2, const State& c2_loc)
{
	double dist = EuclideanDist(c1_loc, c2_loc);
	return (dist < (c1.x_size + c2.x_size));
}

} // namespace clutter
