#ifndef HELPERS_HPP
#define HELPERS_HPP

#include <pushplan/types.hpp>
#include <pushplan/constants.hpp>

#include <smpl/time.h>
#include <smpl/console/console.h>

#include <cmath>

namespace clutter
{

static double GetTime()
{
	using namespace smpl;
	return to_seconds(clock::now().time_since_epoch());
}

template <typename T>
inline
int sgn(T val) {
	return (T(0) < val) - (val < T(0));
}

inline
float normalize_angle(float angle)
{
	// normalize to [-2*pi, 2*pi] range
    if (std::fabs(angle) > 2.0 * M_PI) {
        angle = std::fmod(angle, 2.0 * M_PI);
    }

    if (angle < -M_PI) {
        angle += 2.0 * M_PI;
    }
    if (angle > M_PI) {
        angle -= 2.0 * M_PI;
    }

    return angle;
}

inline
float shortest_angle_diff(float af, float ai)
{
    return normalize_angle(af - ai);
}

/// Return the shortest distance between two angles.
inline
float shortest_angle_dist(float af, float ai)
{
    return std::fabs(shortest_angle_diff(af, ai));
}

// converts continuous (radians) version of angle into discrete
// maps 0->0, [YAWRES/2, 3/2*YAWRES)->1, [3/2*YAWRES, 5/2*YAWRES)->2,...
inline
int ContYawToDisc(float yaw)
{
	int num_yaw_bins = 2.0 * M_PI / YAWRES;
	return (int)(normalize_angle(yaw + YAWRES / 2.0) / (2.0 * M_PI) * (num_yaw_bins));
}

inline
void ContToDisc(const Pointf& in, Point& out)
{
	out.x = (in.x / XYRES) + sgn(in.x) * 0.5;
	out.y = (in.y / XYRES) + sgn(in.y) * 0.5;
	out.yaw = ContYawToDisc(in.yaw);
}

inline
void DiscToCont(const Point& in, Pointf& out)
{
	out.x = in.x * XYRES;
	out.y = in.y * XYRES;
	out.yaw = in.yaw * YAWRES;
}

inline
float dot(const Pointf& a, const Pointf& b)
{
	return a.x*b.x + a.y*b.y;
}

inline
Pointf vector(const Pointf& from, const Pointf& to)
{
	Pointf v;
	v.x = to.x - from.x;
	v.y = to.y - from.y;

	return v;
}

}

#endif // HELPERS_HPP
