#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <vector>
#include <map>
#include <string>

namespace clutter
{

enum class LowLevelConflictHeuristic
{
	ZERO,
	BINARY,
	COUNT,
	OURS,
	LLHC_TYPES
};
enum class HighLevelConflictHeuristic
{
	ZERO,
	CONFLICT_COUNT,
	AGENT_COUNT,
	AGENT_PAIRS,
	HLHC_TYPES
};
enum class ConflictPrioritisation
{
	RANDOM,
	EARLIEST,
	CONFLICTS,
	CP_TYPES
};
enum class MAPFAlgo
{
	WHCA,
	VCBS,
	ECBS,
	CBSWP,
	PBS,
	OURS,
	ALGO_TYPES
};
enum class MAMOActionType
{
	PUSH,
	PICKPLACE,
	RETRIEVEOOI,
	DUMMY,
	ACTION_TYPES
};
enum class PushResult
{
	START_INSIDE_OBSTACLE,
	START_UNREACHABLE,
	IK_SUCCESS,
	IK_JOINT_LIMIT,
	IK_OBSTACLE,
	IK_CATCHALL_FAIL,
	NO_OOI_COLLISION,
	FAIL_IN_SIM,
	SUCCESS_IN_SIM,
	PUSH_RESULTS
};
enum class PickPlaceResult
{
	PREGRASPS_UNREACHABLE,
	GRASP_IK_FAIL,
	POSTGRASP_IK_FAIL,
	BAD_ATTACH,
	ATTACH_COLLIDES,
	PLAN_TO_POSTGRASP_FAIL,
	RETRACT_IK_FAIL,
	FAIL_IN_SIM,
	SUCCESS_IN_SIM,
	PICKPLACE_RESULTS
};

extern bool FRIDGE;

extern double MAPF_PLANNING_TIME;
extern double RES;
extern double GOAL_THRESH;

extern int WINDOW;
extern int GRID;

extern double R_SPEED;

extern bool SAVE;
extern bool CC_2D;
extern bool CC_3D;

extern double DF_RES;

extern LowLevelConflictHeuristic LLHC;
extern HighLevelConflictHeuristic HLHC;
extern ConflictPrioritisation CP;
extern MAPFAlgo ALGO;

extern const std::vector<int> YCB_OBJECTS;
extern const std::map<int, std::string> YCB_OBJECT_NAMES;
extern const std::map<int, std::vector<double>> YCB_OBJECT_DIMS;

extern double DEG5;
extern double LOG2PI;

extern int SAMPLES;

extern double GRIPPER_WIDTH_2;

extern int CELL_COST_FACTOR;

} // namespace clutter


#endif // CONSTANTS_HPP
