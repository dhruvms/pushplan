#include <pushplan/planner.hpp>
#include <pushplan/helpers.hpp>

#include <smpl/debug/visualizer_ros.h>
#include <ros/ros.h>

#include <string>
#include <fstream>

using namespace clutter;

void SaveData(
	int scene_id,
	int mapf_calls, int mapf_sucesses, int not_lucky, int not_rearranged,
	bool dead, bool rearrange, std::uint32_t violation)
{
	std::string filename(__FILE__);
	auto found = filename.find_last_of("/\\");
	filename = filename.substr(0, found + 1) + "../dat/MAIN.csv";

	bool exists = FileExists(filename);
	std::ofstream STATS;
	STATS.open(filename, std::ofstream::out | std::ofstream::app);
	if (!exists)
	{
		STATS << "UID,"
				<< "MAPFCalls,MAPFSuccesses,"
				<< "NotLucky,NotRearranged,"
				<< "Timeout?,Rearranged?,ExecViolation?\n";
	}

	STATS << scene_id << ','
			<< mapf_calls << ',' << mapf_sucesses << ','
			<< not_lucky << ',' << not_rearranged << ','
			<< dead << ',' << rearrange << ','
			<< violation << '\n';
	STATS.close();
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "whca");
	ros::NodeHandle nh;
	ros::NodeHandle ph("~");

	smpl::VisualizerROS visualizer(nh, 100);
	smpl::viz::set_visualizer(&visualizer);

	// Let publishers set up
	ros::Duration(1.0).sleep();

	// read from NONE file
	std::string filename(__FILE__), results(__FILE__);
	auto found = filename.find_last_of("/\\");
	filename = filename.substr(0, found + 1) + "../dat/FIRST.txt";
	results = results.substr(0, found + 1) + "../dat/RESULTS.csv";

	std::ifstream NONE;
	NONE.open(filename);

	if (NONE.is_open())
	{
		std::string line, level;
		while (!NONE.eof())
		{
			getline(NONE, line);
			if (line.length() == 0) {
				break;
			}
			int scene_id = std::stoi(line);
			if (scene_id < 100000)
			{
				level = "0";
				// ROS_WARN("Planning for a scene with no movable objects!");
			}
			else if (scene_id < 200000) {
				level = "5";
			}
			else if (scene_id < 300000) {
				level = "10";
			}
			else {
				level = "15";
			}

			std::string planfile(__FILE__);
			auto found = planfile.find_last_of("/\\");
			planfile = planfile.substr(0, found + 1) + "../../../../simplan/src/simplan/data/clutter_scenes/";
			planfile += level + "/plan_" + line + "_SCENE.txt";
			ROS_WARN("Run planner on: %s", planfile.c_str());


			Planner p;
			if (!p.Init(planfile, scene_id)) {
				continue;
			}

			int mapf_calls = 0, mapf_sucesses = 0, not_lucky = 0, not_rearranged = 0;
			bool dead = false, rearrange = true;
			std::uint32_t violation;
			do
			{
				++mapf_calls;
				if (p.Plan())
				{
					++mapf_sucesses;

					ROS_WARN("Try extraction before rearrangement!");
					if (p.Alive() && p.TryExtract()) {
						break;
					}
					++not_lucky;

					ROS_WARN("Try rearrangement!");
					if (p.Alive()) {
						rearrange = p.Rearrange();
					}

					ROS_WARN("Try extraction after rearrangement!");
					if (p.Alive() && p.TryExtract()) {
						break;
					}
					++not_rearranged;
				}
			}
			while (p.Alive() && rearrange);
			dead = !p.Alive();

			if (p.Alive()) {
				violation = p.RunSim();

				if (violation == 0) {
					ROS_WARN("SUCCESS!!!");
				}
				else {
					ROS_ERROR("FAILURE!!!");
				}
			}
			else {
				violation |= 0x00000008;
				ROS_ERROR("Planner terminated!!!");
			}

			SaveData(
				scene_id,
				mapf_calls, mapf_sucesses, not_lucky, not_rearranged,
				dead, rearrange, violation);
		}
	}
	else
	{
		ROS_ERROR("Planner init error");
		return false;
	}

	NONE.close();

	return 0;
}
