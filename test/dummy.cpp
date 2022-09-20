#include <pushplan/search/planner.hpp>
#include <pushplan/utils/helpers.hpp>
#include <pushplan/utils/constants.hpp>

#include <smpl/debug/visualizer_ros.h>
#include <ros/ros.h>

#include <string>
#include <fstream>

using namespace clutter;

int main(int argc, char** argv)
{
	ros::init(argc, argv, "dummy");
	ros::NodeHandle nh;
	ros::NodeHandle ph("~");

	smpl::VisualizerROS visualizer(nh, 100);
	smpl::viz::set_visualizer(&visualizer);

	// Let publishers set up
	ros::Duration(1.0).sleep();

	std::string planfile(__FILE__);
	auto found = planfile.find_last_of("/\\");
	int uid;
	ph.getParam("goal/uid", uid);
	planfile = planfile.substr(0, found + 1) + "../../../../simplan/src/simplan/data/clutter_scenes/dummy/plan_";
	planfile += std::to_string(uid) + "_SCENE.txt";
	ROS_WARN("Run planner on: %s", planfile.c_str());

	bool replay;
	ph.getParam("robot/replay", replay);

	Planner p;
	int scene_id = uid;
	// if (!p.Init(planfile, 999999, true)) {
	// 	ROS_ERROR("Init failed");
	// 	return 0;
	// }

	bool ycb;
	ph.getParam("objects/ycb", ycb);
	if (ycb) {
		scene_id = -1;
	}
	if (!p.Init(planfile, scene_id, ycb)) {
		ROS_ERROR("Initialisation failed!");
		return 0;
	}
	ROS_INFO("Planner and simulator init-ed!");

	if (!replay)
	{
		bool rearrange = true;
		do
		{
			bool done;
			if (p.Plan(done))
			{
				if (done)
				{
					SMPL_INFO("Final plan found!");
					break;
				}

				if (p.Alive()) {
					rearrange = p.Rearrange();
				}
			}
		}
		while (rearrange && p.Alive());

		if (p.Alive()) {
			p.RunSim(SAVE);
		}

		if (SAVE) {
			p.SaveData();
		}
	}

	else {
		p.RunSolution();
	}

	// if (p.Alive()) {
	// 	p.RunSim();
	// 	// ROS_ERROR("SHOULD I EXECUTE???");
	// 	// getchar();
	// 	// p.Execute();
	// }

	return 0;
}
