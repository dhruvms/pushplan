#include <pushplan/utils/controller.hpp>
#include <ros/ros.h>

int main (int argc, char ** argv)
{
	ros::init(argc, argv, "suhail_best");
	clutter::RobotController c;
	c.InitControllers();
	c.RaiseTorso(0.3);
	return 0;
}