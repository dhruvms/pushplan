#include <torch/script.h> // One-stop header.
#include <Eigen/Dense>

#include <iostream>
#include <memory>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_rm;

int main(int argc, const char* argv[]) {
	// if (argc != 2) {
	// 	std::cerr << "usage: torchplan <path-to-exported-script-module>\n";
	// 	return -1;
	// }

	torch::jit::script::Module module;
	try {
		// Deserialize the ScriptModule from a file using torch::jit::load().
		module = torch::jit::load("/home/dhruv/work/code/ros/sbpl/src/pushplan/scripts/learning/models/[64,128,256,128,64]_best_traced-triple_task.pth");
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
		return -1;
	}

	std::cout << "ok\n";


	at::TensorOptions options(at::ScalarType::Double);

	// -0.0659, 0.0761, 0.3979
	std::vector<double> x_sample{-0.0809, 0.1866, 0.0750, 0.3310, 1.0000, 0.0300, 0.0300, 0.1500, 0.3631, 0.8550};
	std::vector<int64_t> x_size = {1, 10};
	at::Tensor x_vec = torch::from_blob(x_sample.data(), at::IntList(x_size), options);
	x_vec = x_vec.toType(at::kFloat);
	// std::cout << x_vec << std::endl;

	std::vector<double> push_to;
	double res = 0.2;
	for (double dx = -0.3; dx <= 0.3; dx += res)
	{
		for (double dy = -0.4; dy <= 0.4; dy += res)
		{
			push_to.push_back(dx);
			push_to.push_back(dy);
		}
	}
	int x_offset = int(0.8/res);

	std::vector<int64_t> push_to_size = {(int)push_to.size()/2, 2};
	at::Tensor push_to_vec = torch::from_blob(push_to.data(), at::IntList(push_to_size), options);
	push_to_vec = push_to_vec.toType(at::kFloat);
	// std::cout << push_to_vec << std::endl;

	auto push_params = x_vec.repeat({(int)push_to.size()/2, 1});
	push_params = torch::cat({push_params, push_to_vec}, -1);
	// std::cout << push_params << std::endl;

	for (float t = 0.01; t <= 0.05; t += 0.04)
	{
		auto x_in = torch::cat({push_params, t * torch::ones({(int)push_to.size()/2, 1})}, -1);
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(x_in);

		// Execute the model and turn its output into a tensor.
		at::Tensor output = module.forward(inputs).toTensor();

		auto rows = torch::arange(0, output.size(0), torch::kLong);
		auto cols = torch::ones(output.size(0));
		cols = cols.toType(at::kLong);
		at::Tensor score = (1 - (output.index({rows, cols}) * output.index({rows, cols * 0}))) * 100;

		output = torch::cat({output, score.unsqueeze(1)}, -1);
		std::cout << '\n' << "Threshold = " << t << '\n' << output << '\n';

		float* data = output.data_ptr<float>();
		Eigen::Map<MatrixXf_rm> E(data, output.size(0), output.size(1));
		std::cout << '\n' << E << '\n' << '\n';

		Eigen::MatrixXf poses = E.block(0, 2, E.rows(), 9);
		Eigen::VectorXf pose3 = poses.row(3);
		std::cout << '\n' << "pose3 = " << pose3 << '\n';

		Eigen::Vector3f c1, c2, c3;
		c2 = pose3.tail(3);
		c1 = pose3.segment(3, 3);
		std::cout << '\n' << "c1 = " << c1 << '\n';
		std::cout << '\n' << "c2 = " << c2 << '\n';

		c1.normalize();
		c2 = c2 - (c1.dot(c2) * c1);
		c2.normalize();
		std::cout << '\n' << "c1 = " << c1 << '\n';
		std::cout << '\n' << "c2 = " << c2 << '\n';
		Eigen::Matrix3f R;
		R << c1, c2, c1.cross(c2);
		Eigen::Quaternion<float> q(R);
		std::cout << '\n' << "R = " << R << '\n';
		std::cout << '\n' << "q = " << q.x() << ", " << q.y() << ", " << q.z() << ", " << q.w() << '\n';

		Eigen::Affine3f pose = Eigen::Translation3f(pose3.head(3)) * q;
		std::cout << '\n' << "pose = " << pose.matrix() << '\n';
		Eigen::Affine3d B = pose.cast<double>();
		std::cout << '\n' << "B = " << B.matrix() << '\n';

		// at::Tensor pose3 = poses.index({torch::tensor(3).toType(at::kLong)});


		// double x = 0.0, y = 0.0;
		// int idx = x_offset * int((x + 0.3)/res) + int((y + 0.4)/res);
		// unsigned int cell_score = score[idx].item<double>();
		// std::cout << "(0.0, 0.0) idx = " << idx << " | score = " << cell_score << std::endl;
	}
}
