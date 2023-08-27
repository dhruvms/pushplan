#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
	// if (argc != 2) {
	// 	std::cerr << "usage: torchplan <path-to-exported-script-module>\n";
	// 	return -1;
	// }

	torch::jit::script::Module module;
	try {
		// Deserialize the ScriptModule from a file using torch::jit::load().
		module = torch::jit::load("/home/dhruv/work/code/ros/sbpl/src/pushplan/scripts/learning/models/[128,256,128]_best_traced-double_prob.pth");
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
		return -1;
	}

	std::cout << "ok\n";


	at::TensorOptions options(at::ScalarType::Double);

	std::vector<double> x_sample{-0.1212, 0.0775, 0.0750, -0.6107, 1.0000, 0.0426, 0.0426, 0.1500, 0.8804, 0.8205};
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
	std::cout << push_params << std::endl;

	for (float t = 0.0; t <= 0.2; t += 0.01)
	{
		auto x_in = torch::cat({push_params, t * torch::ones({(int)push_to.size()/2, 1})}, -1);
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(x_in);

		// Execute the model and turn its output into a tensor.
		at::Tensor output = module.forward(inputs).toTensor();

		auto rows = torch::arange(0, output.size(0), torch::kLong);
		auto cols = torch::ones(output.size(0));
		cols = cols.toType(at::kLong);
		at::Tensor score = (1 - (output.index({rows, cols}) * output.index({rows, cols * 0}))) * 1000;

		output = torch::cat({output, score.unsqueeze(1)}, -1);
		std::cout << '\n' << "Threshold = " << t << '\n' << output << '\n';

		// double x = 0.0, y = 0.0;
		// int idx = x_offset * int((x + 0.3)/res) + int((y + 0.4)/res);
		// unsigned int cell_score = score[idx].item<double>();
		// std::cout << "(0.0, 0.0) idx = " << idx << " | score = " << cell_score << std::endl;
	}
}
