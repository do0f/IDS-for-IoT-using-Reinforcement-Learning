#include "dqn.h"


DeepQNetworkImpl::DeepQNetworkImpl(const double learningRate, const int inputDims, const std::vector<int>& fcDims, const int actionsNum,
	const NetworkMode mode, const torch::Device& device) :
	learningRate(learningRate), inputDims(inputDims), fcDims(fcDims), actionsNum(actionsNum), device(device)
{
	fc.reserve(fcDims.size() + 1);

	fc.emplace_back(torch::nn::Linear(inputDims, fcDims[0])); //assume that number of hidden layers is at least one
	for (auto it = fcDims.begin(); it != fcDims.end() - 1; ++it)
		fc.emplace_back(torch::nn::Linear(*it, *(it + 1)));
	fc.emplace_back(torch::nn::Linear(fcDims.back(), actionsNum));

	for (std::size_t i = 0; i < fc.size(); i++)
	{
		std::string name("fc");
		name.append(std::to_string(i));
		torch::nn::Module::register_module(name, fc[i]);
	}

	torch::NoGradGuard no_grad;

	for (auto& p : this->parameters()) {
		p.uniform_(0.01, 0.03); //set random weights for the layers
	}

	if (mode == NetworkMode::Training)
		this->train();
	else if (mode == NetworkMode::Evaluation)
		this->eval();

	this->to(device);
	std::cout << "DQN initialized" << std::endl;
}


torch::Tensor DeepQNetworkImpl::forward(const torch::Tensor& state)
{
	auto actions = torch::relu(fc[0](state)).to(device);
	
	for (auto it = fc.begin() + 1; it != fc.end() - 1; ++it) //reLU activate all layers but last
	{
		actions = torch::relu((*it)(actions)).to(device);
	}
	actions = fc.back()(actions).to(device); //last layer contains Q-values, we don't want them to be activated
	return actions.to(device);
}






// Save the model
//torch::save(policyNet, "D:/model.pt");4
//for (const auto& pair : policyNet->named_parameters()) {
//	std::cout << pair.key() << ": " << pair.value() << std::endl;
//}
//