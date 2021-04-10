#include "dqn.h"

void Init_Weights(torch::nn::Module& m)
{
	if ((typeid(m) == typeid(torch::nn::LinearImpl)) || (typeid(m) == typeid(torch::nn::Linear))) {
		auto p = m.named_parameters(false);
		auto w = p.find("weight");
		auto b = p.find("bias");

		if (w != nullptr) torch::nn::init::xavier_uniform_(*w);
		if (b != nullptr) torch::nn::init::constant_(*b, 0.01);
	}
}


DeepQNetworkImpl::DeepQNetworkImpl(const double learningRate, const int inputDims, const std::vector<int>& fcDims, const int actionsNum,
	const NetworkMode mode, const torch::Device& device) :
	learningRate(learningRate), inputDims(inputDims), fcDims(fcDims), actionsNum(actionsNum), device(device)
{
	fc.reserve(fcDims.size() + 1);

	fc.emplace_back(torch::nn::Linear(inputDims, fcDims[0])); //assume that number of hidden layers is at leat one
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
	try {
		for (auto& p : this->parameters()) {
			p.uniform_(0.01, 0.03); // or whatever initialization you are looking for, see link below
		}
	}
	catch (c10::Error e)
	{
		std::cout << e.what();
	}

	if (mode == NetworkMode::Training)
		this->train();
	else if (mode == NetworkMode::Evaluation)
		this->eval();

	std::clog << "DQN initialized" << std::endl;
}

DeepQNetworkImpl::~DeepQNetworkImpl()
{
	std::clog << "Well..." << std::endl;
}

torch::Tensor DeepQNetworkImpl::forward(const torch::Tensor& state)
{
	auto actions = torch::relu(fc[0](state));
	
	for (auto it = fc.begin() + 1; it != fc.end(); ++it)
	{
		actions = torch::relu((*it)(actions));
		
	}
	//std::cout <<"Q values		"<< actions << std::endl;
	return actions;
}






// Save the model
//torch::save(policyNet, "D:/model.pt");4
//for (const auto& pair : policyNet->named_parameters()) {
//	std::cout << pair.key() << ": " << pair.value() << std::endl;
//}
//