#pragma once
#include <torch/torch.h>
#include "dqn.h"
#include "experience.h"



class Agent
{
public:
	Agent(const double gamma, const double epsilon, const double learningRate, const unsigned int inputDims, const unsigned int batchSize,
		const unsigned int actionsNum, const unsigned int maxMemorySize = 100'000, const double epsilonEnd = 0.01, const double epsilonDec = 5e-4,
		const torch::Device& device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
	Agent() = delete;
	~Agent();

	int chooseAction(const torch::Tensor& state, DeepQNetwork& policyNet);
	void storeExperience(const Experience& experience);

	void learn(DeepQNetwork& policyNet, DeepQNetwork& targetNet, torch::optim::Adam& optimizer);

private:
	const double gamma;
	double epsilon;
	const double learningRate;
	const unsigned int inputDims;
	const unsigned int batchSize;
	const unsigned int actionsNum;
	std::vector<long long> actionSpace;
	const unsigned int maxMemorySize;
	unsigned int memoryCount = 0;
	const double epsilonEnd;
	const double epsilonDec;
	const torch::Device device;
	std::unique_ptr<ExperienceMemory> memory;
};
