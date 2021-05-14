#include "agent.h"

Agent::Agent(const double gamma, const double epsilon, const double learningRate, const unsigned int inputDims, const unsigned int batchSize,
	const unsigned int actionsNum, const unsigned int maxMemorySize, const double epsilonEnd, const double epsilonDec,
	const torch::Device& device) :
	gamma(gamma), epsilon(epsilon), learningRate(learningRate), inputDims(inputDims), batchSize(batchSize), actionsNum(actionsNum),
	maxMemorySize(maxMemorySize), epsilonEnd(epsilonEnd), epsilonDec(epsilonDec), device(device), 
	memory(std::make_unique<ExperienceMemory>(ExperienceMemory(maxMemorySize)))
{
	actionSpace.reserve(actionsNum);
	for (unsigned int i = 0; i < actionsNum; i++)
		actionSpace.push_back(i);
}

Agent::~Agent()
{
	std::clog << "Finished training" << std::endl;
}

int Agent::chooseAction(const torch::Tensor& state, DeepQNetwork& policyNet, bool isGreedyPolicy)
{
	bool exploit;
	if (isGreedyPolicy)
	{
		double rand = std::rand();
		if (epsilon > epsilonEnd)
			epsilon -= epsilonDec;
		std::cout << "EPSILON " << epsilon << std::endl;
		exploit = rand / RAND_MAX > epsilon;
	}
	else
		exploit = true;
	if (exploit) //rand/RAND_MAX is a number between 0.0 and 1.0
	{
		return policyNet->forward(state).argmax(1).item<int>();
	}
	else
	{
		return actionSpace[std::rand() % actionsNum];
	}
	
}

void Agent::storeExperience(const Experience& experience)
{
	this->memory->push(experience);
}

void Agent::learn(DeepQNetwork& policyNet, DeepQNetwork& targetNet, torch::optim::Adam& optimizer)
{
	if (memory->canProvideSample(batchSize))
	{
		//create vectors for creating a tensor then
		std::vector<torch::Tensor> states;
		std::vector<long long> actions;	
		std::vector<double> rewards;
		std::vector<torch::Tensor> newStates;
		std::vector<bool> dones;

		memory->sample(batchSize, states, actions, rewards, dones, newStates);
		
		auto statesTensor = torch::stack({ states }).to(this->device);
		auto newStatesTensor = torch::stack({ newStates }).to(this->device);

		auto currentQValues = policyNet->forward(statesTensor).to(this->device);

		auto nextQValues = targetNet->forward(newStatesTensor).to(this->device);
		//std::cout << "next Q" << nextQValues << std::endl;
		for (auto i = 0; i < batchSize; i++)
		{
			if (dones[i])
				for (auto j = 0; j < actionsNum; ++j)
					nextQValues[i][0][j] = 0.0;
		}
		auto nextQValuesMaxValues = std::get<0>(nextQValues.max(2)).to(this->device);
		//std::cout << "next Q" << nextQValues << std::endl;
		auto targetQValues = currentQValues.clone().to(this->device);
		for (auto i = 0; i < batchSize; i++)
		{
			targetQValues[i][0][actions[i]] = (nextQValuesMaxValues[i][0] * gamma + rewards[i]);
		}
		//std::cout << "target Q" << targetQValues << std::endl;
		//std::cout << "CURRENT Q" << currentQValues;
		auto loss = torch::nn::functional::mse_loss(currentQValues[0], targetQValues[0]);
		optimizer.zero_grad();
		std::cout << "loss " << loss << std::endl;
		loss.backward();
		optimizer.step();
	}
	else
		return; //wait for more experience
}