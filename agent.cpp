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

int Agent::chooseAction(const torch::Tensor& state, DeepQNetwork policyNet)
{
	double rand = std::rand();
	if (epsilon > epsilonEnd)
		epsilon -= epsilonDec;
	std::cout << "EPSILON " << epsilon << std::endl;
	if (rand / RAND_MAX > epsilon) //rand/RAND_MAX is a number between 0.0 and 1.0
	//if(1)
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

void Agent::learn(DeepQNetwork policyNet, DeepQNetwork targetNet, torch::optim::Adam& optimizer)
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
		
		auto statesTensor = torch::stack({ states });
		auto newStatesTensor = torch::stack({ newStates });
		
		/*std::cout << "statesTensor " << statesTensor << std::endl;
		std::cout << "actionsTensor " << actionsTensor << std::endl;
		std::cout << "rewardsTensor " << rewardsTensor << std::endl;
		std::cout << "newStatesTensor " << newStatesTensor << std::endl;*/

		//auto currentQValues = policyNet->forward(statesTensor).gather(2, actionsTensor.unsqueeze(-1)).squeeze(-1);
		auto currentQValues = policyNet->forward(statesTensor);
		//std::cout << "cur Q" << currentQValues << std::endl;

		
		auto nextQValues = targetNet->forward(newStatesTensor);
		//std::cout << "next Q" << nextQValues << std::endl;
		for (auto i = 0; i < batchSize; i++)
		{
			if (dones[i])
				for (auto j = 0; j < actionsNum; ++j)
					nextQValues[i][0][j] = 0.0;
		}
		auto nextQValuesMaxValues = std::get<0>(nextQValues.max(2));
		//std::cout << "next Q" << nextQValues << std::endl;
	
		//nextQValues = nextQValues.detach();

		
			auto targetQValues = currentQValues.clone();

			//std::cout << "target Q" << targetQValues << std::endl;
			for (auto i = 0; i < batchSize; i++)
			{
				targetQValues[i][0][actions[i]] = (nextQValuesMaxValues[i][0] * gamma + rewards[i]);
			}
		/*	std::cout << "target Q" << targetQValues << std::endl;
			std::cout << "CURRENT Q" << currentQValues;*/


			/*auto k = policyNet->parameters();
			for (auto& g : k)
			{
				std::cout << g;
			}*/
			auto loss = torch::nn::functional::mse_loss(currentQValues, targetQValues);
			optimizer.zero_grad();
			std::cout << "loss " << loss << std::endl;
			loss.backward();
			optimizer.step();

		/*	k = policyNet->parameters();
			for (auto& g : k)
			{
				std::cout << g;
			}*/
		
	}
	else
		return; //wait for more experience
}