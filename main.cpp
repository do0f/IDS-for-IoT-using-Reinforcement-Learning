#include <iostream>
#include <torch/torch.h>
#include <memory>
#include <vector>
#include <random>
#include <algorithm>
#include <tuple>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include "agent.h"
#include "dqn.h"
#include "enviroment.h"
#include "experience.h"




int main()
{
	//hyperparameters
	constexpr auto batchSize = 16;
	constexpr auto gamma = 0.999;
	constexpr auto epsilon = 1.0;
	constexpr auto epsilonEnd = 0.01;
	constexpr auto epsilonDec = 0.001;
	constexpr auto targetUpdate = 64;
	constexpr auto memorySize = 1'000ULL;
	constexpr auto learningRate = 0.001;
	constexpr auto numEpisodes = 6'000;
	auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

	//enviroment parameters
	constexpr auto inputDims = 10;
	std::vector<int>fcDims{ 256, 256 };
	constexpr auto actionsNum = 2;
	Agent agent(gamma, epsilon, learningRate, inputDims, batchSize, actionsNum, memorySize, epsilonEnd, epsilonDec, device);

	DeepQNetwork policyNet(learningRate, inputDims, fcDims, actionsNum, NetworkMode::Training, device);

	DeepQNetwork targetNet(learningRate, inputDims, fcDims, actionsNum, NetworkMode::Evaluation, device);

	torch::save(policyNet, "D:/firstPolicyNet.pt");
	torch::load(targetNet, "D:/firstPolicyNet.pt");

	torch::optim::Adam optimizer(policyNet->parameters(), learningRate);

	Enviroment env("D:/dataset_norm.csv", OperatingMode::Training);


	for (auto i = 0; i < numEpisodes; i++) //one episode is one line in dataset -> episode continues until agent chooses right action
	{
		auto stepNum = 0;
		auto actionNum = 0;
		env.reset();

		auto state = torch::from_blob(env.getState().data(), { 1, inputDims }).clone().to(device);

		while (!env.isDone())
		{
			auto action = agent.chooseAction(state, policyNet);
			auto reward = env.step(action);
			auto nextState = torch::from_blob(env.getState().data(), { 1, inputDims }).clone().to(device);

			auto l = policyNet->forward(state).to(device);
			std::cout <<"Q values	\n"<< l << std::endl;

			/*if(actionNum == 0)*/
			/*std::cout << "state " << state << std::endl << "action " << action << std::endl<<"reward " << reward << std::endl
				<< "next state " << nextState <<std::endl << "done " << env.isDone() << std::endl;*/
			agent.storeExperience({state, action, reward, nextState, env.isDone()});
			state = nextState;

			agent.learn(policyNet, targetNet, optimizer);

			++stepNum;

			if (reward > 0)
			{
				auto label = action == 0 ? 0 : 1;
				std::cout << "Action: " << action << "	Label:" << label << std::endl;
				std::cout << "Agent is right! Label was set correctly at " << actionNum << " attempt\n\n\n" << std::endl;
			}
			else
			{
				auto label = action == 0 ? 1 : 0;
				std::cout << "Action: " << action << "	Label:" << label << std::endl;
				std::cout << "Agent is wrong! Being wrong for " << actionNum << " attempts..." << std::endl;
			}
			++actionNum;
			//update targetNetwork every x steps
			if (stepNum % targetUpdate == 0)
			{
				std::cout << "Saving weights and updating target..." << std::endl;
				torch::save(policyNet, "D:/policyNet.pt");
				torch::load(targetNet, "D:/policyNet.pt");
			}
		}
	}
	return 0;
}