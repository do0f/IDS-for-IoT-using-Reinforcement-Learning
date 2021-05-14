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

int main(int argc, char** argv) //mode, .csv name
{

	if (argc < 3)
	{
		std::cout << "Too few parameters\n";
		return -1;
	}
	std::string mode = argv[1];
	std::cout << "Current mode: " << mode << "\n";

	std::ios::sync_with_stdio(false);
	//hyperparameters
	constexpr auto batchSize = 64;
	constexpr auto gamma = 0.999;
	constexpr auto epsilon = 1.0;
	constexpr auto epsilonEnd = 0.01;
	constexpr auto epsilonDec = 0.001;
	constexpr auto targetUpdate = 128;
	constexpr auto memorySize = 1'000ULL;
	constexpr auto learningRate = 0.001;
	constexpr auto numEpisodes = 1499;
	auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;


	//enviroment parameters
	constexpr auto inputDims = 10;
	std::vector<int>fcDims{ 32, 32 };
	constexpr auto actionsNum = 2;

	if (mode == "training") //train
	{
		Agent agent(gamma, epsilon, learningRate, inputDims, batchSize, actionsNum, memorySize, epsilonEnd, epsilonDec, device);
		DeepQNetwork policyNet(learningRate, inputDims, fcDims, actionsNum, NetworkMode::Training, device);
		DeepQNetwork targetNet(learningRate, inputDims, fcDims, actionsNum, NetworkMode::Evaluation, device);

		torch::save(policyNet, "firstPolicyNet.pt");
		torch::load(targetNet, "firstPolicyNet.pt");

		torch::optim::Adam optimizer(policyNet->parameters(), learningRate);
		std::string trainName = argv[2];
		Enviroment env(trainName);

		for (auto i = 0; i < numEpisodes; i++) //one episode is one line in dataset -> episode continues until agent chooses right action
		{
			auto stepNum = 0;
			auto actionNum = 0;
			if (env.reset() == 0)
				break;
			auto state = torch::from_blob(env.getState().data(), { 1, inputDims }).clone().to(device);

			while (!env.isDone())
			{
				auto action = agent.chooseAction(state, policyNet, true);
				auto reward = env.step(action);
				auto nextState = torch::from_blob(env.getState().data(), { 1, inputDims }).clone().to(device);

				auto l = policyNet->forward(state).to(device);
				std::cout << "Q values	\n" << l << std::endl;

				agent.storeExperience({ state, action, reward, nextState, env.isDone() });
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
					torch::save(policyNet, "policyNet.pt");
					torch::load(targetNet, "policyNet.pt");
				}
			}

			if (i % targetUpdate == 0)
			{
				std::cout << "Saving weights and updating target..." << std::endl;
				torch::save(policyNet, "policyNet.pt");
				torch::load(targetNet, "policyNet.pt");
			}
		}
		std::cout << "Finished training. Saving weights." << std::endl;
		torch::save(policyNet, "policyNet.pt");
	}
	else
	if (mode == "testing") //test
	{
		Agent agent(gamma, epsilon, learningRate, inputDims, batchSize, actionsNum, memorySize, epsilonEnd, epsilonDec, device);
		DeepQNetwork policyNet(learningRate, inputDims, fcDims, actionsNum, NetworkMode::Evaluation, device);
		torch::load(policyNet, "policyNet.pt");
		const std::string testName = argv[2];
		const std::string testResultName("testResult.txt");
		Enviroment env(testName);

		unsigned TP = 0, FP = 0, TN=0, FN=0;
		for (auto i = 0; i < numEpisodes; i++)
		{
			if (env.reset() == 0)
				break;
			auto state = torch::from_blob(env.getState().data(), { 1, inputDims }).clone().to(device);
			auto action = agent.chooseAction(state, policyNet, false);
			auto reward = env.step(action);

			if (reward > 0)
			{
				auto label = action == 0 ? 0 : 1;
				std::cout << "Action: " << action << "	Label:" << label << std::endl;
				if (action == 0)
					TP++;//state is normal -> agent marks it normal
				else
					TN++;//state is malicious -> agent marks it malicious
				std::cout << "Agent is right!\n" << std::endl;
			}
			else
			{
				auto label = action == 0 ? 1 : 0;
				std::cout << "Action: " << action << "	Label:" << label << std::endl;
				if (action == 0) 
					FP++;//state is malicious -> agent marks it normal
				else
					FN++; //state is normal -> agent marks it malicious
				std::cout << "Agent is wrong!" << std::endl;
			}
		}
		auto accuracy = static_cast<double>(TP+TN) /
			static_cast<double>(TP + TN + FP + FN);
		auto precision = static_cast<double>(TP) /
			static_cast<double>(TP + FP);
		auto recall = static_cast<double>(TP)/
			static_cast<double>(TP + FN);


		std::cout << "Finished testing. Saving results." << std::endl;

		std::ofstream res(testResultName);
		res << "TP: " << TP << '\n' << "TN: " << TN << '\n' << "FP: " << FP << '\n' << "FN: " << FN << '\n';
		res << "accuracy: " << accuracy << '\n' << "precision: " << precision << '\n' << "recall: " << recall << '\n';
	}
	else
		std::cout << "unknown mode" << "\n";
	return 0;
}