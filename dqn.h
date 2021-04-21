// dqn.h : включаемый файл для стандартных системных включаемых файлов
// или включаемые файлы для конкретного проекта.

#pragma once
#include <torch/torch.h>
#include <iostream>

enum class NetworkMode { Training, Evaluation };

class DeepQNetworkImpl : public torch::nn::Module
{
public:
	DeepQNetworkImpl(const double learningRate, const int inputDims, const std::vector<int>& fcDims, const int actionsNum, const NetworkMode mode,
		const torch::Device& device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
	DeepQNetworkImpl() = delete; //we don't do that here
	torch::Tensor forward(const torch::Tensor& state);

private:
	const double learningRate;
	const int inputDims;
	const std::vector<int> fcDims;
	std::vector<torch::nn::Linear> fc;
	const int actionsNum;
	torch::Device device;
};

TORCH_MODULE(DeepQNetwork);

