#pragma once
#include <random>
#include <torch/torch.h>

//state, action, reward, newState
struct Experience
{
public:
	Experience(const torch::Tensor& state, const long long int action, const double reward, const torch::Tensor& newState, const bool done);
	Experience() = delete;
	~Experience();

	std::tuple<torch::Tensor, long long, double, torch::Tensor, bool> experience;
};

//state, action, reward, newState
class ExperienceMemory
{
public:
	ExperienceMemory(const std::size_t capacity);
	ExperienceMemory() = delete;
	~ExperienceMemory();
	void push(const Experience& experience);
	void ExperienceMemory::sample(const int batchSize, std::vector<torch::Tensor>& states, std::vector<long long>& actions,
		std::vector<double>& rewards, std::vector<bool>& dones, std::vector<torch::Tensor>& newStates);
	bool canProvideSample(const int batchSize);
private:
	std::vector<Experience> memory;
	const std::size_t capacity;
	std::size_t pushCount = 0;
	std::vector<std::size_t> memoryIndexes; //helper for picking random sample

};
