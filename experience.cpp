#include "experience.h"


Experience::Experience(const torch::Tensor& state, const long long action, const double reward, const torch::Tensor& newState, const bool done) :
	experience(std::make_tuple(state, action, reward, newState, done))
{
}
Experience::~Experience()
{
}


ExperienceMemory::ExperienceMemory(const std::size_t capacity) : capacity(capacity)
{
	memory.reserve(capacity);

	memoryIndexes.reserve(capacity);
}

ExperienceMemory::~ExperienceMemory() {}
void ExperienceMemory::push(const Experience& experience)
{
	if (memory.size() < capacity)
	{
		memory.emplace_back(experience);
		memoryIndexes.push_back(pushCount);
	}
	else
		memory[pushCount % capacity] = experience;

	pushCount++;
}

void ExperienceMemory::sample(const int batchSize, std::vector<torch::Tensor>& states, std::vector<long long int>& actions,
	std::vector<double>& rewards, std::vector<bool>& dones, std::vector<torch::Tensor>& newStates)
{
	std::shuffle(memoryIndexes.begin(), memoryIndexes.end(),
		std::mt19937{ std::random_device{}() });
	states.reserve(batchSize);
	actions.reserve(batchSize);
	rewards.reserve(batchSize);
	newStates.reserve(batchSize);
	dones.reserve(batchSize);

	for (auto i = 0; i != batchSize; ++i)
	{
		auto index = memoryIndexes[i];
		states.push_back(std::get<0>(memory[index].experience));
		actions.push_back(std::get<1>(memory[index].experience));
		rewards.push_back(std::get<2>(memory[index].experience));
		newStates.push_back(std::get<3>(memory[index].experience));
		dones.push_back(std::get<4>(memory[index].experience));
	}
}

bool ExperienceMemory::canProvideSample(const int batchSize)
{
	return memory.size() >= batchSize;
}

