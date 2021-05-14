#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <fstream>

class CSVRow
{
public:
	float operator[](std::size_t index);
	std::size_t size() const;
	void readNextRow(std::istream& str);
private:
	std::vector<std::string>    m_data;
};

std::istream& operator>>(std::istream& str, CSVRow& data);

std::pair<std::vector<float>, int> getData(std::ifstream& file);

enum class OperatingMode { Training, Testing };

class Enviroment
{
public:
	Enviroment(const std::string& file);
	Enviroment() = delete;
	bool reset();
	double step(int action);
	std::vector<float> Enviroment::getState();
	bool isDone();

private:
	std::ifstream fileStream;
	std::vector<float> state;
	OperatingMode mode;
	int stepNum = 0;
	int label = 0;
	bool done = false;
};
