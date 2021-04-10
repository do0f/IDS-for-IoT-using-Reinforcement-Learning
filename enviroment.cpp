#include "enviroment.h"

float CSVRow::operator[](std::size_t index)
{
	std::string& eg = m_data[index];
	return std::atof(eg.c_str());
}

std::size_t CSVRow::size() const
{
	return m_data.size();
}

void CSVRow::readNextRow(std::istream& str)
{
	std::string         line;
	std::getline(str, line);

	std::stringstream   lineStream(line);
	std::string         cell;

	m_data.clear();
	while (std::getline(lineStream, cell, ','))
	{
		m_data.push_back(cell);
	}
	// This checks for a trailing comma with no data after it.
	if (!lineStream && cell.empty())
	{
		// If there was a trailing comma then add an empty element.
		m_data.push_back("");
	}
}

std::istream& operator>>(std::istream& str, CSVRow& data)
{
	data.readNextRow(str);
	return str;
}

std::pair<std::vector<float>, int> getData(std::ifstream& file) {
	std::vector<float> features;
	int label;

	CSVRow  row;
	file >> row;

	for (std::size_t loop = 0; loop < row.size() - 1; ++loop) {
		features.emplace_back(row[loop]);
	}
	// Push final column to label vector
	label = row[row.size() - 1];

	return std::make_pair(features, label);
}

Enviroment::Enviroment(const std::string& file, OperatingMode mode) : mode(mode)
{
	fileStream.open(file);
	getData(fileStream); //throw out first one
}
void Enviroment::reset() //new episode
{
	stepNum = 0;
	done = false;
}

double Enviroment::step(int action)
{
	stepNum++;
	auto reward = action == label ? 1.0 / stepNum : -1.0; //if agent is mistaking, reward always will be -1
														//else reward is positive, but gets smaller the more steps agent does
	if (reward > 0.0)
		done = true;
	else
		done = false;
	return reward;
}

std::vector<float> Enviroment::getState()
{
	if (stepNum == 0) //set new state
	{
		auto data = getData(fileStream);
		label = data.second;
		state = data.first;
	}
	
	return state; //else return the old one
}

bool Enviroment::isDone()
{
	return done;
}