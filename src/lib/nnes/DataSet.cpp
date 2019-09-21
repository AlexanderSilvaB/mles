#include "DataSet.hpp"
#include <iostream>

using namespace nnes;
using namespace std;
using namespace Eigen;

DataSet::DataSet(const DataSet& dataSet)
{
    this->inputSize = dataSet.inputSize;
    this->outputSize = dataSet.outputSize;
    this->inputs = dataSet.inputs;
    this->outputs = dataSet.outputs;
}

DataSet::DataSet(unsigned int inputSize, unsigned int outputSize)
{
    this->inputSize = inputSize;
    this->outputSize = outputSize;
}

DataSet::~DataSet()
{

}

VectorXd DataSet::createInput()
{
    return VectorXd(inputSize);
}

VectorXd DataSet::createOutput()
{
    return VectorXd(outputSize);
}

VectorXd DataSet::createEntry()
{
    return VectorXd(inputSize + outputSize);
}

void DataSet::add(const VectorXd& input, const VectorXd& output)
{
    if(input.rows() != inputSize)
        throw;
    else if(output.rows() != outputSize)
        throw;

    inputs.push_back(input);
    outputs.push_back(output);
}

void DataSet::add(const VectorXd& entry)
{
    if(entry.rows() != inputSize + outputSize)
        throw;

    inputs.push_back(entry.block(0,0,inputSize,1));
    outputs.push_back(entry.block(inputSize,0,outputSize,1));
}

unsigned int DataSet::size()
{
    return inputs.size();
}

VectorXd& DataSet::getInput(unsigned int pos)
{
    return inputs[pos];
}

VectorXd& DataSet::getOutput(unsigned int pos)
{
    return outputs[pos];
}

DataSet DataSet::toTestSet()
{
    DataSet testSet(*this);
    
    for(unsigned int i = 0; i < testSet.size(); i++)
        testSet.outputs[i].setZero();

    return testSet;
}

void DataSet::print()
{
    for(unsigned int i = 0; i < inputs.size(); i++)
    {
        cout << "Entry " << i << ": " << inputs[i].transpose() << " -> " << outputs[i].transpose() << endl;
    }
}