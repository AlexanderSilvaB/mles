#include "DataSet.hpp"
#include <iostream>
#include <fstream>

using namespace mles;
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

void DataSet::clear()
{
    inputs.clear();
    outputs.clear();
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

unsigned int DataSet::getOutputMaxIndex(unsigned int pos)
{
    VectorXd& o = getOutput(pos);
    unsigned int p = 0;
    double m = 0;
    for(int i = 0; i < o.rows(); i++)
    {
        for(int j = 0; j < o.cols(); j++)
        {
            if(i == 0 && j == 0)
            {
                p = i * o.cols() + j;
                m = o(i, j);
            }
            else if(o(i, j) > m)
            {
                p = i * o.cols() + j;
                m = o(i, j);
            }
        }
    }
    return p;
}

bool DataSet::load(const std::string& fileName)
{
    ifstream f(fileName+".dataset");
    if(!f.good())
        return false;

    int sz;
    double v;

    f >> sz;
    if(sz != inputSize)
        return false;
    
    f >> sz;
    if(sz != outputSize)
        return false;

    f >> sz;

    clear();

    VectorXd input = createInput();
    VectorXd output = createOutput();

    for(unsigned int i = 0; i < sz; i++)
    {
        for(unsigned int j = 0; j < inputSize; j++)
        {
            f >> v;
            input[j] = v;
        }
        for(unsigned int j = 0; j < outputSize; j++)
        {
            f >> v;
            output[j] = v;
        }
        add(input, output);
    }

    return true;
}

bool DataSet::save(const std::string& fileName)
{
    ofstream f(fileName+".dataset", ios::trunc);
    if(!f.good())
        return false;

    f << inputSize << " " << outputSize << endl;

    f << size() << endl;

    for(unsigned int i = 0; i < inputs.size(); i++)
    {
        f << inputs[i].transpose() << " " << outputs[i].transpose() << endl;
    }

    return true;
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