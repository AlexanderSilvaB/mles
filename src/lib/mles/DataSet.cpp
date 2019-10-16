#include "DataSet.hpp"
#include <iostream>
#include <fstream>
#include <climits>
#include <map>
#include <cmath>

using namespace mles;
using namespace std;
using namespace Eigen;

DataSet::DataSet()
{
    this->inputSize = 0;
    this->outputSize = 0;
}

DataSet::DataSet(const DataSet& dataSet)
{
    this->inputSize = dataSet.inputSize;
    this->outputSize = dataSet.outputSize;
    this->inputs = dataSet.inputs;
    this->outputs = dataSet.outputs;
    this->inputHeader = dataSet.inputHeader;
    this->outputHeader = dataSet.outputHeader;
}

DataSet::DataSet(unsigned int inputSize, unsigned int outputSize)
{
    this->inputSize = inputSize;
    this->outputSize = outputSize;
    this->inputHeader.resize(inputSize);
    this->outputHeader.resize(outputSize);

    stringstream ss;
    for(unsigned int i = 0; i < inputSize; i++)
    {
        ss.str("I");
        ss << i;
        this->inputHeader[i] = ss.str();
    }
    for(unsigned int i = 0; i < outputSize; i++)
    {
        ss.str("O");
        ss << i;
        this->inputHeader[i] = ss.str();
    }
}

DataSet::~DataSet()
{

}

unsigned int DataSet::getInputSize()
{
    return inputSize;
}

unsigned int DataSet::getOutputSize()
{
    return outputSize;
}

void DataSet::split(DataSet& trainingSet, DataSet& testSet, double ratio)
{
    unsigned int sz = size();
    unsigned int trainSize = sz * ratio;

    trainingSet.inputSize = testSet.inputSize = inputSize;
    trainingSet.outputSize = testSet.outputSize = outputSize;
    trainingSet.inputHeader = testSet.inputHeader = inputHeader;
    trainingSet.outputHeader = testSet.outputHeader = outputHeader;
    trainingSet.clear();
    testSet.clear();

    for(unsigned int i = 0; i < sz; i++)
    {
        if(i < trainSize)
        {
            trainingSet.add(getInput(i), getOutput(i));
        }
        else
        {
            testSet.add(getInput(i), getOutput(i));
        }
    }
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
    string w;

    f >> sz;
    if(sz != inputSize)
        return false;
    
    f >> sz;
    if(sz != outputSize)
        return false;

    for(unsigned int i = 0; i < inputSize; i++)
    {
        f >> w;
        inputHeader[i] = w;
    }

    for(unsigned int i = 0; i < outputSize; i++)
    {
        f >> w;
        outputHeader[i] = w;
    }

    classMap.clear();

    for(unsigned int i = 0; i < inputSize; i++)
    {
        string h;
        f >> h;
        f >> sz;
        vector<string>& list = classMap[ this->inputHeader[i] ];
        for(unsigned int j = 0; j < sz; j++)
        {
            f >> w;
            list.push_back(w);
        }
    }

    for(unsigned int i = 0; i < outputSize; i++)
    {
        string h;
        f >> h;
        f >> sz;
        vector<string>& list = classMap[ this->outputHeader[i] ];
        for(unsigned int j = 0; j < sz; j++)
        {
            f >> w;
            list.push_back(w);
        }
    }

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

    for(unsigned int i = 0; i < inputSize; i++)
    {
        f << this->inputHeader[i];
        if(i < inputSize-1)
            f << " ";
    }
    f << " ";
    for(unsigned int i = 0; i < outputSize; i++)
    {
        f << this->outputHeader[i];
        if(i < outputSize-1)
            f << " ";
    }
    f << endl;

    for(unsigned int i = 0; i < inputSize; i++)
    {
        f << this->inputHeader[i] << " ";
        vector<string>& list = classMap[ this->inputHeader[i] ];
        f << list.size() << " ";
        for(unsigned int j = 0; j < list.size(); j++)
        {
            f << list[j];
            if(j < list.size()-1)
                f << " ";
        }
        f << endl;
    }
    for(unsigned int i = 0; i < outputSize; i++)
    {
        f << this->outputHeader[i] << " ";
        vector<string>& list = classMap[ this->outputHeader[i] ];
        f << list.size() << " ";
        for(unsigned int j = 0; j < list.size(); j++)
        {
            f << list[j];
            if(j < list.size()-1)
                f << " ";
        }
        f << endl;
    }

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

void DataSet::print(bool onlyOutput)
{
    if(!onlyOutput)
    {
        for(unsigned int i = 0; i < inputSize; i++)
        {
            cout << this->inputHeader[i];
            if(i < inputSize-1)
                cout << " ";
        }
        cout << " -> ";
        for(unsigned int i = 0; i < outputSize; i++)
        {
            cout << this->outputHeader[i];
            if(i < outputSize-1)
                cout << " ";
        }
        cout << endl;
    }
    for(unsigned int i = 0; i < inputs.size(); i++)
    {
        cout << i << ": ";
        if(!onlyOutput)
            cout << inputs[i].transpose() << " -> ";
        cout << outputs[i].transpose() << endl;
    }
}

void DataSet::binarize(double threshold)
{
    for(unsigned int i = 0; i < outputs.size(); i++)
    {
        for(unsigned int j = 0; j < outputSize; j++)
        {
            outputs[i][j] = (outputs[i][j] < threshold) ? 0 : 1;
        }
    }
}

void DataSet::classify()
{
    for(unsigned int i = 0; i < outputs.size(); i++)
    {
        for(unsigned int j = 0; j < outputSize; j++)
        {
            outputs[i][j] = round(outputs[i][j]);
        }
    }
}

void DataSet::binarizeToMax()
{
    int index;
    for(unsigned int i = 0; i < outputs.size(); i++)
    {
        outputs[i].maxCoeff(&index);
        outputs[i].setZero();
        outputs[i][index] = 1;
    }
}

bool DataSet::isNumeric(const std::string& text, double& value)
{
    stringstream ss(text);
    double v;
    ss >> v;
    ss = stringstream();
    ss << v;
    if(text == ss.str())
    {
        value = v;
        return true;
    }
    return false;
}

bool DataSet::fromCSV(const std::string& fileName, bool header, int chucksize)
{
    ifstream f(fileName);
    if(!f.good())
        return false;

    string line, word;
    stringstream ss;
    int i;
    double v;

    if(chucksize < 0)
        chucksize = INT_MAX;

    if(header)
    {
        getline(f, line);
        ss = stringstream(line);
        i = 0;
        while(getline(ss, word, ',')) 
        {
            if(i < inputSize)
                inputHeader[i] = word;
            else if(i < inputSize + outputSize)
                outputHeader[i - inputSize] = word;
            else
            {
                cout << "Invalid header column [" << word << "]"  << endl;
                return false;
            }
            i++;
        }
        if(i != inputSize + outputSize)
        {
            cout << "Invalid header size" << endl;
            return false;
        }
    }

    classMap.clear();

    int j = 1;
    while(j <= chucksize && getline(f, line))
    {
        ss = stringstream(line);
        i = 0;
        VectorXd entry = createEntry();
        while(std::getline(ss, word, ',')) 
        {
            if(i < inputSize + outputSize)
            {
                if(isNumeric(word, v))
                    entry[i] = v;
                else
                {
                    vector<string> list;
                    if(i < inputSize)
                        list = classMap[ inputHeader[i] ];
                    else
                        list = classMap[ outputHeader[i - inputSize] ];
                    
                    vector<string>::iterator it = find(list.begin(), list.end(), word);
                    if(it != list.end())
                    {
                        entry[i] = distance(list.begin(), it);
                    }
                    else
                    {
                        entry[i] = list.size();
                        if(i < inputSize)
                            classMap[ inputHeader[i] ].push_back(word);
                        else
                            classMap[ outputHeader[i - inputSize] ].push_back(word);

                    }
                }
            }
            i++;
        }
        if(i != inputSize + outputSize)
            cout << "Line #" << j << " was truncated [Invalid size]" << endl;
        add(entry);
        j++;
    }

    return true;
}

bool DataSet::toCSV(const std::string& fileName)
{
    ofstream f(fileName, ios::trunc);
    if(!f.good())
        return false;

    for(unsigned int i = 0; i < inputSize; i++)
        f << inputHeader[i] << ",";
    for(unsigned int i = 0; i < outputSize; i++)
    {
        f << outputHeader[i];
        if(i < outputSize-1)
            f << ",";
    }
    f << endl;
    for(unsigned int i = 0; i < size(); i++)
    {
        VectorXd& in = getInput(i);
        VectorXd& out = getOutput(i);
        for(unsigned int j = 0; j < inputSize; j++)
        {
            vector<string>& list = classMap[ inputHeader[j] ];
            if(list.size() > 0 && (int)in[j] < list.size())
                f << list[ (int)in[j] ];
            else
                f << in[j];
            f << ",";
        }
        for(unsigned int j = 0; j < outputSize; j++)
        {
            vector<string>& list = classMap[ outputHeader[j] ];
            if(list.size() > 0 && (int)out[j] < list.size())
                f << list[ (int)out[j] ];
            else
                f << out[j];
            if(j < outputSize-1)
                f << ",";
        }
        f << endl;
    }

    return true;
}