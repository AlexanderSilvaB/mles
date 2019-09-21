#include "nnes.hpp"
#include "SigmoidActivation.hpp"
#include <iostream>
#include <fstream>

using namespace nnes;
using namespace std;
using namespace Eigen;

NN::NN()
{
    this->inputSize = 0;
    this->outputSize = 0;
    this->defaultActivation = new SigmoidActivation();
    this->inputActivation = this->defaultActivation;
    this->outputActivation = this->defaultActivation;
    this->isVerbose = false;
}

NN::NN(unsigned int inputSize, unsigned int outputSize)
{
    this->inputSize = inputSize;
    this->outputSize = outputSize;
    this->defaultActivation = new SigmoidActivation();
    this->inputActivation = this->defaultActivation;
    this->outputActivation = this->defaultActivation;
    this->isVerbose = false;
}

NN::~NN()
{
    if(this->defaultActivation != NULL)
        delete this->defaultActivation;
}

void NN::verbose(bool isVerbose)
{
    this->isVerbose = isVerbose;
}

void NN::setDefaultActivation(Activation *activation)
{
    if(activation == NULL)
    {
        activation = new SigmoidActivation();
    }
    if(inputActivation == defaultActivation)
        inputActivation = activation;
    if(outputActivation == defaultActivation)
        outputActivation = activation;
    for(int i = 0; i < layersActivation.size(); i++)
    {
        if(layersActivation[i] == defaultActivation)
            layersActivation[i] = activation;
    }
    if(defaultActivation != NULL)
        delete defaultActivation;
    defaultActivation = activation;
}

void NN::setInputSize(unsigned int size)
{
    inputSize = size;
}

void NN::setOutputSize(unsigned int size)
{
    outputSize = size;
}

void NN::setInputActivation(Activation* activation)
{
    if(activation == NULL)
        activation = defaultActivation;
    this->inputActivation = activation;
}

void NN::setOutputActivation(Activation* activation)
{
    if(activation == NULL)
        activation = defaultActivation;
    this->outputActivation = activation;
}

void NN::addLayer(unsigned int size, Activation* activation)
{
    layersSize.push_back(size);

    if(activation == NULL)
        activation = defaultActivation;

    layersActivation.push_back(activation);
}

void NN::insertLayer(unsigned int pos, unsigned int size, Activation* activation)
{
    vector<unsigned int>::iterator it = layersSize.begin();
    advance(it, pos);
    layersSize.insert(it, size);

    if(activation == NULL)
        activation = defaultActivation;

    vector<Activation*>::iterator itA = layersActivation.begin();
    advance(itA, pos);
    layersActivation.insert(itA, activation);
}

void NN::changeLayer(unsigned int pos, unsigned int size, Activation* activation)
{
    vector<unsigned int>::iterator it = layersSize.begin();
    advance(it, pos);
    *it = size;

    if(activation == NULL)
        activation = defaultActivation;

    vector<Activation*>::iterator itA = layersActivation.begin();
    advance(itA, pos);
    *itA = activation;
}

void NN::removeLayer(unsigned int pos)
{
    vector<unsigned int>::iterator it = layersSize.begin();
    advance(it, pos);
    layersSize.erase(it);
}

TrainingResults NN::train(DataSet& trainingSet, unsigned int epochs, double error, double learningRate)
{
    TrainingResults results;
    results.error = 1.0f;
    results.epochs = 0;

    MatrixXd X(trainingSet.size(), inputSize + 1);
    X.block(0, 0, trainingSet.size(), 1) = MatrixXd::Ones(trainingSet.size(), 1);

    MatrixXd y(trainingSet.size(), outputSize);

    for(unsigned int i = 0; i < trainingSet.size(); i++)
    {
        X.block(i, 1, 1, inputSize) = trainingSet.getInput(i).transpose();
        y.block(i, 0, 1, outputSize) = trainingSet.getOutput(i).transpose();
    }

    for(results.epochs = 0; results.epochs < epochs; results.epochs++)
    {
        MatrixXd a = X;
        for(unsigned int j = 0; j < layers.size(); j++)
        {
            a = layers[j].forward(a);
        }

        MatrixXd delta = layers.back().backward(y, NULL);
        for(int j = layers.size() - 2; j >= 0; j--)
        {
            delta = layers[j].backward(delta, &layers[j + 1]);
        }

        a = X;
        for(unsigned int j = 0; j < layers.size(); j++)
        {
            a = layers[j].update(learningRate, a);
        }

        results.error = layers.back().error(y);

        if(isVerbose)
            if(results.epochs % 1000 == 0)
                results.print();

        if(results.error < error)
            break;
    }


    return results;
}

void NN::test(DataSet& testSet)
{
    for(unsigned int i = 0; i < testSet.size(); i++)
    {
        testSet.getOutput(i) = test(testSet.getInput(i));
    }
}

VectorXd NN::test(const VectorXd& data)
{
    MatrixXd a(1, inputSize + 1);
    a.block(0, 1, 1, inputSize) = data.transpose();
    a(0,0) = 1;

    for(unsigned int j = 0; j < layers.size(); j++)
    {
        a = layers[j].forward(a);
    }

    return a;
}

bool NN::load(const std::string& fileName)
{
    ifstream f(fileName);

    // Default activation
    string name;
    f >> name;
    
    // Verbose
    f >> isVerbose;

    // Layers sizes
    int nLayers;
    int layerSize;

    f >> nLayers;

    layersSize.resize(nLayers - 2);
    layersActivation.resize(nLayers - 2);

    f >> inputSize;
    for(int i = 0; i < layersSize.size(); i++)
    {
        f >> layerSize;
        layersSize[i] = layerSize;
    }
    f >> outputSize;

    // Layers activation
    f >> name;
    inputActivation = defaultActivation;
    
    for(int i = 0; i < layersSize.size(); i++)
    {
        f >> name;
        layersActivation[i] = defaultActivation;
    }

    f >> name;
    outputActivation = defaultActivation;

    // Building
    build();

    // Layers
    for(int i = 0; i < layers.size(); i++)
        layers[i].load(f);

    return true;
}

bool NN::save(const std::string& fileName)
{
    ofstream f(fileName, ios::trunc);
    if(!f.good())
        return false;

    // Default activation
    f << defaultActivation->getName() << endl;

    // Verbose
    f << isVerbose << endl;

    // Layers size
    f << (2 + layersSize.size()) << " ";
    f << inputSize << " ";
    for(int i = 0; i < layersSize.size(); i++)
        f << layersSize[i] << " ";
    f << outputSize << endl;

    // Layers activation
    f << inputActivation->getName() << " ";
    for(int i = 0; i < layersActivation.size(); i++)
        f << layersActivation[i]->getName() << " ";
    f << outputActivation->getName() << endl;

    // Layers
    f << endl;

    for(int i = 0; i < layers.size(); i++)
        layers[i].write(f);

    return true;
}

void NN::build()
{
    // Creating structure
    layers.clear();

    vector<unsigned int> sizes = layersSize;
    sizes.insert(sizes.begin(), inputSize);
    sizes.insert(sizes.end(), outputSize);

    vector<Activation*> activations = layersActivation;
    activations.insert(activations.begin(), inputActivation);
    activations.insert(activations.end(), outputActivation);

    unsigned int i = 1;
    for(i = 1; i < sizes.size() - 1; i++)
    {
        if(activations[i - 1] == NULL)
            activations[i - 1] = defaultActivation;
        layers.push_back( Layer( i - 1, sizes[i - 1] + 1, sizes[i] + 1, activations[i - 1], false ) );
    }

    i--;
    if(activations[i] == NULL)
        activations[i] = defaultActivation;
    layers.push_back( Layer( i, sizes[i] + 1, sizes[i + 1], activations[i], true ) );
}

DataSet NN::createDataSet()
{
    return DataSet(inputSize, outputSize);
}