#include "nnes.hpp"
#include "SigmoidActivation.hpp"
#include "AsinhActivation.hpp"
#include "AtanActivation.hpp"
#include "BentIActivation.hpp"
#include "IsruActivation.hpp"
#include "ReluActivation.hpp"
#include "SoftPlusActivation.hpp"
#include "TanhActivation.hpp"
#include "LinearActivation.hpp"
#include "SoftMaxActivation.hpp"
#include <iostream>
#include <fstream>
#include <climits>
#include <chrono>
#include <sstream>

using namespace nnes;
using namespace std;
using namespace Eigen;

#define LOAD_ARGS(act) {            \
                            stringstream ss; \
                            int sz = act->getParametersSize(); \
                            va_list ap; \
                            va_start(ap, name); \
                            for(int j = 0; j < sz; j++) \
                            { \
                                ss << (double)va_arg(ap, double); \
                                ss << " "; \
                            } \
                            va_end(ap); \
                            act->load(ss); \
                        }

NN::NN()
{
    this->inputSize = 0;
    this->outputSize = 0;
    init();
}

NN::NN(unsigned int inputSize, unsigned int outputSize)
{
    this->inputSize = inputSize;
    this->outputSize = outputSize;
    init();
}

NN::~NN()
{

}

void NN::init()
{
    registerActivation<SigmoidActivation>();
    registerActivation<AsinhActivation>();
    registerActivation<AtanActivation>();
    registerActivation<BentIActivation>();
    registerActivation<IsruActivation>();
    registerActivation<ReluActivation>();
    registerActivation<SoftPlusActivation>();
    registerActivation<TanhActivation>();
    registerActivation<LinearActivation>();
    registerActivation<SoftMaxActivation>();

    this->defaultActivation = ActivationPtr(new SigmoidActivation());
    this->inputActivation = this->defaultActivation;
    this->outputActivation = this->defaultActivation;
    this->isVerbose = false;
}

ActivationPtr NN::getActivation(const string& name, bool allowDefault)
{
    for(int i = 0; i < supportedActivations.size(); i++)
    {
        if(supportedActivations[i]->is(name))
        {
            return ActivationPtr(supportedActivations[i]->clone());
        }
    }
    if(allowDefault)
        return defaultActivation;
    return ActivationPtr();
}

void NN::verbose(bool isVerbose)
{
    this->isVerbose = isVerbose;
}

void NN::setDefaultActivation(string name, ...)
{
    defaultActivation = getActivation(name, true);
    LOAD_ARGS(defaultActivation);

    inputActivation = defaultActivation;
    outputActivation = defaultActivation;
}

void NN::setInputSize(unsigned int size)
{
    inputSize = size;
}

void NN::setOutputSize(unsigned int size)
{
    outputSize = size;
}

void NN::setInputActivation(string name, ...)
{
    inputActivation = getActivation(name, true);
    LOAD_ARGS(inputActivation);
}

void NN::setOutputActivation(string name, ...)
{
    outputActivation = getActivation(name, true);
    LOAD_ARGS(outputActivation);
}

void NN::addLayer(unsigned int size, string name, ...)
{
    layersSize.push_back(size);
    ActivationPtr act = getActivation(name, true);
    LOAD_ARGS(act);
    layersActivation.push_back(act);
}

void NN::insertLayer(unsigned int pos, unsigned int size, string name, ...)
{
    vector<unsigned int>::iterator it = layersSize.begin();
    advance(it, pos);
    layersSize.insert(it, size);

    ActivationPtr act = getActivation(name, true);
    LOAD_ARGS(act);

    vector< ActivationPtr >::iterator itA = layersActivation.begin();
    advance(itA, pos);
    layersActivation.insert(itA, act);
}

void NN::changeLayer(unsigned int pos, unsigned int size, string name, ...)
{
    vector<unsigned int>::iterator it = layersSize.begin();
    advance(it, pos);
    *it = size;

    ActivationPtr act = getActivation(name, true);
    LOAD_ARGS(act);

    vector< ActivationPtr >::iterator itA = layersActivation.begin();
    advance(itA, pos);
    *itA = act;
}

void NN::removeLayer(unsigned int pos)
{
    vector<unsigned int>::iterator it = layersSize.begin();
    advance(it, pos);
    layersSize.erase(it);
}

TrainingResults NN::train(DataSet& trainingSet, const TrainingSettings& settings)
{
    unsigned int epochs = 0;
    if(settings.epochs < 0)
        epochs = INT_MAX;
    else
        epochs = settings.epochs;

    MatrixXd X, y;

    int batchSize, pickSize;
    if(settings.batch <= 0)
    {
        batchSize = trainingSet.size();
    }
    else
    {
        batchSize = settings.batch;
    }
    
    double error = 0, lastError = 0;
    int countError = 0;

    TrainingResults results;
    results.error = 1.0f;
    results.epochs = 0;
    results.finished = false;
    results.elapsedTime = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for(results.epochs = 0; results.epochs < epochs; results.epochs++)
    {
        error = 0;
        for(int n = 0; n < trainingSet.size(); n += batchSize)
        {
            pickSize = trainingSet.size() - n;
            if(pickSize > batchSize)
                pickSize = batchSize;
            
            X = MatrixXd(pickSize, inputSize + 1);
            X.block(0, 0, pickSize, 1) = MatrixXd::Ones(pickSize, 1);

            y = MatrixXd(pickSize, outputSize);

            for(unsigned int i = n, j = 0; j < pickSize; i++, j++)
            {
                X.block(j, 1, 1, inputSize) = trainingSet.getInput(i).transpose();
                y.block(j, 0, 1, outputSize) = trainingSet.getOutput(i).transpose();
            }

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
                a = layers[j].update(settings.learningRate, a);
            }

            error += layers.back().error(y) / pickSize;
        }
        results.error = error;
        if(error == lastError)
        {
            countError++;
        }
        lastError = error;

        if(countError > settings.localMinimaLimit)
        {
            countError = 0;
            build();
        }

        if(isVerbose)
            if(results.epochs % 1000 == 0)
                results.print();

        if(results.error < settings.maxError)
                break;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    results.elapsedTime = chrono::duration_cast<chrono::microseconds>(elapsed).count() / 1000000.0;

    results.finished = true;


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

    ActivationPtr activation;

    // Default activation
    string name;
    f >> name;
    activation = getActivation(name);
    if(!activation)
        return false;
    activation->load(f);
    defaultActivation = activation;
    
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
    activation = getActivation(name);
    if(!activation)
        return false;
    activation->load(f);
    inputActivation = activation;
    
    for(int i = 0; i < layersSize.size(); i++)
    {
        f >> name;
        activation = getActivation(name);
        if(!activation)
            return false;
        activation->load(f);
        layersActivation[i] = activation;
    }

    f >> name;
    activation = getActivation(name);
    if(!activation)
        return false;
    activation->load(f);
    outputActivation = activation;

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
    f << defaultActivation->getName() << " ";
    defaultActivation->write(f);
    f << endl;

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
    inputActivation->write(f);
    f << " ";
    for(int i = 0; i < layersActivation.size(); i++)
    {
        f << layersActivation[i]->getName() << " ";
        layersActivation[i]->write(f);
        f << " ";
    }
    f << outputActivation->getName() << " ";
    outputActivation->write(f);
    f << endl;

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

    vector< ActivationPtr > activations = layersActivation;
    activations.insert(activations.begin(), inputActivation);
    activations.insert(activations.end(), outputActivation);

    unsigned int i = 1;
    for(i = 1; i < sizes.size() - 1; i++)
    {
        layers.push_back( Layer( i - 1, sizes[i - 1] + 1, sizes[i] + 1, activations[i - 1], false ) );
    }

    i--;
    layers.push_back( Layer( i, sizes[i] + 1, sizes[i + 1], activations[i], true ) );
}

DataSet NN::createDataSet()
{
    return DataSet(inputSize, outputSize);
}