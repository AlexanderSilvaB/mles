#include "mles.hpp"
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

using namespace mles;
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
    this->outputActivation = this->defaultActivation;
    this->isVerbose = false;
    this->transformInput = false;
    this->transformOutput = false;
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

void NN::setOutputActivation(string name, ...)
{
    outputActivation = getActivation(name, true);
    LOAD_ARGS(outputActivation);
}

void NN::setInputTransformation(const Eigen::VectorXd& a, const Eigen::VectorXd& b)
{
    inputA = a;
    inputB = b;
    transformInput = true;
}

void NN::setOutputTransformation(const Eigen::VectorXd& a, const Eigen::VectorXd& b)
{
    outputA = a;
    outputB = b;
    transformOutput = true;
}

Eigen::VectorXd NN::createInputVector()
{
    return VectorXd(inputSize);
}

Eigen::VectorXd NN::createOutputVector()
{
    return VectorXd(outputSize);
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

void NN::reset()
{
    for(unsigned int j = 0; j < layers.size(); j++)
        layers[j].reset();
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
    results.errorCode = 0;

    if(trainingSet.getInputSize() != inputSize || trainingSet.getOutputSize() != outputSize)
    {
        results.finished = true;
        results.elapsedTime = 0;
        results.errorCode = 1;
        return results;
    }

    double thError = settings.maxError / 10000000.0;

    Eigen::VectorXd input, output;

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
                input = trainingSet.getInput(i);
                output = trainingSet.getOutput(i);
                if(transformInput)
                {
                    input = input.cwiseProduct(inputA) + inputB;
                }
                if(transformOutput)
                {
                    output = output.cwiseProduct(outputA) + outputB;
                }
                X.block(j, 1, 1, inputSize) = input.transpose();
                y.block(j, 0, 1, outputSize) = output.transpose();
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
        if(abs(error - lastError) < thError)
        {
            countError++;
        }
        else
        {
            countError = 0;
        }
        lastError = error;

        if(settings.allowReset && countError > settings.localMinimaLimit)
        {
            cout << "Reseting..." << endl;
            countError = 0;
            reset();
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
    VectorXd input = data;
    if(transformInput)
    {
        input = input.cwiseProduct(inputA) + inputB;
    }

    MatrixXd a(1, inputSize + 1);
    a.block(0, 1, 1, inputSize) = input.transpose();
    a(0,0) = 1;

    for(unsigned int j = 0; j < layers.size(); j++)
    {
        a = layers[j].forward(a);
    }

    VectorXd output = a.transpose();
    if(transformOutput)
    {
        output -= outputB;
        output = output.cwiseProduct(outputA.cwiseInverse());
    }

    return output;
}

bool NN::load(const std::string& fileName)
{
    ifstream f(fileName+".nn");
    if(!f.good())
        return false;

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

    // Transformations
    f >> transformInput;
    f >> transformOutput;
    if(transformInput)
    {
        inputA = createInputVector();
        inputB = createInputVector();
        double v;
        for(int i = 0; i < inputSize; i++)
        {
            f >> v;
            inputA(i) = v;
            f >> v;
            inputB(i) = v;
        }
    }
    if(transformOutput)
    {
        outputA = createOutputVector();
        outputB = createOutputVector();
        double v;
        for(int i = 0; i < outputSize; i++)
        {
            f >> v;
            outputA(i) = v;
            f >> v;
            outputB(i) = v;
        }
    }

    // Building
    build();

    // Layers
    for(int i = 0; i < layers.size(); i++)
        layers[i].load(f);

    return true;
}

bool NN::save(const std::string& fileName)
{
    ofstream f(fileName+".nn", ios::trunc);
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
    for(int i = 0; i < layersActivation.size(); i++)
    {
        f << layersActivation[i]->getName() << " ";
        layersActivation[i]->write(f);
        f << " ";
    }
    f << outputActivation->getName() << " ";
    outputActivation->write(f);
    f << endl;

    // Transformations
    f << transformInput << " " << transformOutput << endl;
    if(transformInput)
    {
        for(int i = 0; i < inputSize; i++)
            f << inputA(i) << " " << inputB(i) << " ";
    }
    f << endl;
    if(transformOutput)
    {
        for(int i = 0; i < outputSize; i++)
            f << outputA(i) << " " << outputB(i) << " ";
    }
    f << endl;

    // Layers
    f << endl;

    for(int i = 0; i < layers.size(); i++)
        layers[i].write(f);

    return true;
}

void NN::print()
{
    // Default activation
    cout << "Default activation: " << defaultActivation->getName() << endl;

    // Verbose
    cout << "Verbose: " << isVerbose << endl;

    // Layers size
    cout << "Input size: " << inputSize << endl;
    cout << "Output size: " << outputSize << endl;
    cout << "Output activation: " << outputActivation->getName() << endl;
    cout << "Hidden layers: " << layersSize.size() << endl;
    cout << "Hidden layers sizes: ";
    for(int i = 0; i < layersSize.size(); i++)
        cout << layersSize[i] << " ";
    cout << endl;

    // Layers activation
    cout << "Hidden layers activations: ";
    for(int i = 0; i < layersActivation.size(); i++)
        cout << layersActivation[i]->getName() << " ";
    cout << endl;

    // Layers
    cout << "Layers weights:" << endl;

    for(int i = 0; i < layers.size(); i++)
    {
        layers[i].print();
        cout << endl;
    }
}

void NN::build()
{
    // Creating structure
    layers.clear();

    vector<unsigned int> sizes = layersSize;
    sizes.insert(sizes.begin(), inputSize);
    sizes.insert(sizes.end(), outputSize);

    vector< ActivationPtr > activations = layersActivation;
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