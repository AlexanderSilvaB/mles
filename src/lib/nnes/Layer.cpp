#include "Layer.hpp"
#include "SigmoidActivation.hpp"
#include <iostream>

using namespace nnes;
using namespace std;
using namespace Eigen;

Layer::Layer()
{

}

Layer::Layer(unsigned int id, unsigned int inputSize, unsigned int outputSize, ActivationPtr activation, bool isOutput)
{
    this->id = id;
    this->isOutput = isOutput;
    weight = MatrixXd::Random(inputSize, outputSize);
    this->activation = activation;
    // cout << id << endl;
    // cout << weight << endl;
}

Layer::~Layer()
{
    
}

double Layer::error(const Eigen::MatrixXd& y)
{
    MatrixXd error = A - y;
    return error.norm();
}

MatrixXd Layer::forward(const MatrixXd& x)
{
    // cout << "X: " << x << endl;
    // cout << "weight: " << weight << endl;
    z = x * weight;
    // cout << "z: " << z << endl;
    activation->getActivation(z, A);
    activation->getDerivative(z, dz);

    // cout << A << endl;
    return A;
}

MatrixXd Layer::backward(const MatrixXd& y, Layer *layer)
{
    MatrixXd error;
    if(isOutput)
    {
        error = A - y;
        delta = error.cwiseProduct(dz);
    }
    else
    {
        error = layer->delta * layer->weight.transpose();
        delta = error.cwiseProduct(dz);
    }
    return delta;
}

MatrixXd Layer::update(double learningRate, const MatrixXd& a)
{
    auto ad = a.transpose() * delta;
    weight -= learningRate * ad;
    return A;
}

void Layer::load(ifstream& f)
{
    double w;
    for(int i = 0; i < weight.rows(); i++)
    {
        for(int j = 0; j < weight.cols(); j++)
        {
            f >> w;
            weight(i, j) = w;
        }
    }
}

void Layer::write(ofstream& f)
{
    for(int i = 0; i < weight.rows(); i++)
    {
        for(int j = 0; j < weight.cols(); j++)
            f << weight(i, j) << " ";
    }
    f << endl;
}

void Layer::print()
{
    cout << weight << endl;
}