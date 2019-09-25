#include "QLearning.hpp"
#include <iostream>
#include <fstream>

using namespace std;
using namespace mles;
using namespace Eigen;

// QEnv
QEnv::QEnv()
{

}

QEnv::~QEnv()
{

}

int QEnv::getNumActions()
{
    return 0;
}

int QEnv::getNumStates()
{
    return 0;
}

int QEnv::getAction()
{
    return 0;
}

bool QEnv::step(int action, int& nextState, double& reward)
{
    return true;
}

int QEnv::reset()
{
    return 0;
}

bool QEnv::ready()
{
    return true;
}

void QEnv::toTrain()
{

}

void QEnv::toTest()
{

}

void QEnv::onDone()
{

}

// QLearning
QLearning::QLearning(QEnv* env)
{
    this->env = env;
    this->Q = MatrixXd(env->getNumStates(), env->getNumActions());
    this->Q.setZero();
    this->explorationFactor = 0.5;
    this->discountFactor = 0.6;
    this->exploitationFactor = 250.0;
}

QLearning::~QLearning()
{

}

void QLearning::train(const TrainingSettings& settings)
{
    bool done = false;
    int action, state, nextState;
    double reward, oldValue, newValue, nextMax;
    double explorationChance;
    double exploration;

    env->toTrain();

    for(int i = 0; i < settings.epochs; i++)
    {
        state = env->reset();
        done = false;
        while(!done)
        {
            if(!env->ready())
                continue;

            explorationChance = (rand() % 100) * 0.01;
            exploration = (explorationFactor * ( (1.0 * exploitationFactor) / i));
            if(exploration < 0.1)
                exploration = 0.1;
            // cout << i << " - " << exploration << endl;
            if(explorationChance <= exploration)
            {
                action = env->getAction();
            }
            else
            {
                Q.row(state).maxCoeff(&action);
            }

            done = env->step(action, nextState, reward);

            oldValue = Q(state, action);
            nextMax = Q.row(nextState).maxCoeff();

            newValue = (1.0 - settings.learningRate) * oldValue + settings.learningRate * ( reward + discountFactor * nextMax );
            Q(state, action) = newValue;
            // cout << "Q(" << state << ", " << action << ") " << oldValue << " -> " << newValue << endl;

            state = nextState;
        }
        env->onDone();

        if(i % 100 == 0)
            cout << "Epoch: " << i << endl;
    }
    cout << "Epoch: " << settings.epochs << endl;
}

void QLearning::run()
{
    bool done = false;
    int action, state;
    double reward;

    env->toTest();

    state = env->reset();
    while(!done)
    {
        if(!env->ready())
            continue;

        Q.row(state).maxCoeff(&action);
        done = env->step(action, state, reward);
    }
    env->onDone();
}

bool QLearning::load(const std::string& fileName)
{
    ifstream f(fileName+".ql");
    if(!f.good())
        return false;

    double w;
    for(int i = 0; i < Q.rows(); i++)
    {
        for(int j = 0; j < Q.cols(); j++)
        {
            f >> w;
            Q(i, j) = w;
        }
    }
    return true;
}

bool QLearning::save(const string& fileName)
{
    ofstream f(fileName+".ql", ios::trunc);
    if(!f.good())
        return false;

    for(int i = 0; i < Q.rows(); i++)
    {
        for(int j = 0; j < Q.cols(); j++)
            f << Q(i, j) << " ";
        f << endl;
    }
    return true;
}