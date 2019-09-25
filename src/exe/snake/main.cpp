#include <iostream>
#include <mles/mles.hpp>
#include "SnakeEnv.hpp"

using namespace std;
using namespace mles;

int main(int argc, char *argv[])
{
    std::srand((unsigned int) time(0));

    SnakeEnv snake;
    QLearning qlearning(&snake);
    if(!qlearning.load("snake"))
    {
        cout << "Training..." << endl;
        TrainingSettings settings;
        settings.epochs = 10000;
        qlearning.train(settings);
        cout << "Training finished!" << endl;
        
        qlearning.save("snake");
    }

    cout << "Running..." << endl;
    while(true)
    {
        qlearning.run();
    }

    return 0;
}