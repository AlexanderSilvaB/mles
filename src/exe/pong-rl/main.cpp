#include <iostream>
#include <mles/mles.hpp>
#include "PongEnv.hpp"

using namespace std;
using namespace mles;

int main(int argc, char *argv[])
{
    std::srand((unsigned int) time(0));

    PongEnv pong;
    QLearning qlearning(&pong);
    if(!qlearning.load("pong"))
    {
        cout << "Training..." << endl;

        qlearning.setDiscountFactor(0.3);

        TrainingSettings settings;
        settings.epochs = 10000;
        qlearning.train(settings);
        cout << "Training finished!" << endl;
        
        qlearning.save("pong");
    }

    cout << "Running..." << endl;
    while(true)
    {
        qlearning.run();
    }

    return 0;
}