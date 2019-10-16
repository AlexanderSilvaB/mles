#include <iostream>
#include <mles/mles.hpp>
#include "MNISTLoader.hpp"

using namespace std;
using namespace mles;

int main(int argc, char *argv[])
{
    // Randomize
    std::srand((unsigned int) time(0));

    NN nn;
    DataSet dataset = MNISTLoader::Load("data/mnist/", 10);

    DataSet trainingSet, testSet;
    dataset.split(trainingSet, testSet, 0.8);

    // Try to load the model
    bool loaded = nn.load("mnist");
    if(!loaded)
    {
        nn.setInputSize(dataset.getInputSize());
        nn.setOutputSize(dataset.getOutputSize());
        nn.setDefaultActivation("relu");
        nn.setOutputActivation("softmax");
        nn.verbose(true);
        nn.addLayer(32);
        nn.build();

        cout << "Training..." << endl;
        TrainingSettings settings;
        settings.epochs = 10;
        settings.batch = -1;
        settings.maxError = 0.02;
        settings.allowReset = false;
        TrainingResults results = nn.train(trainingSet, settings);
        cout << "Training results: " << endl;
        results.print();

        nn.save("mnist");
    }

    // Using the model
    cout << "Evaluation: " << endl;
    nn.test(testSet);
    testSet.binarizeToMax();
    testSet.print(true);
    

    return 0;
}