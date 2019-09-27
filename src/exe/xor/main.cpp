#include <iostream>
#include <mles/mles.hpp>

using namespace std;
using namespace mles;

int main(int argc, char *argv[])
{
    // Randomize
    std::srand((unsigned int) time(0));

    // Construct the NN model
    cout << "Creating NN1..." << endl;

    NN nn1(2, 1);
    nn1.verbose(false);
    nn1.setDefaultActivation("relu");
    nn1.addLayer(2); 
    nn1.build();

    DataSet trainingSet = nn1.createDataSet();

    auto entry = trainingSet.createEntry();

    entry << 0, 0, 0;
    trainingSet.add(entry);
    entry << 0, 1, 1;
    trainingSet.add(entry);
    entry << 1, 0, 1;
    trainingSet.add(entry);
    entry << 1, 1, 0;
    trainingSet.add(entry);

    cout << "Training set: " << endl;
    trainingSet.print();

    cout << "Training NN1..." << endl;
    TrainingSettings settings;
    settings.epochs = 100000;
    settings.maxError = 0.000001;
    TrainingResults results = nn1.train(trainingSet, settings);
    cout << "Training results NN1: " << endl;
    results.print();

    DataSet testSet1 = trainingSet.toTestSet();
    nn1.test(testSet1);
    testSet1.binarize();

    cout << "Test results NN1: " << endl;
    testSet1.print();

    cout << "Saving NN1..." << endl;
    nn1.save("xor");

    // Testing NN 2
    cout << "Loading NN2..." << endl;

    NN nn2;
    nn2.load("xor");

    DataSet testSet2 = trainingSet.toTestSet();
    nn2.test(testSet2);
    testSet2.binarize();

    cout << "Test results NN2: " << endl;
    testSet2.print();

    return 0;
}