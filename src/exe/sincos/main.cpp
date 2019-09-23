#include <iostream>
#include <nnes/nnes.hpp>
#include <cmath>

using namespace std;
using namespace nnes;

int main(int argc, char *argv[])
{
    // Randomize
    std::srand((unsigned int) time(0));

    // Construct the NN model
    NN nn(2, 1);
    nn.setDefaultActivation("relu");
    nn.setOutputActivation("linear", 1.0);
    nn.verbose(true);
    nn.addLayer(3); 
    nn.addLayer(5);
    nn.addLayer(3);
    nn.build();

    DataSet trainingSet = nn.createDataSet();
    DataSet testSet = nn.createDataSet();
    auto entry = trainingSet.createEntry();

    double x1, x2, y;
    for(y = 0; y < 1; y += 0.01)
    {
        x1 = sin(3.14 * y);
        x2 = cos(3.14 * y);
        entry << x1, x2, y;
        if( ((int)(y * 100)) % 5 == 0 )
            trainingSet.add(entry);
        else
            testSet.add(entry);
    }

    cout << "Training set: " << endl;
    trainingSet.print();

    cout << "Training..." << endl;
    TrainingSettings settings;
    settings.epochs = -1;
    settings.batch = 10;
    settings.maxError = 0.005;
    TrainingResults results = nn.train(trainingSet, settings);
    cout << "Training results: " << endl;
    results.print();

    nn.test(testSet);

    cout << "Test results: " << endl;
    testSet.print();

    nn.save("sincos.nn");

    return 0;
}