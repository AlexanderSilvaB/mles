#include <iostream>
#include <nnes/nnes.hpp>
#include <nnes/SigmoidActivation.hpp>
#include <nnes/ReluActivation.hpp>
#include <nnes/TanhActivation.hpp>
#include <cmath>

using namespace std;
using namespace nnes;

int main(int argc, char *argv[])
{
    NN nn(2, 1);
    nn.setDefaultActivation(new SigmoidActivation());
    nn.verbose(true);
    nn.addLayer(3); 
    nn.build();

    DataSet trainingSet = nn.createDataSet();
    auto entry = trainingSet.createEntry();

    double x1, x2, y;
    for(y = 0; y < 1; y += 0.1)
    {
        x1 = sin(3.14 * y);
        x2 = cos(3.14 * y);
        entry << x1, x2, y;
        trainingSet.add(entry);
    }

    cout << "Training..." << endl;
    TrainingResults results = nn.train(trainingSet, 100000, 0.001);
    cout << "Training results: " << endl;
    results.print();

    DataSet testSet = trainingSet.toTestSet();
    nn.test(testSet);

    cout << "Test results: " << endl;
    testSet.print();

    return 0;
}