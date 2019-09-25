#include <iostream>
#include <mles/mles.hpp>

using namespace std;
using namespace mles;

int main(int argc, char *argv[])
{
    // Randomize
    std::srand((unsigned int) time(0));

    NN nn;
    double x, y;

    // Try to load the model
    bool loaded = nn.load("quad");
    if(!loaded)
    {
        // Construct the NN model if not loaded
        nn.setInputSize(2);
        nn.setOutputSize(4);
        nn.setDefaultActivation("sigmoid");
        nn.verbose(true); 
        nn.addLayer(4);
        nn.addLayer(6);
        nn.addLayer(4);
        nn.build();

        DataSet trainingSet = nn.createDataSet();
        auto entry = trainingSet.createEntry();

        double op1, op2, op3, op4;
        for(int i = 0; i < 100; i++)
        {
            x = (rand() % 200) - 100;
            y = (rand() % 200) - 100;
            op1 = op2 = op3 = op4 = 0;
            if(x >= 0 && y >= 0)
                op1 = 1;
            else if(x < 0 && y >= 0)
                op2 = 1;
            else if(x < 0 && y < 0)
                op3 = 1;
            else
                op4 = 1;
            entry << x, y, op1, op2, op3, op4;
            trainingSet.add(entry);
        }

        cout << "Training set: " << endl;
        trainingSet.print();

        cout << "Training..." << endl;
        TrainingSettings settings;
        settings.epochs = -1;
        settings.batch = -1;
        settings.maxError = 0.01;
        TrainingResults results = nn.train(trainingSet, settings);
        cout << "Training results: " << endl;
        results.print();

        nn.save("quad");
    }

    // Using the model
    DataSet testSet = nn.createDataSet();
    auto data = testSet.createEntry();

    cout << "Evaluation: Enter values for X,Y between -100 and 100" << endl;
    cout << "To exit use X or Y equals 0" << endl;
    unsigned int pred;
    while(true)
    {
        cout << "X: ";
        cin >> x;
        cout << "Y: ";
        cin >> y;
        if(x == 0 || y == 0)
            break;
        testSet.clear();
        data << x, y, 0, 0, 0, 0;
        testSet.add(data);
        nn.test(testSet);
        pred = testSet.getOutputMaxIndex(0);
        cout << "Result: " << (pred+1) << "° quad" << endl;
    }

    return 0;
}