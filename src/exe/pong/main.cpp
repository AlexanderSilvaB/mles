#include <iostream>
#include <nnes/nnes.hpp>
#include "Pong.hpp"

using namespace std;
using namespace nnes;

int main(int argc, char *argv[])
{
    std::srand((unsigned int) time(0));

    Pong pong;
    NN nn;
    float py, bx, by, bv, v;


    if(!nn.load("pong.nn"))
    {
        pong.setMode(BEST);

        // Model creation
        cout << "Creating model" << endl;
        nn.setDefaultActivation("relu");
        nn.setOutputActivation("sigmoid");
        nn.setInputSize(3);
        nn.setOutputSize(1);
        nn.verbose(true); 
        nn.addLayer(3);
        nn.addLayer(3);
        nn.build();

        // Setting the data transformation
        auto inputA = nn.createInputVector();
        auto inputB = nn.createInputVector();
        inputA(0) = 1.0/500.0;
        inputA(1) = 1.0/500.0;
        inputA(2) = 1.0/800.0;
        inputB.setZero();
        nn.setInputTransformation(inputA, inputB);

        auto outputA = nn.createOutputVector();
        auto outputB = nn.createOutputVector();
        outputA(0) = 1.0/1500.0;
        outputB(0) = 0.5;
        nn.setOutputTransformation(outputA, outputB);
        
        // Data creation
        cout << "Creating training set" << endl;
        DataSet trainingSet = nn.createDataSet();
        auto entry = trainingSet.createEntry();

        int s1 = 0, s2 = 0;
        int i = 0;
        while(true)
        {
            pong.update();
            if(!pong.isReady())
                continue;

            pong.getScore(s1, s2);
            if(s1 > 0 || s2 > 0)
                break;

            if(i % 2 == 0)
            {
                pong.getBallPosition(bx, by);
                pong.getPlayerPosition(py);
                bv = pong.getBallSpeed();
                v = pong.getPlayerSpeed();
                entry << by, py, bv, v;
                trainingSet.add(entry);
            }

            i++;
        }

        pong.finish();

        cout << "Training set size: " << trainingSet.size() << endl;
        trainingSet.print();

        cout << "Training..." << endl;
        TrainingSettings settings;
        settings.epochs = MAX_EPOCHS;
        settings.batch = NO_BATCH;
        settings.maxError = 0.003;
        TrainingResults results = nn.train(trainingSet, settings);
        cout << "Training results: " << endl;
        results.print();

        nn.save("pong.nn");
    }

    pong.setMode(EXTERN);
    pong.reset();

    DataSet testSet = nn.createDataSet();
    auto data = testSet.createEntry();

    while(true)
    {
        pong.update();
        if(!pong.isReady())
            continue;

        pong.getBallPosition(bx, by);
        pong.getPlayerPosition(py);
        bv = pong.getBallSpeed();

        testSet.clear();
        data << by, py, bv, 0;
        testSet.add(data);

        nn.test(testSet);
        v = testSet.getOutput(0)[0];
        pong.setPlayerSpeed(v);

        cout << "Selected speed: " << v << endl;
    }
    return 0;
}