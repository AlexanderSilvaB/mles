#ifndef _MLES_TRAINING_HPP_
#define _MLES_TRAINING_HPP_

#include <iostream>
#include <string>

namespace mles
{
    #define MAX_EPOCHS -1
    #define NO_BATCH -1

    typedef struct TrainingSettings_t
    {
        int epochs;
        double maxError;
        double learningRate;
        int batch;
        int localMinimaLimit;

        TrainingSettings_t()
        {
            epochs = 10000;
            maxError = 0.01;
            learningRate = 0.01;
            batch = -1;
            localMinimaLimit = 10;
        }
    }TrainingSettings;

    typedef struct
    {
        bool finished;
        unsigned int epochs;
        double error;
        double elapsedTime;

        int errorCode; 

        void print()
        {
            if(errorCode != 0)
            {
                std::cout << "Error code: " << errorCode << std::endl;
            }
            else
            {
                std::cout << "Error: " << error << std::endl;
                std::cout << "Epochs: " << epochs << std::endl;
                if(finished)
                    std::cout << "Elapsed time: " << elapsedTime << "s" << std::endl;
            }
        }

    }TrainingResults;
}

#endif
