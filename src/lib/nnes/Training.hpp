#ifndef _NNES_TRAINING_HPP_
#define _NNES_TRAINING_HPP_

#include <iostream>

namespace nnes
{
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
            learningRate = 0.1;
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

        void print()
        {
            std::cout << "Error: " << error << std::endl;
            std::cout << "Epochs: " << epochs << std::endl;
            if(finished)
                std::cout << "Elapsed time: " << elapsedTime << "s" << std::endl;
        }

    }TrainingResults;
}

#endif
