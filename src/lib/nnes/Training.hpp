#ifndef _NNES_TRAINING_HPP_
#define _NNES_TRAINING_HPP_

#include <iostream>

namespace nnes
{
    typedef struct
    {
        unsigned int epochs;
        double error;

        void print()
        {
            std::cout << "Error: " << error << std::endl;
            std::cout << "Epochs: " << epochs << std::endl;
        }

    }TrainingResults;
}

#endif
