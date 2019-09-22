#ifndef _NNES_RELU_ACTIVATION_HPP_
#define _NNES_RELU_ACTIVATION_HPP_

#include <cmath>
#include "Activation.hpp"

namespace nnes
{
    class ReluActivation : public Activation
    {
        public:
            ReluActivation() : Activation("relu")
            {

            }

            ~ReluActivation()
            {

            }

            double getActivation(double x)
            {
                if(x > 0)
                    return x;
                return 0;
            }

            double getDerivative(double x)
            {
                if(x <= 0)
                    return 0;
                return 1;
            }

            Activation* clone()
            {
                return new ReluActivation();
            }
    };
}

#endif
