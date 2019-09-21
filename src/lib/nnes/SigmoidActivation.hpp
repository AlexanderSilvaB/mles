#ifndef _NNES_SIGMOID_ACTIVATION_HPP_
#define _NNES_SIGMOID_ACTIVATION_HPP_

#include <cmath>
#include "Activation.hpp"

namespace nnes
{
    class SigmoidActivation : public Activation
    {
        public:
            SigmoidActivation() : Activation("sigmoid")
            {

            }

            ~SigmoidActivation()
            {

            }

            double getActivation(double x)
            {
                return 1.0 / (1.0 + exp(-x));
            }

            double getDerivative(double x)
            {
                double sx = getActivation(x);
                return (sx * (1.0 - sx));
            }
    };
}

#endif
