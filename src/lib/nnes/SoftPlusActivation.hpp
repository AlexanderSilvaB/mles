#ifndef _NNES_SOFTPLUS_ACTIVATION_HPP_
#define _NNES_SOFTPLUS_ACTIVATION_HPP_

#include <cmath>
#include "Activation.hpp"

namespace nnes
{
    class SoftPlusActivation : public Activation
    {
        public:
            SoftPlusActivation() : Activation("softplus")
            {

            }

            ~SoftPlusActivation()
            {

            }

            double getActivation(double x)
            {
                return log(1 + exp(x));
            }

            double getDerivative(double x)
            {
                return 1.0 / (1.0 + exp(-x));
            }

            Activation* clone()
            {
                return new SoftPlusActivation();
            }
    };
}

#endif
