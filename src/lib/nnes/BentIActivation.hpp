#ifndef _NNES_BENTI_ACTIVATION_HPP_
#define _NNES_BENTI_ACTIVATION_HPP_

#include <cmath>
#include "Activation.hpp"

namespace nnes
{
    class BentIActivation : public Activation
    {
        public:
            BentIActivation() : Activation("benti")
            {

            }

            ~BentIActivation()
            {

            }

            double getActivation(double x)
            {
                return (0.5 * (sqrt(x*x + 1) - 1)) + x;
            }

            double getDerivative(double x)
            {
                return (x / (2.0 * sqrt(x*x + 1))) + 1.0;
            }

            Activation* clone()
            {
                return new BentIActivation();
            }
    };
}

#endif
