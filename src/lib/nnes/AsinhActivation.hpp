#ifndef _NNES_ASINH_ACTIVATION_HPP_
#define _NNES_ASINH_ACTIVATION_HPP_

#include <cmath>
#include "Activation.hpp"

namespace nnes
{
    class AsinhActivation : public Activation
    {
        public:
            AsinhActivation() : Activation("asinh")
            {

            }

            ~AsinhActivation()
            {

            }

            double getActivation(double x)
            {
                return asinh(x);
            }

            double getDerivative(double x)
            {
                return 1.0 / sqrt(x*x + 1.0);
            }

            Activation* clone()
            {
                return new AsinhActivation();
            }
    };
}

#endif
