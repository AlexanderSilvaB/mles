#ifndef _MLES_ATAN_ACTIVATION_HPP_
#define _MLES_ATAN_ACTIVATION_HPP_

#include <cmath>
#include "Activation.hpp"

namespace mles
{
    class AtanActivation : public Activation
    {
        public:
            AtanActivation() : Activation("atan")
            {

            }

            ~AtanActivation()
            {

            }

            double getActivation(double x)
            {
                return atan(x);
            }

            double getDerivative(double x)
            {
                return 1.0 / (x*x + 1.0);
            }

            Activation* clone()
            {
                return new AtanActivation();
            }
    };
}

#endif
