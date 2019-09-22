#ifndef _NNES_TANH_ACTIVATION_HPP_
#define _NNES_TANH_ACTIVATION_HPP_

#include <cmath>
#include "Activation.hpp"

namespace nnes
{
    class TanhActivation : public Activation
    {
        public:
            TanhActivation() : Activation("tanh")
            {

            }

            ~TanhActivation()
            {

            }

            double getActivation(double x)
            {
                return tanh(x);
            }

            double getDerivative(double x)
            {
                double y = getActivation(x);
                return 1.0 - y*y;
            }

            Activation* clone()
            {
                return new TanhActivation();
            }
    };
}

#endif
