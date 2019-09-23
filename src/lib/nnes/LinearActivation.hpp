#ifndef _NNES_LINEAR_ACTIVATION_HPP_
#define _NNES_LINEAR_ACTIVATION_HPP_

#include <cmath>
#include <iostream>
#include "Activation.hpp"

namespace nnes
{
    class LinearActivation : public Activation
    {
        private:
            double a;
        public:
            LinearActivation() : Activation("linear", 1)
            {
                a = 1.0;
            }

            ~LinearActivation()
            {

            }

            double getActivation(double x)
            {
                return a*x;
            }

            double getDerivative(double x)
            {
                return a;
            }

            Activation* clone()
            {
                LinearActivation* linear = new LinearActivation();
                linear->a = a;
                return linear;
            }

            void load(std::istream& f)
            {
                f >> a;
            }

            void write(std::ostream& f)
            {
                f << a;
            }
    };
}

#endif
