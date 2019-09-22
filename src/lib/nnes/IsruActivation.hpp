#ifndef _NNES_ISRU_ACTIVATION_HPP_
#define _NNES_ISRU_ACTIVATION_HPP_

#include <cmath>
#include "Activation.hpp"

namespace nnes
{
    class IsruActivation : public Activation
    {
        private:
            double a;
        public:
            IsruActivation() : Activation("isru", 1)
            {
                this->a = 1.0;
            }

            ~IsruActivation()
            {

            }

            double getActivation(double x)
            {
                return x / sqrt(1.0 + a*x*x);
            }

            double getDerivative(double x)
            {
                double y = 1 / sqrt(1.0 + a*x*x);
                return y*y*y;
            }

            Activation* clone()
            {
                IsruActivation* isru = new IsruActivation();
                isru->a = a;
                return isru;
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
