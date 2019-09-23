#ifndef _NNES_SOFTMAX_ACTIVATION_HPP_
#define _NNES_SOFTMAX_ACTIVATION_HPP_

#include <cmath>
#include <iostream>
#include "Activation.hpp"

namespace nnes
{
    class SoftMaxActivation : public Activation
    {
        public:
            SoftMaxActivation() : Activation("softmax")
            {

            }

            ~SoftMaxActivation()
            {

            }

            double getActivation(double x)
            {
                return exp(x);
            }

            double getDerivative(double x)
            {
                double sx = getActivation(x);
                return (sx * (1.0 - sx));
            }

            void getActivation(Eigen::MatrixXd& x, Eigen::MatrixXd& y)
            {
                y.resize(x.rows(), x.cols());
                double m = x.maxCoeff();
                for(int i = 0; i < x.rows(); i++)
                {
                    for(int j = 0; j < x.cols(); j++)
                        y(i,j) = this->getActivation(x(i,j) - m);
                }
                double s = y.sum();
                y /= s;
            }

            void getDerivative(Eigen::MatrixXd& x, Eigen::MatrixXd& y)
            {
                Eigen::MatrixXd d;
                getActivation(x, d);
                Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(d.rows(), d.cols());
                y = d.cwiseProduct(ones - d);
            }

            Activation* clone()
            {
                return new SoftMaxActivation();
            }
    };
}

#endif
