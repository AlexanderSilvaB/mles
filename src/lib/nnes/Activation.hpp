#ifndef _NNES_ACTIVATION_HPP_
#define _NNES_ACTIVATION_HPP_

#include <Eigen/Dense>
#include <string>

namespace nnes
{
    class Activation
    {
        private:
            std::string name;
        public:
            Activation(const std::string& name)
            {
                this->name = name;
            }

            virtual ~Activation()
            {
                
            }

            std::string getName()
            {
                return name;
            }

            virtual bool is(const std::string& name)
            {
                return name == this->name;
            }

            virtual double getActivation(double x)
            {
                return x;
            }

            virtual double getDerivative(double x)
            {
                return 0;
            }

            void getActivation(Eigen::MatrixXd& x, Eigen::MatrixXd& y)
            {
                y.resize(x.rows(), x.cols());
                for(int i = 0; i < x.rows(); i++)
                {
                    for(int j = 0; j < x.cols(); j++)
                        y(i,j) = this->getActivation(x(i,j));
                }
            }

            void getDerivative(Eigen::MatrixXd& x, Eigen::MatrixXd& y)
            {
                y.resize(x.rows(), x.cols());
                for(int i = 0; i < x.rows(); i++)
                {
                    for(int j = 0; j < x.cols(); j++)
                        y(i,j) = this->getDerivative(x(i,j));
                }
            }
    };
}

#endif
