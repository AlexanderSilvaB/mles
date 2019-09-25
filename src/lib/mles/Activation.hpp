#ifndef _MLES_ACTIVATION_HPP_
#define _MLES_ACTIVATION_HPP_

#include <Eigen/Dense>
#include <string>
#include <ostream>
#include <istream>
#include <memory>

namespace mles
{
    class Activation;
    typedef std::shared_ptr<Activation> ActivationPtr;

    class Activation
    {
        private:
            std::string name;
            int parametersSize;
        public:
            Activation(const std::string& name, int parametersSize = 0)
            {
                this->name = name;
                this->parametersSize = parametersSize;
            }

            virtual ~Activation()
            {
                
            }

            std::string getName()
            {
                return name;
            }

            int getParametersSize()
            {
                return parametersSize;
            }

            bool is(const std::string& name)
            {
                return name == this->name;
            }

            virtual void getActivation(Eigen::MatrixXd& x, Eigen::MatrixXd& y)
            {
                y.resize(x.rows(), x.cols());
                for(int i = 0; i < x.rows(); i++)
                {
                    for(int j = 0; j < x.cols(); j++)
                        y(i,j) = this->getActivation(x(i,j));
                }
            }

            virtual void getDerivative(Eigen::MatrixXd& x, Eigen::MatrixXd& y)
            {
                y.resize(x.rows(), x.cols());
                for(int i = 0; i < x.rows(); i++)
                {
                    for(int j = 0; j < x.cols(); j++)
                        y(i,j) = this->getDerivative(x(i,j));
                }
            }

            virtual double getActivation(double x)
            {
                return x;
            }

            virtual double getDerivative(double x)
            {
                return 0;
            }

            virtual Activation* clone()
            {
                return new Activation(name);
            }

            virtual void load(std::istream& f)
            {

            }

            virtual void write(std::ostream& f)
            {

            }
    };
}

#endif
