#ifndef _NNES_LAYER_HPP_
#define _NNES_LAYER_HPP_

#include <Eigen/Dense>
#include "Activation.hpp"
#include <fstream>

namespace nnes
{
    class Layer
    {
        private:
            unsigned int id;
            bool isOutput;
            Eigen::MatrixXd weight;
            Activation *activation;

            Eigen::MatrixXd z, dz, A, delta;

        public:
            Layer();
            Layer(unsigned int id, unsigned int inputSize, unsigned int outputSize, Activation *activation, bool isOutput);
            virtual ~Layer();

            double error(const Eigen::MatrixXd& y);
            Eigen::MatrixXd forward(const Eigen::MatrixXd& x);
            Eigen::MatrixXd backward(const Eigen::MatrixXd& y, Layer *layer);
            Eigen::MatrixXd update(double learningRate, const Eigen::MatrixXd& a);

            void load(std::ifstream& f);
            void write(std::ofstream& f);

    };
}

#endif
