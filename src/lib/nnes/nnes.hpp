#ifndef _NNES_NNES_HPP_
#define _NNES_NNES_HPP_

#include <vector>
#include <string>
#include <Eigen/Dense>
#include "Activation.hpp"
#include "DataSet.hpp"
#include "Training.hpp"
#include "Layer.hpp"

namespace nnes
{
    class NN
    {
        private:
            bool isVerbose;
            unsigned int inputSize, outputSize;
            Activation* defaultActivation;
            Activation* inputActivation;
            Activation* outputActivation;
            std::vector<unsigned int> layersSize;
            std::vector<Activation*> layersActivation;
            std::vector<Layer> layers;
        public:
            NN();
            NN(unsigned int inputSize, unsigned int outputSize);
            virtual ~NN();

            void verbose(bool v);

            void setInputSize(unsigned int size);
            void setOutputSize(unsigned int size);
            void setInputActivation(Activation* activation);
            void setOutputActivation(Activation* activation);
            void setDefaultActivation(Activation* activation);
            void addLayer(unsigned int size, Activation* activation = NULL);
            void insertLayer(unsigned int pos, unsigned int size, Activation* activation = NULL);
            void changeLayer(unsigned int pos, unsigned int size, Activation* activation = NULL);
            void removeLayer(unsigned int pos);

            TrainingResults train(DataSet& trainingSet, unsigned int epochs = 10000, double error = 0.01, double learningRate = 0.1);
            void test(DataSet& testSet);
            Eigen::VectorXd test(const Eigen::VectorXd& data);

            bool load(const std::string& fileName);
            bool save(const std::string& fileName);
            void build();

            DataSet createDataSet();
    };
}

#endif
