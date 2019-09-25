#ifndef _MLES_NN_HPP_
#define _MLES_NN_HPP_

#include <vector>
#include <string>
#include <memory>
#include <stdarg.h>
#include <Eigen/Dense>
#include "Activation.hpp"
#include "DataSet.hpp"
#include "Training.hpp"
#include "Layer.hpp"

namespace mles
{
    class NN
    {
        private:
            bool isVerbose;
            unsigned int inputSize, outputSize;
            ActivationPtr defaultActivation;
            ActivationPtr outputActivation;
            std::vector<unsigned int> layersSize;
            std::vector< ActivationPtr > layersActivation;
            std::vector<Layer> layers;
            
            bool transformInput, transformOutput;
            Eigen::VectorXd inputA, inputB;
            Eigen::VectorXd outputA, outputB;

            std::vector< ActivationPtr > supportedActivations;
            ActivationPtr getActivation(const std::string& name, bool allowDefault = false);
            void init();
            void reset();
        public:
            NN();
            NN(unsigned int inputSize, unsigned int outputSize);
            virtual ~NN();

            void verbose(bool v);

            template<class A>
            void registerActivation()
            {
                supportedActivations.push_back(ActivationPtr(new A()));
            }

            void setInputSize(unsigned int size);
            void setOutputSize(unsigned int size);
            void setOutputActivation(std::string name = "", ...);
            void setDefaultActivation(std::string name = "", ...);

            Eigen::VectorXd createInputVector();
            Eigen::VectorXd createOutputVector();

            void setInputTransformation(const Eigen::VectorXd& a, const Eigen::VectorXd& b);
            void setOutputTransformation(const Eigen::VectorXd& a, const Eigen::VectorXd& b);

            void addLayer(unsigned int size, std::string name = "", ...);
            void insertLayer(unsigned int pos, unsigned int size, std::string name = "", ...);
            void changeLayer(unsigned int pos, unsigned int size, std::string name = "", ...);
            void removeLayer(unsigned int pos);

            TrainingResults train(DataSet& trainingSet, const TrainingSettings& settings);
            void test(DataSet& testSet);
            Eigen::VectorXd test(const Eigen::VectorXd& data);

            bool load(const std::string& fileName);
            bool save(const std::string& fileName);
            void print();
            void build();

            DataSet createDataSet();
    };
}

#endif
