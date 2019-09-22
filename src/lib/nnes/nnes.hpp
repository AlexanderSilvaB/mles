#ifndef _NNES_NNES_HPP_
#define _NNES_NNES_HPP_

#include <vector>
#include <string>
#include <memory>
#include <stdarg.h>
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
            ActivationPtr defaultActivation;
            ActivationPtr inputActivation;
            ActivationPtr outputActivation;
            std::vector<unsigned int> layersSize;
            std::vector< ActivationPtr > layersActivation;
            std::vector<Layer> layers;

            std::vector< ActivationPtr > supportedActivations;
            ActivationPtr getActivation(const std::string& name, bool allowDefault = false);
            void init();
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
            void setInputActivation(std::string name = "", ...);
            void setOutputActivation(std::string name = "", ...);
            void setDefaultActivation(std::string name = "", ...);
            void addLayer(unsigned int size, std::string name = "", ...);
            void insertLayer(unsigned int pos, unsigned int size, std::string name = "", ...);
            void changeLayer(unsigned int pos, unsigned int size, std::string name = "", ...);
            void removeLayer(unsigned int pos);

            TrainingResults train(DataSet& trainingSet, const TrainingSettings& settings);
            void test(DataSet& testSet);
            Eigen::VectorXd test(const Eigen::VectorXd& data);

            bool load(const std::string& fileName);
            bool save(const std::string& fileName);
            void build();

            DataSet createDataSet();
    };
}

#endif
