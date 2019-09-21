#ifndef _NNES_DATA_SET_HPP_
#define _NNES_DATA_SET_HPP_

#include <vector>
#include <Eigen/Dense>

namespace nnes
{
    class DataSet
    {
        private:
            unsigned int inputSize, outputSize;
            std::vector < Eigen::VectorXd > inputs;
            std::vector < Eigen::VectorXd > outputs;
        public:
            DataSet(const DataSet& dataSet);
            DataSet(unsigned int inputSize, unsigned int outputSize);
            virtual ~DataSet();

            Eigen::VectorXd createInput();
            Eigen::VectorXd createOutput();
            Eigen::VectorXd createEntry();

            void add(const Eigen::VectorXd& input, const Eigen::VectorXd& output);
            void add(const Eigen::VectorXd& entry);

            unsigned int size();
            Eigen::VectorXd& getInput(unsigned int pos);
            Eigen::VectorXd& getOutput(unsigned int pos);

            DataSet toTestSet();

            void print();
    };
}

#endif
