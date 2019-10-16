#ifndef _MLES_DATA_SET_HPP_
#define _MLES_DATA_SET_HPP_

#include <vector>
#include <Eigen/Dense>
#include <string>
#include <map>

namespace mles
{
    class DataSet
    {
        private:
            unsigned int inputSize, outputSize;
            std::vector < Eigen::VectorXd > inputs;
            std::vector < Eigen::VectorXd > outputs;
            std::vector < std::string > inputHeader;
            std::vector < std::string > outputHeader;
            std::map < std::string, std::vector < std::string > > classMap;
            bool isNumeric(const std::string& text, double& value);
        public:
            DataSet();
            DataSet(const DataSet& dataSet);
            DataSet(unsigned int inputSize, unsigned int outputSize);
            virtual ~DataSet();

            unsigned int getInputSize();
            unsigned int getOutputSize();

            void split(DataSet& trainingSet, DataSet& testSet, double ratio = 0.8);

            Eigen::VectorXd createInput();
            Eigen::VectorXd createOutput();
            Eigen::VectorXd createEntry();

            void add(const Eigen::VectorXd& input, const Eigen::VectorXd& output);
            void add(const Eigen::VectorXd& entry);
            void clear();

            bool load(const std::string& fileName);
            bool save(const std::string& fileName);

            bool toCSV(const std::string& fileName);
            bool fromCSV(const std::string& fileName, bool header = true, int chucksize = -1);

            unsigned int size();
            Eigen::VectorXd& getInput(unsigned int pos);
            Eigen::VectorXd& getOutput(unsigned int pos);
            unsigned int getOutputMaxIndex(unsigned int pos);

            DataSet toTestSet();

            void binarize(double threshold = 0.5);
            void classify();
            void binarizeToMax();

            void print(bool onlyOutput = false);
    };
}

#endif
