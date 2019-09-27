#ifndef _MLES_QLEARNING_HPP_
#define _MLES_QLEARNING_HPP_

#include <Eigen/Dense>
#include <map>
#include <string>
#include "Training.hpp"

namespace mles
{
    class QEnv
    {
        public:
            QEnv();
            virtual ~QEnv();
            virtual void toTrain();
            virtual void toTest();
            virtual int getNumStates();
            virtual int getNumActions();
            virtual int getAction();
            virtual int reset();
            virtual bool ready();
            virtual void onEpoch(int epoch);
            virtual bool step(int action, int& nextState, double& reward);
            virtual void onDone();
    };

    class QLearning
    {
        private:
            QEnv *env;
            Eigen::MatrixXd Q;
            double explorationFactor;
            double discountFactor;
            double exploitationFactor;
        public:
            QLearning(QEnv* env);
            virtual ~QLearning();
            
            void setDiscountFactor(double factor);
            void setExplorationFactor(double factor);
            void setExploitationFactor(double factor);

            void train(const TrainingSettings& settings);
            void run();

            bool load(const std::string& fileName);
            bool save(const std::string& fileName);
    
    };
}

#endif
