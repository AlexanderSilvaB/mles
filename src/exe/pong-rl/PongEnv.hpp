#ifndef _PONG_ENV_HPP_
#define _PONG_ENV_HPP_

#include <mles/QLearning.hpp>
#include "Pong.hpp"
#include <iostream>
#include <bitset>

class PongEnv : public mles::QEnv
{
    private:
        Pong pong;
    public:
        PongEnv() : QEnv()
        {
            pong.setMode(MANUAL);
            reset();
        }

        void toTrain()
        {
            pong.setWinMargin(2);
        }

        void toTest()
        {
            pong.setInterval(30);
            pong.setWinMargin(10);
        }

        int getNumStates()
        {
            return 5;
        }

        int getNumActions()
        {
            return 3;
        }

        int getAction()
        {
            int action = rand() % getNumActions();
            return action;
        }

        int reset()
        {
            pong.reset();
            return getState();
        }

        // State of 3 bits [ vertical distance from ball to paddle(0, 1, 2, 3, 4){3 bits} ]
        int getState()
        {
            float y, bx, by;
            pong.getPlayerPosition(y);
            pong.getBallPosition(bx, by);

            int dD = 0;

            float ddy = y - by;
            if(abs(ddy) < 10)
                dD = 0;
            else if(ddy > 0)
            {
                if(ddy < 50)
                    dD = 1;
                else
                    dD = 2;
            }
            else
            {
                if(ddy > -50)
                    dD = 3;
                else
                    dD = 4;
            }

            int state = dD;

            // std::cout << std::bitset<3>(state) << std::endl;
            return state;
        }

        void onEpoch(int epoch)
        {
            if(epoch % 100 == 0)
            {
                std::cout << "Visualizing train" << std::endl;
                pong.setInterval(30);
            }
            else
                pong.setInterval(0);
        }

        bool ready()
        {
            if(pong.isReady())
                return true;
            pong.update();
            return false;
        }

        bool step(int action, int& nextState, double& reward)
        {
            PongCommands command = NONE;
            switch(action)
            {
                case 0:
                    command = NONE;
                    break;
                case 1:
                    command = UP;
                    break;
                case 2:
                    command = DOWN;
                    break;
                default:
                    break;
            }

            pong.control(command);
            reward = 10 * (double)pong.update();
        
            nextState = getState();

            return pong.isFinished();
        }
};

#endif

