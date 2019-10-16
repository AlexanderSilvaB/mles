#ifndef _SNAKE_ENV_HPP_
#define _SNAKE_ENV_HPP_

#include <mles/QLearning.hpp>
#include "Snake.hpp"
#include <iostream>
#include <bitset>

class SnakeEnv : public mles::QEnv
{
    private:
        Snake snake;
    public:
        SnakeEnv() : QEnv()
        {
            snake = Snake(20, 20, 10);
            snake.setMode(EXTERN);
            reset();
        }

        void toTrain()
        {
            snake.resize(10, 10, 10);
            snake.setInterval(0);
        }

        void toTest()
        {
            snake.resize(50, 50, 10);
            snake.setInterval(30);
            snake.setLives(3);
        }

        int getNumStates()
        {
            return 1 + 0b11111111;
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
            snake.reset();
            return getState();
        }

        // State of 8 bits [ direction(0, 1, 2, 3){2 bits}, foodDirection(0, 1, 2, 3){2 bits}, danger(left, top, right, bottom){4 bits}]
        int getState()
        {
            int d, bd;
            bool left, top, right, bottom;
            d = snake.getDirection();
            bd = snake.getFoodDirection();
            snake.getDanger(left, top, right, bottom);
            int state = d;
            state |= (bd & 0b11) << 2;
            state |= (left & 0x1) << 4;
            state |= (top & 0x1) << 5;
            state |= (right & 0x1) << 6;
            state |= (bottom & 0x1) << 7;
            return state;
        }

        void onEpoch(int epoch)
        {
            if(epoch % 2500 == 0)
            {
                std::cout << "Evaluating current training" << std::endl;
                snake.setInterval(10);
                snake.setLives(1);
                snake.resize(50, 50, 10);
            }
            else
            {
                snake.setInterval(0);
                snake.setLives(3);
                snake.resize(10, 10, 10);
            }
        }

        bool ready()
        {
            if(snake.isReady())
                return true;
            snake.update();
            return false;
        }

        bool step(int action, int& nextState, double& reward)
        {
            SnakeCommands command = STRAIGHT;
            switch(action)
            {
                case 0:
                    command = STRAIGHT;
                    break;
                case 1:
                    command = LEFT;
                    break;
                case 2:
                    command = RIGHT;
                    break;
                default:
                    break;
            }

            snake.control(command);
            reward = 10 * (double)snake.update();
        
            nextState = getState();

            return snake.isFinished();
        }
};

#endif

