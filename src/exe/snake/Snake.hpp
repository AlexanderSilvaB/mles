#ifndef _SNAKE_HPP_
#define _SNAKE_HPP_

#include <opencv2/opencv.hpp>
#include <vector>

enum SnakeModes
{
    MANUAL, EXTERN
};

enum SnakeCommands
{
    STRAIGHT = 0, LEFT, RIGHT
};

class Snake
{
    private:
        cv::Mat img;
        bool ready, finished;
        int score;
        int key, count, lives;
        SnakeModes mode;
        SnakeCommands command;

        int direction;
        int berryX, berryY;
        int headX, headY;
        int lastHeadX, lastHeadY;
        int width, height;
        int size;
        int textLimit;
        int dt;
        
        std::vector< std::pair<int, int> > tail;
        void generateBerry();
    public:
        Snake(int width = 30, int height = 20, int size = 10);
        virtual ~Snake();

        void resize(int width = 30, int height = 20, int size = 10);

        void setMode(SnakeModes mode);
        void control(SnakeCommands commands);

        void getHeadPosition(int& x, int& y);
        void getBerryPosition(int& x, int& y);
        int getScore();
        int getLives();
        int getDirection();
        int getFoodDirection();
        void getDanger(bool& left, bool& top, bool& right, bool& bottom);

        int getWidth();
        int getHeight();

        void setInterval(int dt);

        bool isReady();
        bool isFinished();

        int update();
        void reset();
        void finish();

};

#endif
