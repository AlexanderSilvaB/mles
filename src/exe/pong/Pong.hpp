#ifndef _PONG_HPP_
#define _PONG_HPP_

#include <opencv2/opencv.hpp>

enum PongModes
{
    GOOD, BEST, MANUAL, EXTERN
};

class Pong
{
    private:
        cv::Mat img;
        bool ready, finished;
        int scoreP1, scoreP2;
        int count, key;
        float baseV;
        PongModes mode;

        float ballX, ballY;
        float p1X, p1Y, p2X, p2Y;
        float wP, hP, bR;
        float p1V, p2V, bVx, bVy;
    public:
        Pong(int width = 500, int height = 400);
        virtual ~Pong();

        void setMode(PongModes mode);
        
        float getPlayerSpeed();
        void setPlayerSpeed(float v);
        
        void getBallPosition(float& x, float& y);
        float getBallSpeed();

        void getPlayerPosition(float& y);
        void getOponentPosition(float& y);

        void getScore(int& p1, int& p2);

        bool isReady();
        bool isFinished();

        void update();
        void reset();
        void finish();
};

#endif
